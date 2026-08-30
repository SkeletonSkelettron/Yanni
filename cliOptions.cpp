#include "cliOptions.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

CliOptions gCli;
std::string gEffectiveConfig;

void PrintCliHelp(const char *exeName) {
  printf(
      "\n%s -- netConfig.json-ის პარამეტრების გადაფარვა ბრძანების ველიდან\n\n"
      "  --set KEY=VALUE [KEY=VALUE ...]   ერთი ან რამდენიმე პარამეტრი\n"
      "  --topology-only                   ქსელი აჩვენე და გამოდი\n"
      "  -q, --quiet                       ტრენინგის მიმდინარეობა კონსოლში\n"
      "                                    არ დაბეჭდო (yanni.log ივსება)\n"
      "  -h, --help                        ეს ტექსტი\n\n"
      "KEY ორი სახისაა:\n"
      "  Name              ზედა დონის პარამეტრი (LearningRate, ThreadCount...)\n"
      "  <ლეიერი>.Name     კონკრეტული ლეიერის პარამეტრი, ნომერი 0-დან\n\n"
      "მაგალითები:\n"
      "  %s --topology-only\n"
      "  %s --set 1.Size=800 2.Size=300\n"
      "  %s --set LearningRate=0.3 1.DropuOutSize=0.1 --topology-only\n"
      "  %s --set 1.ActivationFunction=ReLU --set ThreadCount=4\n\n"
      "netConfig.json არ იცვლება -- გადაფარვა მხოლოდ ამ გაშვებაზე მოქმედებს.\n\n",
      exeName, exeName, exeName, exeName, exeName);
}

CliOptions ParseCli(int argc, char **argv) {
  CliOptions o;
  for (int i = 1; i < argc; i++) {
    const std::string a = argv[i];

    if (a == "-h" || a == "--help") {
      o.help = true;
    } else if (a == "--topology-only" || a == "--topology_only") {
      o.topologyOnly = true;
    } else if (a == "--quiet" || a == "-q") {
      o.quiet = true;
    } else if (a == "--set") {
      // ყველა მომდევნო KEY=VALUE ერთ --set-ს ეკუთვნის, სანამ ახალი
      // დროშა არ დაიწყება -- რამდენიმე ლეიერი ერთ ბრძანებაში რომ ეტეოდეს
      int taken = 0;
      while (i + 1 < argc && argv[i + 1][0] != '-') {
        const std::string kv = argv[++i];
        const size_t eq = kv.find('=');
        if (eq == std::string::npos || eq == 0 || eq + 1 == kv.size()) {
          printf("!! --set expects KEY=VALUE, got '%s'\n", kv.c_str());
          o.bad = true;
        } else {
          o.sets.emplace_back(kv.substr(0, eq), kv.substr(eq + 1));
        }
        taken++;
      }
      if (taken == 0) {
        printf("!! --set with no KEY=VALUE after it\n");
        o.bad = true;
      }
    } else {
      printf("!! unknown argument '%s'\n", a.c_str());
      o.bad = true;
    }
  }
  return o;
}

// არსებული მნიშვნელობის ტიპს ვინარჩუნებთ: JSON-ში Size მთელია,
// LearningRate წილადი, ActivationFunction სტრიქონი. ტიპის შეცვლა
// ჩუმად გატეხავდა initNeUnetFromJson-ის შემდგომ წაკითხვას.
static bool AssignPreservingType(nlohmann::json &slot, const std::string &text,
                                 const std::string &key) {
  try {
    if (slot.is_string()) {
      slot = text;
    } else if (slot.is_boolean()) {
      slot = (text == "true" || text == "1");
    } else if (slot.is_number_integer() || slot.is_number_unsigned()) {
      size_t pos = 0;
      long v = std::stol(text, &pos);
      if (pos != text.size())
        throw std::invalid_argument("trailing");
      slot = v;
    } else if (slot.is_number_float()) {
      size_t pos = 0;
      double v = std::stod(text, &pos);
      if (pos != text.size())
        throw std::invalid_argument("trailing");
      slot = v;
    } else {
      printf("!! %s: unsupported value type in netConfig.json\n", key.c_str());
      return false;
    }
  } catch (...) {
    printf("!! %s: '%s' is not a valid value for this parameter\n", key.c_str(),
           text.c_str());
    return false;
  }
  return true;
}

static void PrintKeys(const nlohmann::json &json) {
  printf("   top-level: ");
  for (auto it = json.begin(); it != json.end(); ++it)
    if (it.key() != "Layers")
      printf("%s ", it.key().c_str());
  printf("\n   per-layer: ");
  if (json.contains("Layers") && !json["Layers"].empty())
    for (auto it = json["Layers"][0].begin(); it != json["Layers"][0].end();
         ++it)
      printf("%s ", it.key().c_str());
  printf("\n");
}

bool ApplyOverrides(nlohmann::json &json, const CliOptions &opt) {
  if (opt.sets.empty())
    return true;

  bool ok = true;
  printf("\n=== overrides ===\n");
  for (const auto &kv : opt.sets) {
    const std::string &key = kv.first;
    const std::string &val = kv.second;

    nlohmann::json *slot = nullptr;
    const size_t dot = key.find('.');

    if (dot == std::string::npos) {
      if (!json.contains(key)) {
        printf("!! unknown parameter '%s'\n", key.c_str());
        PrintKeys(json);
        ok = false;
        continue;
      }
      slot = &json[key];
    } else {
      const std::string idxText = key.substr(0, dot);
      const std::string name = key.substr(dot + 1);
      char *end = nullptr;
      const long idx = strtol(idxText.c_str(), &end, 10);
      if (*end != '\0') {
        printf("!! '%s' is not a layer number\n", idxText.c_str());
        ok = false;
        continue;
      }
      if (!json.contains("Layers") || idx < 0 ||
          idx >= (long)json["Layers"].size()) {
        printf("!! layer %ld does not exist (network has %zu layers, 0..%zu)\n",
               idx, json["Layers"].size(), json["Layers"].size() - 1);
        ok = false;
        continue;
      }
      if (!json["Layers"][idx].contains(name)) {
        printf("!! layer %ld has no parameter '%s'\n", idx, name.c_str());
        PrintKeys(json);
        ok = false;
        continue;
      }
      slot = &json["Layers"][idx][name];
    }

    const std::string before = slot->dump();
    if (!AssignPreservingType(*slot, val, key)) {
      ok = false;
      continue;
    }
    printf("  %-24s %12s -> %s\n", key.c_str(), before.c_str(),
           slot->dump().c_str());
  }
  printf("\n");
  return ok;
}
