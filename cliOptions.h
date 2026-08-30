#ifndef CLIOPTIONS_H
#define CLIOPTIONS_H

#include <nlohmann/json.hpp>
#include <string>
#include <utility>
#include <vector>

// ბრძანების ველიდან მიღებული პარამეტრები.
//
// netConfig.json რჩება ბაზისურ კონფიგად; დროშები მას გადააფარებენ.
// ასე ერთი და იმავე ფაილიდან შეიძლება ათეული ექსპერიმენტის გაშვება
// სკრიპტიდან, JSON-ის ხელით რედაქტირების გარეშე.
struct CliOptions {
  bool topologyOnly = false; // ქსელი აჩვენე და გამოდი, ტრენინგის გარეშე
  bool quiet = false;        // ტრენინგის მიმდინარეობა კონსოლში არ დაბეჭდო
  bool help = false;
  bool bad = false; // პარსინგის შეცდომა
  // key -> value. key არის "Name" (ზედა დონე) ან "<ლეიერი>.<Name>"
  std::vector<std::pair<std::string, std::string>> sets;
};

extern CliOptions gCli;

// netConfig.json გადაფარვების გამოყენების შემდეგ. ლოგში მთლიანად იწერება,
// რომ ყოველი გაშვება თვითაღწერადი იყოს -- ვერც ერთი პარამეტრი ვერ
// გამომრჩება, რადგან ეს თავად კონფიგია და არა მისი ხელით გადაწერა.
extern std::string gEffectiveConfig;

CliOptions ParseCli(int argc, char **argv);
void PrintCliHelp(const char *exeName);

// json-ს ადგილზე ცვლის. აბრუნებს false-ს, თუ რომელიმე გასაღები
// ან მნიშვნელობა არასწორია (შეცდომა უკვე დაბეჭდილია).
bool ApplyOverrides(nlohmann::json &json, const CliOptions &opt);

#endif // CLIOPTIONS_H
