#include "readMnist.h"
#include <filesystem>
#include <fstream>
#include <iostream>
#include <cctype>
#include <cstdlib>
#include <unistd.h>
#include <vector>

namespace fs = std::filesystem;

unsigned int in(std::ifstream &icin, unsigned int size) {
  unsigned int ans = 0;
  for (long int i = 0; i < size; i++) {
    unsigned char x;
    icin.read((char *)&x, 1);
    unsigned int temp = x;
    ans <<= 8;
    ans += temp;
  }
  return ans;
}

static fs::path DataDir() {
  return fs::read_symlink("/proc/self/exe").parent_path() / "mnist";
}

// ცნობილი დატასეტები. ახლის დამატება ერთი ჩანაწერია.
static const DataSetInfo kKnown[] = {
    {"MNIST", "train-images.idx3-ubyte", "train-labels.idx1-ubyte",
     "t10k-images.idx3-ubyte", "t10k-labels.idx1-ubyte", false, 10},
    {"EMNIST digits", "emnist-digits-train-images-idx3-ubyte",
     "emnist-digits-train-labels-idx1-ubyte",
     "emnist-digits-test-images-idx3-ubyte",
     "emnist-digits-test-labels-idx1-ubyte", true, 10},
    {"EMNIST letters", "emnist-letters-train-images-idx3-ubyte",
     "emnist-letters-train-labels-idx1-ubyte",
     "emnist-letters-test-images-idx3-ubyte",
     "emnist-letters-test-labels-idx1-ubyte", true, 26},
    {"EMNIST balanced", "emnist-balanced-train-images-idx3-ubyte",
     "emnist-balanced-train-labels-idx1-ubyte",
     "emnist-balanced-test-images-idx3-ubyte",
     "emnist-balanced-test-labels-idx1-ubyte", true, 47},
};

// ერთი და იგივე დატასეტი ორი სახელით ვრცელდება: MNIST წერტილით
// (train-images.idx3-ubyte), EMNIST ტირით (…-train-images-idx3-ubyte).
// გამყოფს idx-ის წინ ვცვლით და ორივეს ვცდით -- მხოლოდ ამ ორს.
static std::string FlipSeparator(const std::string &name) {
  const size_t p = name.rfind("idx");
  if (p == std::string::npos || p == 0)
    return name;
  std::string alt = name;
  if (alt[p - 1] == '-')
    alt[p - 1] = '.';
  else if (alt[p - 1] == '.')
    alt[p - 1] = '-';
  else
    return name;
  return alt;
}

// არსებული ფაილის გზა, ან ცარიელი, თუ ვერც ერთი ვარიანტი ვერ მოიძებნა
static fs::path Resolve(const fs::path &dir, const std::string &name) {
  if (fs::exists(dir / name))
    return dir / name;
  const std::string alt = FlipSeparator(name);
  if (alt != name && fs::exists(dir / alt))
    return dir / alt;
  return {};
}

// IDX სურათების სათაურიდან ჩანაწერების რაოდენობა; 0 თუ ფაილი ვერ წაიკითხა
static long CountOf(const fs::path &p) {
  std::ifstream f(p, std::ios::binary);
  if (!f.is_open())
    return 0;
  in(f, 4);
  return (long)in(f, 4);
}

std::vector<DataSetInfo> AvailableDataSets() {
  std::vector<DataSetInfo> found;
  const fs::path dir = DataDir();
  for (const DataSetInfo &d : kKnown) {
    if (!Resolve(dir, d.trainImages).empty() &&
        !Resolve(dir, d.trainLabels).empty() &&
        !Resolve(dir, d.testImages).empty() &&
        !Resolve(dir, d.testLabels).empty())
      found.push_back(d);
  }
  return found;
}

std::string DataSetSlug(const std::string &name) {
  std::string out;
  for (char c : name)
    out += (c == ' ') ? '-' : (char)std::tolower((unsigned char)c);
  return out;
}

static void PrintSets(const std::vector<DataSetInfo> &sets,
                      const fs::path &dir) {
  for (size_t i = 0; i < sets.size(); i++)
    printf("  %zu) %-16s %-16s %7ld train / %6ld test, %d classes\n", i + 1,
           DataSetSlug(sets[i].name).c_str(), sets[i].name.c_str(),
           CountOf(Resolve(dir, sets[i].trainImages)),
           CountOf(Resolve(dir, sets[i].testImages)), sets[i].classCount);
}

DataSetInfo ChooseDataSet(const std::string &requested) {
  std::vector<DataSetInfo> sets = AvailableDataSets();

  if (sets.empty()) {
    std::cout << "no dataset found in " << DataDir() << std::endl;
    return DataSetInfo{"", "", "", "", "", false, 0};
  }
  const fs::path dir = DataDir();

  if (!requested.empty()) {
    // რიგითი ნომერი -- იგივე, რაც სიაში ჩანს
    char *end = nullptr;
    const long n = strtol(requested.c_str(), &end, 10);
    if (*end == '\0' && n >= 1 && n <= (long)sets.size()) {
      std::cout << "dataset: " << sets[n - 1].name << std::endl;
      return sets[n - 1];
    }

    // სახელი: ჯერ ზუსტი დამთხვევა, მერე ერთმნიშვნელოვანი პრეფიქსი.
    // "emnist" ორივე EMNIST-ს დაემთხვევა, ამიტომ ბუნდოვნებას ვწყვეტთ
    // და არ ვირჩევთ თვითნებურად
    const std::string want = DataSetSlug(requested);
    int exact = -1, prefix = -1, prefixCount = 0;
    for (size_t i = 0; i < sets.size(); i++) {
      const std::string slug = DataSetSlug(sets[i].name);
      if (slug == want)
        exact = (int)i;
      else if (slug.compare(0, want.size(), want) == 0) {
        prefix = (int)i;
        prefixCount++;
      }
    }
    if (exact >= 0) {
      std::cout << "dataset: " << sets[exact].name << std::endl;
      return sets[exact];
    }
    if (prefixCount == 1) {
      std::cout << "dataset: " << sets[prefix].name << std::endl;
      return sets[prefix];
    }
    if (prefixCount > 1)
      printf("\n!! --dataset '%s' matches more than one:\n\n",
             requested.c_str());
    else
      printf("\n!! --dataset '%s' matches nothing. available:\n\n",
             requested.c_str());
    PrintSets(sets, dir);
    printf("\n");
    exit(1);
  }

  if (sets.size() == 1) {
    std::cout << "dataset: " << sets[0].name << std::endl;
    return sets[0];
  }

  std::cout << "\nseveral datasets found:\n" << std::endl;
  PrintSets(sets, dir);

  // არაინტერაქტიულ გაშვებაზე კითხვა უაზროა და პროგრამა გაიჭედებოდა
  if (!isatty(fileno(stdin))) {
    std::cout << "\n  (not a terminal, using " << sets[0].name << ")"
              << std::endl;
    return sets[0];
  }

  while (true) {
    std::cout << "\nwhich one? [1-" << sets.size() << "]: " << std::flush;
    std::string line;
    if (!std::getline(std::cin, line)) { // EOF -- პირველზე ვჩერდებით
      std::cout << sets[0].name << std::endl;
      return sets[0];
    }
    try {
      size_t n = std::stoul(line);
      if (n >= 1 && n <= sets.size())
        return sets[n - 1];
    } catch (...) {
    }
    std::cout << "  enter a number between 1 and " << sets.size() << std::endl;
  }
}

void ReadMNISTMod(std::vector<std::vector<float>> &images,
                  std::vector<int> &labels, bool train,
                  const DataSetInfo &ds) {
  if (ds.classCount == 0)
    return;

  const fs::path dir = DataDir();
  const fs::path imgPath = Resolve(dir, train ? ds.trainImages : ds.testImages);
  const fs::path labPath = Resolve(dir, train ? ds.trainLabels : ds.testLabels);

  std::ifstream icin(imgPath, std::ios::binary);
  // შემოწმება სათაურის წაკითხვამდე: დახურული ნაკადიდან in() ნულებს
  // აბრუნებდა და num/rows/cols ნაგვით ივსებოდა
  if (!icin.is_open()) {
    std::cout << "cannot open " << imgPath << std::endl;
    return;
  }
  unsigned int num, rows, cols;
  in(icin, 4), num = in(icin, 4), rows = in(icin, 4), cols = in(icin, 4);

  std::vector<float> img(rows * cols);
  images.reserve(images.size() + num);
  for (long int i = 0; i < (long int)num; i++) {
    for (unsigned int x = 0; x < rows; x++)
      for (unsigned int y = 0; y < cols; y++) {
        float v = (float)in(icin, 1);
        img[ds.transpose ? rows * y + x : cols * x + y] = v;
      }
    images.push_back(img);
  }
  icin.close();

  // ადრე ეს ფაილი შედარებით გზით იხსნებოდა (სურათები კი exe_dir-ით):
  // სხვა სამუშაო დირექტორიიდან გაშვებისას ლეიბლები ჩუმად ნულდებოდა
  icin.open(labPath, std::ios::binary);
  if (!icin.is_open()) {
    std::cout << "cannot open " << labPath << std::endl;
    return;
  }
  unsigned int labCount;
  in(icin, 4), labCount = in(icin, 4);
  if (labCount != num)
    std::cout << "!! " << imgPath.filename() << " has " << num << " images but "
              << labPath.filename() << " has " << labCount << " labels"
              << std::endl;

  labels.reserve(labels.size() + num);
  for (long int i = 0; i < (long int)num; i++)
    labels.push_back(in(icin, 1));
}
