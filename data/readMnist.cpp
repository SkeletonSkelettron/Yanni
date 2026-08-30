#include "readMnist.h"
#include <filesystem>
#include <fstream>
#include <iostream>
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
    if (fs::exists(dir / d.trainImages) && fs::exists(dir / d.trainLabels) &&
        fs::exists(dir / d.testImages) && fs::exists(dir / d.testLabels))
      found.push_back(d);
  }
  return found;
}

DataSetInfo ChooseDataSet() {
  std::vector<DataSetInfo> sets = AvailableDataSets();

  if (sets.empty()) {
    std::cout << "no dataset found in " << DataDir() << std::endl;
    return DataSetInfo{"", "", "", "", "", false, 0};
  }
  if (sets.size() == 1) {
    std::cout << "dataset: " << sets[0].name << std::endl;
    return sets[0];
  }

  const fs::path dir = DataDir();
  std::cout << "\nseveral datasets found:\n" << std::endl;
  for (size_t i = 0; i < sets.size(); i++)
    printf("  %zu) %-18s %7ld train / %6ld test, %d classes\n", i + 1,
           sets[i].name.c_str(), CountOf(dir / sets[i].trainImages),
           CountOf(dir / sets[i].testImages), sets[i].classCount);

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
  const fs::path imgPath = dir / (train ? ds.trainImages : ds.testImages);
  const fs::path labPath = dir / (train ? ds.trainLabels : ds.testLabels);

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
