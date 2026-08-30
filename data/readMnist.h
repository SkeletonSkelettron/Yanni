#ifndef READMNIST_H
#define READMNIST_H

#include <cstring>
#include <fstream>
#include <string>
#include <vector>

// ერთი დატასეტის აღწერა. ყველა IDX ფორმატშია, ე.ი. მხოლოდ ფაილების
// სახელები და ორიენტაცია განსხვავდება.
struct DataSetInfo {
  std::string name;
  std::string trainImages, trainLabels;
  std::string testImages, testLabels;
  // EMNIST სურათებს ტრანსპონირებულად ინახავს: ფაილში row-major ჩანაწერი
  // 90 გრადუსით შემობრუნებულ ციფრს იძლევა. MNIST-ს ეს არ სჭირდება.
  bool transpose;
  int classCount;
};

unsigned int in(std::ifstream &icin, unsigned int size);

// mnist/ დირექტორიაში ნაპოვნი დატასეტები, რომელთაც ოთხივე ფაილი აქვთ
std::vector<DataSetInfo> AvailableDataSets();

// ერთი თუ იპოვა -- მას აბრუნებს ხმაურის გარეშე; რამდენიმე -- ეკითხება.
// არაინტერაქტიულ გაშვებაზე (მილი, სკრიპტი) პირველს ირჩევს.
DataSetInfo ChooseDataSet();

void ReadMNISTMod(std::vector<std::vector<float>> &images,
                  std::vector<int> &labels, bool train,
                  const DataSetInfo &ds);

#endif // READMNIST_H
