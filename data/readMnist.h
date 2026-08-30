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

// ჩვენებითი სახელი ბრძანების ველისთვის: პატარა ასოები, ხარვეზი -> ტირე.
// "EMNIST digits" -> "emnist-digits". ცალკე ველად არ ვინახავთ, რომ ორი
// სახელი ვერასდროს დაშორდეს ერთმანეთს.
std::string DataSetSlug(const std::string &name);

// requested ცარიელი თუ არაა, პრომპტის ნაცვლად მას ეძებს: ან რიგით ნომერს
// (სიაში ჩანს), ან სახელს -- ზუსტად, ან ერთმნიშვნელოვან პრეფიქსად.
// ცარიელზე: ერთი დატასეტი -- ხმაურის გარეშე; რამდენიმე -- ეკითხება;
// არაინტერაქტიულ გაშვებაზე პირველს ირჩევს.
DataSetInfo ChooseDataSet(const std::string &requested = "");

void ReadMNISTMod(std::vector<std::vector<float>> &images,
                  std::vector<int> &labels, bool train,
                  const DataSetInfo &ds);

#endif // READMNIST_H
