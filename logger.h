#ifndef LOGGER_H
#define LOGGER_H

#include <string>

// yanni.log -- ტრენინგის მიმდინარეობა მანქანურად წასაკითხ ფორმატში.
//
// ყოველი ხაზი ჩაწერისთანავე იძირება დისკზე (flush), ე.ი. ფაილი
// პარალელურად იკითხება მაშინაც, როცა ტრენინგი ჯერ მიმდინარეობს:
//   tail -f yanni.log
//   pandas.read_csv("yanni.log", comment="#")
//
// არსებული ფაილი უკითხავად გადაიწერება.
void LogOpen(const std::string &path);
bool LogIsOpen();

// '# ' პრეფიქსით -- CSV-ის მკითხველები ამას გამოტოვებენ
void LogComment(const char *fmt, ...);
// ნედლი ხაზი: სვეტების სათაური და მონაცემები
void LogLine(const char *fmt, ...);
void LogClose();

#endif // LOGGER_H
