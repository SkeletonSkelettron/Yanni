#include "logger.h"
#include <cstdarg>
#include <cstdio>
#include <fstream>
#include <iostream>

static std::ofstream gLog;

void LogOpen(const std::string &path) {
  // std::ios::trunc -- არსებულს კითხვის გარეშე ვაცარიელებთ
  gLog.open(path, std::ios::out | std::ios::trunc);
  if (!gLog.is_open()) {
    std::cout << "cannot write " << path << ", continuing without a log"
              << std::endl;
    return;
  }
  std::cout << "logging to " << path << std::endl;
}

bool LogIsOpen() { return gLog.is_open(); }

static void Write(const char *prefix, const char *fmt, va_list ap) {
  if (!gLog.is_open())
    return;
  char buf[1024];
  vsnprintf(buf, sizeof buf, fmt, ap);
  gLog << prefix << buf << '\n';
  // ყოველ ხაზზე: სხვა პროცესი მხოლოდ ჩაწერილს ხედავს, ბუფერს არა
  gLog.flush();
}

void LogComment(const char *fmt, ...) {
  va_list ap;
  va_start(ap, fmt);
  Write("# ", fmt, ap);
  va_end(ap);
}

void LogLine(const char *fmt, ...) {
  va_list ap;
  va_start(ap, fmt);
  Write("", fmt, ap);
  va_end(ap);
}

void LogClose() {
  if (gLog.is_open())
    gLog.close();
}
