#include "workerThread.h"
#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#define CPU_RELAX() _mm_pause()
#else
#define CPU_RELAX() ((void)0)
#endif

// ამდენი პაუზის შემდეგ ლოდინი აღარ არის "მოკლე" და ბირთვს ვუთმობთ,
// რომ უქმი ნაკადი პროცესორს არ წვავდეს (მაგ. ეპოქებს შორის).
static const int SPIN_LIMIT = 2000;

void WorkerThread::idle(int &spins) {
  if (++spins < SPIN_LIMIT)
    CPU_RELAX();
  else
    std::this_thread::yield();
}

WorkerThread::WorkerThread() : isRunning(true), hasTask(false) {
  thread.reset(new std::thread([this] { this->startThread(); }));
}

WorkerThread::~WorkerThread() { stop(); }

void WorkerThread::startThread() {
  while (true) {
    int spins = 0;
    while (!hasTask.load(std::memory_order_acquire)) {
      if (!isRunning.load(std::memory_order_relaxed))
        return;
      idle(spins);
    }

    task();

    // release: ამ მომენტამდე ჩაწერილი ყველა შედეგი wait()-ისთვის ხილვადია
    hasTask.store(false, std::memory_order_release);
  }
}

void WorkerThread::doAsync(const std::function<void()> &t) {
  // უსაფრთხოა უსინქრონოდ: გამომძახებელი აქ მხოლოდ წინა wait()-ის შემდეგ
  // ხვდება, ანუ მუშა ნაკადი ამ დროს task-ს არ კითხულობს
  task = t;
  hasTask.store(true, std::memory_order_release);
}

void WorkerThread::wait() {
  int spins = 0;
  while (hasTask.load(std::memory_order_acquire))
    idle(spins);
}

void WorkerThread::stop() {
  isRunning.store(false, std::memory_order_relaxed);
  if (thread && thread->joinable())
    thread->join();
}
