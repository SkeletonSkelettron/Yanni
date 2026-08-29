#ifndef WORKERTHREAD_H
#define WORKERTHREAD_H
#include <atomic>
#include <functional>
#include <memory>
#include <thread>

// სპინ-ბარიერი მუტექსისა და condition_variable-ის ნაცვლად.
//
// ქსელი ერთ ნიმუშზე ~15-ჯერ ასინქრონებს ნაკადებს, ანუ ეპოქაში ~900 000-ჯერ.
// futex-ის სისტემურ გამოძახებას ამ სიხშირეზე 5-13 მიკროწამი უჯდება,
// მაშინ როცა თვით სამუშაო ხშირად მიკროწამზე ნაკლებია. სპინი ბირთვში
// საერთოდ არ შედის და იმავე ბარიერს 0.3-0.9 მიკროწამში ატარებს.
//
// ინტერფეისი (doAsync / wait / stop) უცვლელია.
class WorkerThread {

public:
  WorkerThread();
  ~WorkerThread();
  void doAsync(const std::function<void()> &t);

  void wait();
  void stop();

private:
  void startThread();
  // მოკლე უქმი ნაბიჯი: ჯერ პაუზა, დიდხანს ლოდინისას ნაკადის დათმობა
  static void idle(int &spins);

private:
  std::unique_ptr<std::thread> thread;
  std::function<void()> task;
  std::atomic<bool> isRunning;
  // true: დავალება მზადაა და ჯერ არ დასრულებულა.
  // release/acquire წყვილი ერთდროულად ორ რამეს იცავს: მთავარი ნაკადის
  // მიერ ჩაწერილ task-ს (doAsync -> startThread) და მუშა ნაკადის
  // შედეგებს (startThread -> wait).
  std::atomic<bool> hasTask;
};

#endif // WORKERTHREAD_H
