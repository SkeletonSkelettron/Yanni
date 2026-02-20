#ifndef WORKERTHREAD_H
#define WORKERTHREAD_H
#include <functional>
#include <thread>
#include <list>
#include <mutex>
#include <memory>
#include <atomic>
#include <condition_variable>

class WorkerThread
{

public:
	WorkerThread();
	~WorkerThread();
	void doAsync(const std::function<void()>& t);

	void wait();
	void stop();

private:
	void startThread();
private:
	std::condition_variable cv;
	std::mutex mutex;
	std::unique_ptr<std::thread> thread;
	std::function<void()> task;
	std::atomic<bool> isRunning;
	bool hasTask;
};

#endif // WORKERTHREAD_H