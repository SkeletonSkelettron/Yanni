#ifndef WORKERTHREAD_H
#define WORKERTHREAD_H
#include <functional>
#include <thread>
#include <list>
#include <mutex>
#include <memory>
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
	std::condition_variable itemInQueue;
	std::mutex mutex;
	std::unique_ptr<std::thread> thread;
	std::function<void()> task;
	volatile bool isRunning;
};

#endif // WORKERTHREAD_H