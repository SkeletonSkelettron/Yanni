#include "workerThread.h"

WorkerThread::WorkerThread() :isRunning(false)
{
	thread.reset(new std::thread([this]
		{
			isRunning = true;
			this->startThread();
		}));
}

WorkerThread::~WorkerThread()
{
	stop();
}

void WorkerThread::startThread()
{
	std::unique_lock<std::mutex> lock(mutex);
	do
	{
		while (isRunning && task == NULL)
			itemInQueue.wait(lock);

		lock.unlock();
		const std::function<void()> t = task;
		t();
		task = NULL;
		lock.lock();
		itemInQueue.notify_all();

	} while (isRunning);
	itemInQueue.notify_all();
}

void WorkerThread::doAsync(const std::function<void()>& t)
{
	std::lock_guard<std::mutex> _(mutex);
	task = t;
	itemInQueue.notify_one();

}

void WorkerThread::wait()
{
	std::unique_lock<std::mutex> lock(mutex);
	while (task != NULL)
		itemInQueue.wait(lock);
}

void WorkerThread::stop()
{
	{
		std::lock_guard<std::mutex> lock(mutex);
		isRunning = false;
		itemInQueue.notify_one();
	}
	thread->join();
}