#include "workerThread.h"

WorkerThread::WorkerThread() : isRunning(true), hasTask(false)
{
	thread.reset(new std::thread([this]
		{
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
	while (isRunning)
	{
		cv.wait(lock, [this] { return hasTask || !isRunning; });

		if (!isRunning)
			break;

		// Move task locally so doAsync can safely assign next task after we unlock
		std::function<void()> t = std::move(task);
		lock.unlock();

		t();

		lock.lock();
		hasTask = false;
		cv.notify_all();
	}
}

void WorkerThread::doAsync(const std::function<void()>& t)
{
	std::lock_guard<std::mutex> lock(mutex);
	task = t;
	hasTask = true;
	cv.notify_one();
}

void WorkerThread::wait()
{
	std::unique_lock<std::mutex> lock(mutex);
	cv.wait(lock, [this] { return !hasTask; });
}

void WorkerThread::stop()
{
	{
		std::lock_guard<std::mutex> lock(mutex);
		isRunning = false;
		cv.notify_one();
	}
	if (thread && thread->joinable())
		thread->join();
}