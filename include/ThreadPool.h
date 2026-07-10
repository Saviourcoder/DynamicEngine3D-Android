#pragma once
#include <thread>
#include <vector>
#include <atomic>
#include <functional>
#include <condition_variable>
#include <mutex>
#include <queue>

class ThreadPool
{
public:
    explicit ThreadPool(int numThreads = -1)
    {
        if (numThreads <= 0)
            numThreads = (int)std::thread::hardware_concurrency() - 1;
        if (numThreads < 1) numThreads = 1;

        running = true;
        for (int i = 0; i < numThreads; ++i)
            workers.emplace_back([this] { WorkerLoop(); });
    }

    ~ThreadPool()
    {
        {
            std::lock_guard<std::mutex> lock(queueMutex);
            running = false;
        }
        cv.notify_all();
        for (auto& t : workers) t.join();
    }

    // Splits [0, count) into chunks and runs fn(start, end) across the pool.
    // Blocks until all chunks are done. Call this instead of #pragma omp parallel for.
    void ParallelFor(int count, const std::function<void(int, int)>& fn, int minChunk = 64)
    {
        if (count <= 0) return;

        int numWorkers = (int)workers.size();
        int chunkCount = std::min(numWorkers, std::max(1, count / minChunk));
        if (chunkCount <= 1)
        {
            fn(0, count);
            return;
        }

        int chunkSize = (count + chunkCount - 1) / chunkCount;
        std::atomic<int> remaining{ chunkCount };
        std::condition_variable doneCv;
        std::mutex doneMutex;

        for (int c = 0; c < chunkCount; ++c)
        {
            int start = c * chunkSize;
            int end = std::min(count, start + chunkSize);
            if (start >= end) { remaining--; continue; }

            {
                std::lock_guard<std::mutex> lock(queueMutex);
                jobs.push([&fn, start, end, &remaining, &doneCv, &doneMutex]() {
                    fn(start, end);
                    if (--remaining == 0)
                    {
                        std::lock_guard<std::mutex> l(doneMutex);
                        doneCv.notify_one();
                    }
                    });
            }
            cv.notify_one();
        }

        std::unique_lock<std::mutex> lock(doneMutex);
        doneCv.wait(lock, [&remaining] { return remaining.load() == 0; });
    }

    void Reconfigure(int numThreads)
    {
        if (numThreads <= 0) return;

        // Shut down current workers cleanly first
        {
            std::lock_guard<std::mutex> lock(queueMutex);
            running = false;
        }
        cv.notify_all();
        for (auto& t : workers) t.join();
        workers.clear();

        // Clear any leftover queued jobs from the old config
        {
            std::lock_guard<std::mutex> lock(queueMutex);
            std::queue<std::function<void()>> empty;
            std::swap(jobs, empty);
        }

        // Spin up fresh workers with the new count
        running = true;
        for (int i = 0; i < numThreads; ++i)
            workers.emplace_back([this] { WorkerLoop(); });
    }

private:
    void WorkerLoop()
    {
        while (true)
        {
            std::function<void()> job;
            {
                std::unique_lock<std::mutex> lock(queueMutex);
                cv.wait(lock, [this] { return !running || !jobs.empty(); });
                if (!running && jobs.empty()) return;
                job = std::move(jobs.front());
                jobs.pop();
            }
            job();
        }
    }

    std::vector<std::thread> workers;
    std::queue<std::function<void()>> jobs;
    std::mutex queueMutex;
    std::condition_variable cv;
    bool running = false;
};