#include "thread_pool.hpp"

#include <algorithm>
#include <cstdlib>
#include <stdexcept>
#include <thread>

namespace gimbal {
namespace detail {

// 高频读写的原子量各自独占缓存行，避免生产者与工作线程的伪共享
struct ThreadPool::Job {
    std::size_t count = 0;
    std::function<void(int, std::size_t, std::size_t)> parallel;
    std::function<void(int, std::size_t)> dynamic;
    std::function<void(std::size_t)> worker;
    alignas(64) std::atomic<std::size_t> available{0};
    alignas(64) std::atomic<std::size_t> claimed{0};
    alignas(64) std::atomic<std::size_t> next{0};
    alignas(64) std::atomic<bool> producer_done{false};
    alignas(64) int pending = 0;
};

namespace {

int hardware_threads() {
    const unsigned int hardware = std::thread::hardware_concurrency();
    return hardware > 1 ? static_cast<int>(hardware) - 1 : 0;
}

int environment_threads(const char* name) {
    const char* environment = std::getenv(name);
    if (environment == nullptr) {
        return 0;
    }
    const int requested = std::atoi(environment);
    if (requested < 1) {
        return 0;
    }
    return requested - 1;  // 返回工作线程数：总线程数含调用线程
}

int pipeline_worker_count() {
    const int configured = environment_threads("GIMBAL_PIPELINE_THREADS");
    if (configured > 0) {
        return configured;
    }
    const int available = hardware_threads();
    const int limit = 7;
    return available < limit ? available : limit;
}

int batch_worker_count() {
    const int configured = environment_threads("GIMBAL_BATCH_THREADS");
    if (configured > 0) {
        return configured;
    }
    return hardware_threads();
}

}  // namespace

ThreadPool::ThreadPool(int worker_count) {
    workers_.reserve(static_cast<std::size_t>(worker_count < 0 ? 0 : worker_count));
    for (int i = 0; i < worker_count; ++i) {
        workers_.emplace_back([this, i] { worker_loop(i); });
    }
}

ThreadPool::~ThreadPool() {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        stopping_ = true;
    }
    condition_.notify_all();
    for (std::thread& worker : workers_) {
        if (worker.joinable()) {
            worker.join();
        }
    }
}

ThreadPool& ThreadPool::pool() {
    static ThreadPool instance(pipeline_worker_count());
    return instance;
}

ThreadPool& ThreadPool::batch_pool() {
    static ThreadPool instance(batch_worker_count());
    return instance;
}

void ThreadPool::worker_loop(int worker_index) {
    std::shared_ptr<Job> last;
    for (;;) {
        std::shared_ptr<Job> job;
        {
            std::unique_lock<std::mutex> lock(mutex_);
            condition_.wait(lock, [&] {
                return stopping_ || (job_ != nullptr && job_ != last);
            });
            if (stopping_) {
                return;
            }
            job = job_;
            last = job;
        }
        if (job->dynamic) {
            for (;;) {
                const std::size_t index = job->next.fetch_add(1, std::memory_order_relaxed);
                if (index >= job->count) {
                    break;
                }
                job->dynamic(worker_index + 1, index);
            }
        } else if (job->worker) {
            for (;;) {
                std::size_t current = job->claimed.load(std::memory_order_relaxed);
                if (current >= job->count) {
                    break;
                }
                if (job->available.load(std::memory_order_acquire) <= current) {
                    if (job->producer_done.load(std::memory_order_acquire)) {
                        break;
                    }
#if defined(__x86_64__) || defined(__i386__)
                    __builtin_ia32_pause();
#else
                    std::this_thread::yield();
#endif
                    continue;
                }
                if (job->claimed.compare_exchange_weak(current, current + 1)) {
                    job->worker(current);
                }
            }
        } else {
            const std::size_t participants = static_cast<std::size_t>(participant_count());
            const std::size_t chunk = (job->count + participants - 1) / participants;
            const std::size_t chunk_count = (job->count + chunk - 1) / chunk;
            for (std::size_t c = static_cast<std::size_t>(worker_index) + 1; c < chunk_count;
                 c += participants) {
                const std::size_t begin = c * chunk;
                const std::size_t end = std::min(job->count, begin + chunk);
                job->parallel(static_cast<int>(worker_index) + 1, begin, end);
            }
        }
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (--job->pending == 0) {
                condition_.notify_all();
            }
        }
    }
}

void ThreadPool::parallel_for(
    std::size_t count, const std::function<void(int, std::size_t, std::size_t)>& function) {
    if (count == 0) {
        return;
    }
    std::lock_guard<std::mutex> execution_guard(execution_mutex_);
    const std::size_t participants = static_cast<std::size_t>(participant_count());
    const std::size_t chunk = (count + participants - 1) / participants;
    const std::size_t chunk_count = (count + chunk - 1) / chunk;
    if (worker_count() == 0) {
        for (std::size_t c = 0; c < chunk_count; ++c) {
            const std::size_t begin = c * chunk;
            const std::size_t end = std::min(count, begin + chunk);
            function(0, begin, end);
        }
        return;
    }
    std::shared_ptr<Job> job(new Job());
    job->count = count;
    job->parallel = function;
    job->pending = participant_count();
    {
        std::lock_guard<std::mutex> lock(mutex_);
        job_ = job;
    }
    condition_.notify_all();
    for (std::size_t c = 0; c < chunk_count; c += participants) {
        const std::size_t begin = c * chunk;
        const std::size_t end = std::min(count, begin + chunk);
        function(0, begin, end);
    }
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (--job->pending == 0) {
            condition_.notify_all();
        }
    }
    {
        std::unique_lock<std::mutex> lock(mutex_);
        condition_.wait(lock, [&] { return job->pending == 0; });
    }
}

void ThreadPool::parallel_dynamic(
    std::size_t count, const std::function<void(int, std::size_t)>& function) {
    if (count == 0) {
        return;
    }
    std::lock_guard<std::mutex> execution_guard(execution_mutex_);
    if (worker_count() == 0) {
        for (std::size_t index = 0; index < count; ++index) {
            function(0, index);
        }
        return;
    }
    std::shared_ptr<Job> job(new Job());
    job->count = count;
    job->dynamic = function;
    job->pending = participant_count();
    {
        std::lock_guard<std::mutex> lock(mutex_);
        job_ = job;
    }
    condition_.notify_all();
    for (;;) {
        const std::size_t index = job->next.fetch_add(1, std::memory_order_relaxed);
        if (index >= count) {
            break;
        }
        function(0, index);
    }
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (--job->pending == 0) {
            condition_.notify_all();
        }
    }
    {
        std::unique_lock<std::mutex> lock(mutex_);
        condition_.wait(lock, [&] { return job->pending == 0; });
    }
}

ThreadPool::Pipeline ThreadPool::start_pipeline(
    std::size_t count, const std::function<void(std::size_t)>& worker) {
    // 同一线程池上的流水线与其他任务串行执行，避免 job_ 被并发覆盖
    std::unique_lock<std::mutex> execution_lock(execution_mutex_);
    std::shared_ptr<Job> job(new Job());
    job->count = count;
    job->worker = worker;
    job->pending = participant_count();
    {
        std::lock_guard<std::mutex> lock(mutex_);
        job_ = job;
    }
    condition_.notify_all();
    Pipeline pipeline;
    pipeline.pool_ = this;
    pipeline.job_ = job;
    pipeline.execution_lock_ = std::move(execution_lock);
    return pipeline;
}

void ThreadPool::Pipeline::publish(std::size_t index) {
    job_->available.store(index + 1, std::memory_order_release);
}

void ThreadPool::Pipeline::finish() {
    job_->producer_done.store(true, std::memory_order_release);
    // 生产者结束后调用线程继续领取剩余任务，避免只等不干活
    for (;;) {
        std::size_t current = job_->claimed.load(std::memory_order_relaxed);
        if (current >= job_->count) {
            break;
        }
        if (job_->available.load(std::memory_order_acquire) <= current) {
            // producer_done 已置位，不会再发布新任务，直接结束
            break;
        }
        if (job_->claimed.compare_exchange_weak(current, current + 1)) {
            job_->worker(current);
        }
    }
    {
        std::lock_guard<std::mutex> lock(pool_->mutex_);
        if (--job_->pending == 0) {
            pool_->condition_.notify_all();
        }
    }
    {
        std::unique_lock<std::mutex> lock(pool_->mutex_);
        pool_->condition_.wait(lock, [&] { return job_->pending == 0; });
    }
    const std::size_t published = job_->available.load(std::memory_order_acquire);
    const std::size_t count = job_->count;
    finished_ = true;
    if (execution_lock_.owns_lock()) {
        execution_lock_.unlock();
    }
    if (published < count) {
        throw std::runtime_error("流水线任务未全部发布");
    }
}

}  // namespace detail
}  // namespace gimbal
