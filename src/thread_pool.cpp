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
    std::function<void(int, std::size_t)> dynamic;
    std::function<void(std::size_t)> worker;
    alignas(64) std::atomic<std::size_t> available{0};
    alignas(64) std::atomic<std::size_t> claimed{0};
    alignas(64) std::atomic<std::size_t> next{0};
    std::size_t dynamic_chunk = 1;
    alignas(64) std::atomic<bool> producer_done{false};
    alignas(64) std::atomic<bool> aborted{false};
    std::exception_ptr error;  // 由线程池互斥量保护
    alignas(64) int pending = 0;
};

ThreadPool::ThreadPool(int worker_count) {
    const int count = worker_count < 0 ? 0 : std::min(worker_count, 4096);
    workers_.reserve(static_cast<std::size_t>(count));
    for (int i = 0; i < count; ++i) {
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

void ThreadPool::record_error(Job& job) noexcept {
    try {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!job.error) {
            job.error = std::current_exception();
        }
        job.aborted.store(true, std::memory_order_release);
    } catch (...) {
    }
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
        try {
            if (job->dynamic) {
                for (;;) {
                    if (job->aborted.load(std::memory_order_acquire)) {
                        break;
                    }
                    const std::size_t base =
                        job->next.fetch_add(job->dynamic_chunk, std::memory_order_relaxed);
                    if (base >= job->count) {
                        break;
                    }
                    const std::size_t end = std::min(job->count, base + job->dynamic_chunk);
                    for (std::size_t index = base; index < end; ++index) {
                        if (job->aborted.load(std::memory_order_acquire)) {
                            break;
                        }
                        job->dynamic(worker_index + 1, index);
                    }
                }
            } else if (job->worker) {
                for (;;) {
                    if (job->aborted.load(std::memory_order_acquire)) {
                        break;
                    }
                    std::size_t current = job->claimed.load(std::memory_order_relaxed);
                    if (current >= job->count) {
                        break;
                    }
                    if (job->available.load(std::memory_order_acquire) <= current) {
                        if (job->producer_done.load(std::memory_order_acquire)) {
                            break;
                        }
                        __builtin_ia32_pause();
                        continue;
                    }
                    if (job->claimed.compare_exchange_weak(current, current + 1)) {
                        job->worker(current);
                    }
                }
            }
        } catch (...) {
            record_error(*job);
        }
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (--job->pending == 0) {
                condition_.notify_all();
            }
        }
    }
}

void ThreadPool::parallel_dynamic(
    std::size_t count, const std::function<void(int, std::size_t)>& function,
    std::size_t chunk) {
    if (count == 0) {
        return;
    }
    if (chunk == 0) {
        chunk = 1;
    }
    std::lock_guard<std::mutex> execution_guard(execution_mutex_);
    if (worker_count() == 0) {
        std::atomic<std::size_t> next{0};
        for (;;) {
            const std::size_t base = next.fetch_add(chunk, std::memory_order_relaxed);
            if (base >= count) {
                break;
            }
            const std::size_t end = std::min(count, base + chunk);
            for (std::size_t index = base; index < end; ++index) {
                function(0, index);
            }
        }
        return;
    }
    std::shared_ptr<Job> job(new Job());
    job->count = count;
    job->dynamic = function;
    job->dynamic_chunk = chunk;
    job->pending = participant_count();
    {
        std::lock_guard<std::mutex> lock(mutex_);
        job_ = job;
    }
    condition_.notify_all();
    try {
        for (;;) {
            if (job->aborted.load(std::memory_order_acquire)) {
                break;
            }
            const std::size_t base = job->next.fetch_add(chunk, std::memory_order_relaxed);
            if (base >= count) {
                break;
            }
            const std::size_t end = std::min(count, base + chunk);
            for (std::size_t index = base; index < end; ++index) {
                if (job->aborted.load(std::memory_order_acquire)) {
                    break;
                }
                function(0, index);
            }
        }
    } catch (...) {
        record_error(*job);
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
    std::exception_ptr error;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        error = job->error;
    }
    if (error) {
        std::rethrow_exception(error);
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

ThreadPool::Pipeline::~Pipeline() {
    if (!finished_ && job_ != nullptr) {
        abort();
    }
}

void ThreadPool::Pipeline::publish(std::size_t index) {
    if (!job_ || finished_) {
        throw std::logic_error("流水线已结束，不能再发布任务");
    }
    job_->available.store(index + 1, std::memory_order_release);
}

void ThreadPool::Pipeline::abort() noexcept {
    if (!job_ || finished_) {
        return;
    }
    try {
        job_->aborted.store(true, std::memory_order_release);
        job_->producer_done.store(true, std::memory_order_release);
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
    } catch (...) {
    }
    finished_ = true;
    if (execution_lock_.owns_lock()) {
        execution_lock_.unlock();
    }
}

void ThreadPool::Pipeline::finish() {
    if (finished_) {
        return;
    }
    job_->producer_done.store(true, std::memory_order_release);
    for (;;) {
        if (job_->aborted.load(std::memory_order_acquire)) {
            break;
        }
        std::size_t current = job_->claimed.load(std::memory_order_relaxed);
        if (current >= job_->count) {
            break;
        }
        if (job_->available.load(std::memory_order_acquire) <= current) {
            break;  // producer_done 已置位，不会再有新任务
        }
        if (job_->claimed.compare_exchange_weak(current, current + 1)) {
            try {
                job_->worker(current);
            } catch (...) {
                pool_->record_error(*job_);
                break;
            }
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
    std::exception_ptr error;
    {
        std::lock_guard<std::mutex> lock(pool_->mutex_);
        error = job_->error;
    }
    const std::size_t published = job_->available.load(std::memory_order_acquire);
    const std::size_t count = job_->count;
    finished_ = true;
    if (execution_lock_.owns_lock()) {
        execution_lock_.unlock();
    }
    if (error) {
        std::rethrow_exception(error);
    }
    if (published < count) {
        throw std::runtime_error("流水线任务未全部发布");
    }
}

}  // namespace detail
}  // namespace gimbal
