#pragma once

// 内部：常驻线程池。
//
// 两种用法：
//   parallel_dynamic：按参与者动态领取任务（批量样本级并行）；
//   start_pipeline：生产者发布任务、工作线程领取处理（单样本前向与阶段 Jacobian 重叠）。
//
// 同一线程池上的任务通过内部互斥量串行执行；任务体抛出的异常会被捕获并保存，
// 在等待全部参与者退出后由调用线程重新抛出。Pipeline 的析构会中止未完成的任务，
// 避免异常路径留下自旋的工作线程。

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <exception>
#include <functional>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

namespace gimbal {
namespace detail {

class ThreadPool {
public:
    struct Job;

    explicit ThreadPool(int worker_count);
    ~ThreadPool();

    ThreadPool(const ThreadPool&) = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;

    int worker_count() const { return static_cast<int>(workers_.size()); }
    int participant_count() const { return worker_count() + 1; }

    // 动态领取一段连续下标；chunk 控制每次领取的个数
    void parallel_dynamic(std::size_t count,
                          const std::function<void(int participant, std::size_t index)>& function,
                          std::size_t chunk = 1);

    class Pipeline {
    public:
        Pipeline() = default;
        Pipeline(Pipeline&&) = default;
        Pipeline& operator=(Pipeline&&) = default;
        Pipeline(const Pipeline&) = delete;
        Pipeline& operator=(const Pipeline&) = delete;
        ~Pipeline();

        // 发布已写入数据的任务下标
        void publish(std::size_t index);
        // 等待全部任务完成；任务体异常时在此重新抛出
        void finish();
        // 中止未完成的任务并等待工作线程退出（析构会自动调用）
        void abort() noexcept;

    private:
        friend class ThreadPool;
        ThreadPool* pool_ = nullptr;
        std::shared_ptr<Job> job_;
        std::unique_lock<std::mutex> execution_lock_;
        bool finished_ = false;
    };

    Pipeline start_pipeline(std::size_t count, const std::function<void(std::size_t)>& worker);

private:
    void worker_loop(int worker_index);
    void record_error(Job& job) noexcept;

    std::vector<std::thread> workers_;
    std::mutex execution_mutex_;  // 同一线程池上的任务互相串行
    std::mutex mutex_;
    std::condition_variable condition_;
    std::shared_ptr<Job> job_;
    bool stopping_ = false;
};

}  // namespace detail
}  // namespace gimbal
