#pragma once

// 内部：固定大小线程池。
//
// 两种用法：
//   parallel_for：按参与者编号静态分块执行（主线程是 0 号参与者）；
//   start_pipeline：主线程发布任务、工作线程领取处理，用于前向仿真与
//   阶段 Jacobian 的流水线重叠。

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <functional>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

namespace gimbal {
namespace detail {

class ThreadPool {
public:
    // 任务载体，由线程池内部创建与回收
    struct Job;

    explicit ThreadPool(int worker_count);
    ~ThreadPool();

    ThreadPool(const ThreadPool&) = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;

    int worker_count() const { return static_cast<int>(workers_.size()); }
    int participant_count() const { return worker_count() + 1; }

    // 单样本流水线用的线程池（少量工作线程，避免生产者与工作线程争抢缓存行）
    static ThreadPool& pool();
    // 批量调用用的线程池（全部工作线程，样本级并行）
    static ThreadPool& batch_pool();

    void parallel_for(std::size_t count,
                      const std::function<void(int participant, std::size_t begin,
                                               std::size_t end)>& function);

    // 动态领取：每个参与者按需领取一段下标，适合任务代价相近但核心异构的场景
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
        ~Pipeline() = default;

        // 发布已写入数据的任务下标；finish 等待全部任务完成并释放执行权
        void publish(std::size_t index);
        void finish();

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

    std::vector<std::thread> workers_;
    std::mutex execution_mutex_;  // 同一线程池上的任务互相串行
    std::mutex mutex_;
    std::condition_variable condition_;
    std::shared_ptr<Job> job_;
    bool stopping_ = false;
};

}  // namespace detail
}  // namespace gimbal
