#pragma once

// 对动力学模型参数求灵敏度的仿真：力矩序列是已知输入，56 个模型参数是决策量，
// 用前向灵敏度传播（前向模式）求标量轨迹损失的解析梯度 dL/dp。
//
// 参数维数固定且与步数无关，因此前向传播只需要保存 8×56 的灵敏度矩阵：
//   S_{k+1} = W_k S_k + P_k
//   dL/dp   = Σ_k (∂L/∂X_k)^T S_k
// 其中 W_k 与 P_k 由四个阶段 Jacobian 组装；正向轨迹与损失由
// gimbal::simulation::simulate 完成，与 gimbal::control_gradient 的仿真结果逐位一致。
//
// 参数顺序见 gimbal_dynamics/parameters.hpp：4 个刚体各 10 个（质量、质心 3、
// 惯量对称分量 6），4 个关节安装偏置各 3 个，4 个关节粘性阻尼。
//
// 单样本调用把前向积分放在调用线程、阶段 Jacobian 交给线程池并行处理（流水线），
// 批量调用把样本动态分配给各参与者线程并复用一个线程内的串行实现。
// 内部有两个线程池：单样本流水线池与批量池，各自把同一池上的调用串行化。
// 因此可以安全地从多个外部线程并发调用，但每次并发调用必须使用独立的 Workspace
// 或 BatchWorkspace；共享同一个工作区并发调用会产生数据竞争。
// 线程数可用环境变量 GIMBAL_PIPELINE_THREADS 与 GIMBAL_BATCH_THREADS 调整
// （数值含调用线程，1 表示只用调用线程即串行执行）。

#include <cstddef>
#include <vector>

#include "gimbal_dynamics/math_types.hpp"
#include "gimbal_dynamics/parameters.hpp"
#include "gimbal_dynamics/simulation.hpp"

namespace gimbal {
namespace parameter_gradient {

using Gradient = math::Vector<Real, parameters::kParameterCount>;

// 调用方分配一次、可跨多次调用复用的缓冲
struct Workspace {
    simulation::Trace trace;  // 前向轨迹记录

    void resize(std::size_t num_steps, int refinement);
    void release();
};

// 批量调用的工作区：每个参与者线程一份单样本工作区
struct BatchWorkspace {
    std::vector<Workspace> workspaces;

    void resize(std::size_t batch);
    void release();
};

// 前向仿真 + 前向灵敏度传播。返回损失值。
//   gradient     输出，56 维 dL/dp；可为 nullptr（只做前向）
//   final_state  最后一个主步结束后的状态；可为 nullptr
// base 与 torque 的长度均为 num_steps。
Real simulateAndSensitivity(const ModelParameters& model, const Terrain& terrain,
                            const simulation::State& initial, const BaseState* base,
                            const math::Vector4d* torque, std::size_t num_steps,
                            const simulation::TrajectorySpec& spec,
                            const simulation::LossSpec& loss, Workspace& workspace,
                            Gradient* gradient, simulation::State* final_state);

// 批量前向仿真 + 前向灵敏度传播，样本之间独立、按参与者线程并行。
// batch=0 时直接返回 0，不校验输入。
//   initial、final_state、loss 长度 batch；
//   base、torque 长度 batch*num_steps，按样本连续排列；
//   gradient 长度 batch，返回批内各样本损失的平均值。
Real simulateAndSensitivityBatch(const ModelParameters& model, const Terrain& terrain,
                                 const simulation::State* initial, const BaseState* base,
                                 const math::Vector4d* torque, std::size_t batch,
                                 std::size_t num_steps, const simulation::TrajectorySpec& spec,
                                 const simulation::LossSpec* loss, BatchWorkspace& workspace,
                                 Gradient* gradient, simulation::State* final_state);

}  // namespace parameter_gradient
}  // namespace gimbal
