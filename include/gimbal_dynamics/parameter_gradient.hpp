#pragma once

// 对动力学模型参数求灵敏度的仿真：力矩序列是已知输入，56 个模型参数是决策量，
// 用前向灵敏度传播（前向模式）求标量轨迹损失的解析梯度 dL/dp。
//
// 参数维数固定且与步数无关，因此前向传播只需要保存 8×56 的灵敏度矩阵：
//   S_{k+1} = W_k S_k + P_k
//   dL/dp   = Σ_k (∂L/∂X_k)^T S_k
// 其中 W_k 与 P_k 由正向记录的四个阶段 Jacobian 组装（W = ∂y'/∂y，P = ∂y'/∂p），
// 正向轨迹与损失由 gimbal::simulation::simulate 完成，与 gimbal::control_gradient
// 的仿真结果逐位一致。
//
// 参数顺序见 gimbal_dynamics/parameters.hpp：4 个刚体各 10 个（质量、质心 3、
// 惯量对称分量 6），4 个关节安装偏置各 3 个，4 个关节粘性阻尼。

#include <cstddef>

#include <Eigen/Core>

#include "gimbal_dynamics/parameters.hpp"
#include "gimbal_dynamics/simulation.hpp"

namespace gimbal {
namespace parameter_gradient {

using Gradient = Eigen::Matrix<double, parameters::kParameterCount, 1>;

// 调用方分配一次、可跨多次调用复用的缓冲
struct Workspace {
    simulation::Trace trace;  // 前向轨迹记录

    void resize(std::size_t num_steps, int refinement);
    void release();
};

// 前向仿真 + 前向灵敏度传播。返回损失值。
//   gradient     输出，56 维 dL/dp；可为 nullptr（只做前向）
//   final_state  最后一个主步结束后的状态；可为 nullptr
// base 与 torque 的长度均为 num_steps。
double simulateAndSensitivity(const ModelParameters& model, const Terrain& terrain,
                              const simulation::State& initial, const BaseState* base,
                              const Eigen::Vector4d* torque, std::size_t num_steps,
                              const simulation::TrajectorySpec& spec,
                              const simulation::LossSpec& loss, Workspace& workspace,
                              Gradient* gradient, simulation::State* final_state);

}  // namespace parameter_gradient
}  // namespace gimbal
