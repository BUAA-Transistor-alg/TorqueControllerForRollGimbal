#pragma once

// 定步长前向仿真核心：4 个关节的 8 维状态、RK4 细化积分、标量轨迹损失。
// 公开类型按精度模板化，后端在编译时按精度实例化。

#include <cstddef>
#include <vector>

#include "gimbal_dynamics/dynamics_model.hpp"
#include "gimbal_dynamics/math_types.hpp"

namespace gimbal {
namespace simulation {

// 8 维状态：4 个关节角与 4 个关节角速度
template <typename Real>
struct StateT {
    math::Vector<Real, 4> angle = math::Vector<Real, 4>::zero();
    math::Vector<Real, 4> velocity = math::Vector<Real, 4>::zero();
};

// 标量轨迹损失（时间均值形式）：
//   L = 1/(2K) Σ_k [ Σ_i wa_i (q_i - q*_i)^2 + Σ_i wv_i (v_i - v*_i)^2
//                    + Σ_i wt_i τ_{k,i}^2 ]
template <typename Real>
struct LossSpecT {
    math::Vector<Real, 4> angle_weights = math::Vector<Real, 4>::zero();
    math::Vector<Real, 4> velocity_weights = math::Vector<Real, 4>::zero();
    math::Vector<Real, 4> torque_weights = math::Vector<Real, 4>::zero();
    const math::Vector<Real, 4>* target_angle = nullptr;
    const math::Vector<Real, 4>* target_velocity = nullptr;
};

// 定步长设置：每个主步内做 refinement 次经典 RK4
template <typename Real>
struct TrajectorySpecT {
    Real step = Real(0);
    int refinement = 1;
};

// 前向轨迹记录
template <typename Real>
struct TraceT {
    std::size_t num_steps = 0;
    int refinement = 0;
    std::vector<StateT<Real>> step_end;
    std::vector<StateT<Real>> stage_state;
    std::vector<BaseStateT<Real>> stage_base;

    void resize(std::size_t steps, int refine) {
        num_steps = steps;
        refinement = refine;
        step_end.resize(steps);
        stage_state.resize(steps * static_cast<std::size_t>(refine) * 4);
        stage_base.resize(steps * static_cast<std::size_t>(refine) * 4);
    }

    void release() {
        num_steps = 0;
        refinement = 0;
        step_end.clear();
        stage_state.clear();
        stage_base.clear();
    }
};

// 后端内部按构建精度使用的别名
using State = StateT<Real>;
using LossSpec = LossSpecT<Real>;
using TrajectorySpec = TrajectorySpecT<Real>;
using Trace = TraceT<Real>;

// 配置校验
void validate(const TrajectorySpec& spec, std::size_t num_steps);

// 单步状态导数 y' = f(y, u)
math::Vector<Real, 8> state_derivative(const ModelParameters& model, const Terrain& terrain,
                                       const BaseState& base, const State& state,
                                       const math::Vector<Real, 4>& torque);

Real step_loss(const LossSpec& loss, std::size_t num_steps, std::size_t step, const State& state,
               const math::Vector<Real, 4>& torque);

math::Vector<Real, 8> step_loss_gradient(const LossSpec& loss, std::size_t num_steps,
                                         std::size_t step, const State& state);

Real simulate(const ModelParameters& model, const Terrain& terrain, const State& initial,
              const BaseState* base, const math::Vector<Real, 4>* torque, std::size_t num_steps,
              const TrajectorySpec& spec, const LossSpec& loss, Trace& trace);

}  // namespace simulation
}  // namespace gimbal
