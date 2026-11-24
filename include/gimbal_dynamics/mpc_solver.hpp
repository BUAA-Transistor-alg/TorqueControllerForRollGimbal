#pragma once

// C++ 端的滚动时域 MPC 求解器。
//
// 求解器内部维护热启动解与上一拍实际力矩，每次 solve 输入当前状态、地形、底盘状态与
// 未来 horizon 步目标，返回本拍应施加的力矩。决策变量为 horizon×4 个关节力矩，箱约束为
// 各关节 ±torque_limit_i；代价与梯度由 ControlGradient 处理器的反向伴随法给出，优化器为带
// 箱约束的有限内存拟牛顿法（L-BFGS 方向 + 投影 Armijo 线搜索）。
//
// 代价（K 为预测步数，损失按 1/K 取时间平均）：
//   1/K Σ_k Σ_i [ wa_i (q_ki − q*_ki)^2 + wv_i (v_ki − v*_ki)^2
//                 + wt_i τ_ki^2 + wr_i (τ_ki − τ_{k−1,i})^2
//                 + wl_i hinge_i(q_ki)^2 ]
// 其中 k = 0 的变化率项参照上一次采用的第一步力矩（求解器内部维护），
// 因此 τ_0 与上一拍实际施加力矩的差距也计入变化率代价。
// 目标与限位由调用方通过 ConfigT 与 solve 参数给出；目标指针为空表示目标取零。

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

#include "gimbal_dynamics/dynamics_model.hpp"
#include "gimbal_dynamics/handlers.hpp"
#include "gimbal_dynamics/math_types.hpp"
#include "gimbal_dynamics/simulation.hpp"

namespace tcrg::gimbal_dynamics {
namespace mpc {

// 求解器配置：时域、步长、约束、迭代上限、收敛阈值与完整代价权重
template <typename Real>
struct ConfigT {
    std::size_t horizon = 20;
    Real step = Real(0.01);
    int refinement = 2;
    // 各关节力矩上限（N·m），取实车通信子模组 McuDataPreprocessor 中四个力矩最大值的绝对值；
    // 关节顺序：大 yaw、roll、小 yaw、pitch。
    math::Vector<Real, 4> torque_limit =
        math::Vector<Real, 4>::from(Real(6.561), Real(6.640), Real(2.000), Real(3.992));
    int max_iter = 40;
    Real tolerance = Real(1e-4);
    simulation::LossWeightsT<Real> loss;
};

// 单次求解信息
template <typename Real>
struct InfoT {
    Real loss = Real(0);
    int iterations = 0;
    int evaluations = 0;
    bool converged = false;
    Real projected_gradient = Real(0);
};

template <typename Real>
class SolverT {
public:
    // thread_count 为参与阶段 Jacobian 计算的线程数，1 表示串行
    SolverT(int thread_count, const ModelParametersT<Real>& model, const ConfigT<Real>& config)
        : gradient_(thread_count < 1 ? 1 : thread_count), model_(model), config_(config) {
        validate_config();
        solution_.assign(config_.horizon, math::Vector<Real, 4>::zero());
        previous_torque_ = math::Vector<Real, 4>::zero();
        have_solution_ = false;
    }

    // 清空热启动解与上一拍力矩，供回合切换使用
    void reset() {
        solution_.assign(config_.horizon, math::Vector<Real, 4>::zero());
        previous_torque_ = math::Vector<Real, 4>::zero();
        have_solution_ = false;
    }

    // 最近一次求解得到的完整力矩序列
    const std::vector<math::Vector<Real, 4>>& solution() const { return solution_; }

    const ConfigT<Real>& config() const { return config_; }

    // 求解本拍；target_angle 与 target_velocity 长度为 horizon，velocity 为空表示目标为零。
    // info 可为空。返回本拍应施加的力矩。
    math::Vector<Real, 4> solve(const simulation::StateT<Real>& initial,
                                const TerrainT<Real>& terrain, const BaseStateT<Real>& base,
                                const math::Vector<Real, 4>* target_angle,
                                const math::Vector<Real, 4>* target_velocity,
                                InfoT<Real>* info);

private:
    void validate_config() const;

    ControlGradient<Real> gradient_;
    ModelParametersT<Real> model_;
    ConfigT<Real> config_;
    std::vector<math::Vector<Real, 4>> solution_;
    math::Vector<Real, 4> previous_torque_;
    bool have_solution_ = false;
};

namespace detail {

template <typename Real>
Real vector_dot(const std::vector<Real>& lhs, const std::vector<Real>& rhs) {
    Real total = Real(0);
    for (std::size_t index = 0; index < lhs.size(); ++index) {
        total += lhs[index] * rhs[index];
    }
    return total;
}

}  // namespace detail

template <typename Real>
void SolverT<Real>::validate_config() const {
    if (config_.horizon < 1) {
        throw std::invalid_argument("MPC 预测步数必须大于 0");
    }
    if (!(config_.step > Real(0)) || !std::isfinite(config_.step)) {
        throw std::invalid_argument("MPC 步长必须是大于 0 的有限值");
    }
    if (config_.refinement < 1) {
        throw std::invalid_argument("MPC 细化倍数必须大于等于 1");
    }
    for (int joint = 0; joint < 4; ++joint) {
        if (!(config_.torque_limit[joint] > Real(0)) ||
            !std::isfinite(config_.torque_limit[joint])) {
            throw std::invalid_argument("MPC 各轴力矩上限必须是大于 0 的有限值");
        }
    }
    if (config_.max_iter < 1) {
        throw std::invalid_argument("MPC 迭代上限必须大于等于 1");
    }
    if (!(config_.tolerance > Real(0))) {
        throw std::invalid_argument("MPC 收敛阈值必须大于 0");
    }
    for (int joint = 0; joint < 4; ++joint) {
        if (config_.loss.limit_weights[joint] != Real(0)) {
            const Real lower = config_.loss.limit_lower[joint];
            const Real upper = config_.loss.limit_upper[joint];
            const Real margin = config_.loss.limit_margin[joint];
            if (!std::isfinite(lower) || !std::isfinite(upper) ||
                !(lower + margin < upper - margin)) {
                throw std::invalid_argument("MPC 限位权重非零时软限位区间必须非空");
            }
        }
    }
}

template <typename Real>
math::Vector<Real, 4> SolverT<Real>::solve(const simulation::StateT<Real>& initial,
                                           const TerrainT<Real>& terrain,
                                           const BaseStateT<Real>& base,
                                           const math::Vector<Real, 4>* target_angle,
                                           const math::Vector<Real, 4>* target_velocity,
                                           InfoT<Real>* info) {
    if (target_angle == nullptr) {
        throw std::invalid_argument("MPC 目标角度不能为空");
    }
    const std::size_t count = config_.horizon;
    const std::size_t dimensions = count * 4;
    const math::Vector<Real, 4>& limit = config_.torque_limit;
    std::vector<BaseStateT<Real>> bases(count, base);
    std::vector<math::Vector<Real, 4>> torques(count, math::Vector<Real, 4>::zero());
    std::vector<Real> x(dimensions, Real(0));
    std::vector<Real> gradient(dimensions, Real(0));
    std::vector<Real> trial(dimensions, Real(0));
    std::vector<Real> trial_gradient(dimensions, Real(0));
    std::vector<Real> direction(dimensions, Real(0));
    std::vector<std::vector<Real>> s_history;
    std::vector<std::vector<Real>> y_history;
    std::vector<Real> rho_history;
    const std::size_t memory = 10;
    InfoT<Real> result;
    int evaluations = 0;

    // 箱约束按关节分别施加，索引除以 4 的余数即该分量的关节号
    const auto clamp_torque = [&limit](std::size_t index, Real value) {
        const Real axis_limit = limit[static_cast<int>(index % 4)];
        return std::min(std::max(value, -axis_limit), axis_limit);
    };
    const auto evaluate = [&](const std::vector<Real>& values,
                              std::vector<Real>& output_gradient) {
        for (std::size_t step = 0; step < count; ++step) {
            torques[step] = math::Vector<Real, 4>::from(
                values[step * 4], values[step * 4 + 1], values[step * 4 + 2],
                values[step * 4 + 3]);
        }
        simulation::LossSpecT<Real> loss;
        loss.angle_weights = config_.loss.angle_weights;
        loss.velocity_weights = config_.loss.velocity_weights;
        loss.torque_weights = config_.loss.torque_weights;
        loss.torque_rate_weights = config_.loss.torque_rate_weights;
        loss.limit_weights = config_.loss.limit_weights;
        loss.limit_lower = config_.loss.limit_lower;
        loss.limit_upper = config_.loss.limit_upper;
        loss.limit_margin = config_.loss.limit_margin;
        loss.target_angle = target_angle;
        loss.target_velocity = target_velocity;
        loss.previous_torque = &previous_torque_;
        ControlGradientInputT<Real> input;
        input.model = model_;
        input.terrain = terrain;
        input.initial = initial;
        input.base = bases.data();
        input.torque = torques.data();
        input.num_steps = count;
        input.spec.step = config_.step;
        input.spec.refinement = config_.refinement;
        input.loss = loss;
        ControlGradientOutputT<Real> output;
        gradient_.run(input, output);
        output_gradient.resize(dimensions);
        for (std::size_t step = 0; step < count; ++step) {
            for (int joint = 0; joint < 4; ++joint) {
                output_gradient[step * 4 + static_cast<std::size_t>(joint)] =
                    output.gradient[step][joint];
            }
        }
        ++evaluations;
        return output.loss;
    };

    // 初始解：上一拍解左移一位、末位保持；没有历史时用上一拍力矩铺满预测区间
    if (have_solution_) {
        for (std::size_t step = 0; step + 1 < count; ++step) {
            for (int joint = 0; joint < 4; ++joint) {
                x[step * 4 + static_cast<std::size_t>(joint)] =
                    solution_[step + 1][joint];
            }
        }
        for (int joint = 0; joint < 4; ++joint) {
            x[(count - 1) * 4 + static_cast<std::size_t>(joint)] =
                solution_[count - 1][joint];
        }
    } else {
        for (std::size_t step = 0; step < count; ++step) {
            for (int joint = 0; joint < 4; ++joint) {
                x[step * 4 + static_cast<std::size_t>(joint)] = previous_torque_[joint];
            }
        }
    }
    for (std::size_t index = 0; index < dimensions; ++index) {
        x[index] = clamp_torque(index, x[index]);
    }

    Real loss_value = evaluate(x, gradient);
    Real projected_gradient = Real(0);
    int iteration = 0;
    bool converged = false;
    while (iteration < config_.max_iter) {
        // 投影梯度范数，用作收敛判定
        projected_gradient = Real(0);
        for (std::size_t index = 0; index < dimensions; ++index) {
            const Real projected = clamp_torque(index, x[index] - gradient[index]);
            projected_gradient = std::max(projected_gradient, std::fabs(x[index] - projected));
        }
        if (projected_gradient <= config_.tolerance) {
            converged = true;
            break;
        }

        // 两循环递归得到 L-BFGS 方向
        direction = gradient;
        std::vector<Real> alpha(s_history.size(), Real(0));
        for (std::size_t slot = s_history.size(); slot > 0; --slot) {
            const std::size_t index = slot - 1;
            alpha[index] = rho_history[index] * detail::vector_dot(s_history[index], direction);
            for (std::size_t j = 0; j < dimensions; ++j) {
                direction[j] -= alpha[index] * y_history[index][j];
            }
        }
        Real scale = Real(1);
        if (!s_history.empty()) {
            const Real sy = detail::vector_dot(s_history.back(), y_history.back());
            const Real yy = detail::vector_dot(y_history.back(), y_history.back());
            if (yy > Real(0)) {
                scale = std::min(std::max(sy / yy, Real(1e-10)), Real(1e10));
            }
        }
        for (std::size_t j = 0; j < dimensions; ++j) {
            direction[j] *= scale;
        }
        for (std::size_t index = 0; index < s_history.size(); ++index) {
            const Real beta = rho_history[index] * detail::vector_dot(y_history[index], direction);
            for (std::size_t j = 0; j < dimensions; ++j) {
                direction[j] += s_history[index][j] * (alpha[index] - beta);
            }
        }
        for (std::size_t j = 0; j < dimensions; ++j) {
            direction[j] = -direction[j];
        }
        Real slope = detail::vector_dot(gradient, direction);
        if (!(slope < Real(0))) {
            for (std::size_t j = 0; j < dimensions; ++j) {
                direction[j] = -gradient[j];
            }
            slope = -detail::vector_dot(gradient, gradient);
        }

        // 投影 Armijo 线搜索；步长按方向与梯度尺度初始化
        Real step = Real(1);
        if (s_history.empty()) {
            Real largest = Real(0);
            for (std::size_t j = 0; j < dimensions; ++j) {
                largest = std::max(largest, std::fabs(gradient[j]));
            }
            step = Real(1) / std::max(largest, Real(1e-6));
        }
        Real trial_loss = loss_value;
        bool accepted = false;
        for (int backtrack = 0; backtrack < 30; ++backtrack) {
            for (std::size_t j = 0; j < dimensions; ++j) {
                trial[j] = clamp_torque(j, x[j] + step * direction[j]);
            }
            Real move = Real(0);
            for (std::size_t j = 0; j < dimensions; ++j) {
                move += gradient[j] * (trial[j] - x[j]);
            }
            if (!(move < -Real(1e-16))) {
                step *= Real(0.5);
                if (!(step > Real(1e-12))) {
                    break;
                }
                continue;
            }
            trial_loss = evaluate(trial, trial_gradient);
            if (trial_loss <= loss_value + Real(1e-4) * move) {
                accepted = true;
                break;
            }
            step *= Real(0.5);
            if (!(step > Real(1e-12))) {
                break;
            }
        }

        if (!accepted) {
            // L-BFGS 方向没有可接受步长时退回最速下降，再失败就结束
            Real largest = Real(0);
            for (std::size_t j = 0; j < dimensions; ++j) {
                direction[j] = -gradient[j];
                largest = std::max(largest, std::fabs(gradient[j]));
            }
            step = Real(1) / std::max(largest, Real(1e-6));
            for (int backtrack = 0; backtrack < 30; ++backtrack) {
                for (std::size_t j = 0; j < dimensions; ++j) {
                    trial[j] = clamp_torque(j, x[j] + step * direction[j]);
                }
                Real move = Real(0);
                for (std::size_t j = 0; j < dimensions; ++j) {
                    move += gradient[j] * (trial[j] - x[j]);
                }
                if (!(move < -Real(1e-16))) {
                    step *= Real(0.5);
                    if (!(step > Real(1e-12))) {
                        break;
                    }
                    continue;
                }
                trial_loss = evaluate(trial, trial_gradient);
                if (trial_loss <= loss_value + Real(1e-4) * move) {
                    accepted = true;
                    break;
                }
                step *= Real(0.5);
                if (!(step > Real(1e-12))) {
                    break;
                }
            }
        }
        if (!accepted) {
            break;
        }

        // 用实际步与梯度差更新 L-BFGS 历史
        std::vector<Real> s(dimensions);
        std::vector<Real> y(dimensions);
        for (std::size_t j = 0; j < dimensions; ++j) {
            s[j] = trial[j] - x[j];
            y[j] = trial_gradient[j] - gradient[j];
        }
        const Real sy = detail::vector_dot(s, y);
        const Real ss = detail::vector_dot(s, s);
        const Real yy = detail::vector_dot(y, y);
        if (sy > Real(1e-12) * std::sqrt(ss * yy) && sy > Real(0)) {
            s_history.push_back(std::move(s));
            y_history.push_back(std::move(y));
            rho_history.push_back(Real(1) / sy);
            if (s_history.size() > memory) {
                s_history.erase(s_history.begin());
                y_history.erase(y_history.begin());
                rho_history.erase(rho_history.begin());
            }
        }
        for (std::size_t j = 0; j < dimensions; ++j) {
            x[j] = trial[j];
            gradient[j] = trial_gradient[j];
        }
        loss_value = trial_loss;
        ++iteration;
    }

    for (std::size_t step = 0; step < count; ++step) {
        solution_[step] = math::Vector<Real, 4>::from(
            x[step * 4], x[step * 4 + 1], x[step * 4 + 2], x[step * 4 + 3]);
    }
    previous_torque_ = solution_[0];
    have_solution_ = true;
    if (info != nullptr) {
        info->loss = loss_value;
        info->iterations = iteration;
        info->evaluations = evaluations;
        info->converged = converged;
        info->projected_gradient = projected_gradient;
    }
    return solution_[0];
}

}  // namespace mpc
}  // namespace tcrg::gimbal_dynamics
