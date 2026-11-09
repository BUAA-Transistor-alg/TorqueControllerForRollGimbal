#pragma once

// 公开处理器：按精度模板化，构造时给定线程数并在构造阶段分配线程，
// 之后用 run/runBatch 传入输入结构体并写回输出结构体，批量大小由每次调用指定。
// 线程数、精度均不依赖编译选项或环境变量。

#include <cstddef>
#include <stdexcept>
#include <vector>

#include "gimbal_dynamics/backend_abi.hpp"
#include "gimbal_dynamics/dynamics_model.hpp"
#include "gimbal_dynamics/math_types.hpp"
#include "gimbal_dynamics/parameters.hpp"
#include "gimbal_dynamics/simulation.hpp"

namespace gimbal {
namespace detail {

template <typename Real>
struct BackendTraits;

template <>
struct BackendTraits<double> {
    static GimbalAbiContext* create(const gimbal_abi::ContextConfig& config) {
        return gimbal_backend_double_create(&config);
    }
    static void destroy(GimbalAbiContext* context) { gimbal_backend_double_destroy(context); }
    static void single_step(GimbalAbiContext* context, const gimbal_abi::SingleStepInput* input,
                            gimbal_abi::SingleStepOutput* output) {
        gimbal_backend_double_single_step(context, input, output);
    }
    static void forward(GimbalAbiContext* context, const gimbal_abi::ForwardInput* input,
                        gimbal_abi::ForwardOutput* output) {
        gimbal_backend_double_forward(context, input, output);
    }
    static void forward_batch(GimbalAbiContext* context, std::size_t batch,
                              const gimbal_abi::BatchGradientInput* input,
                              gimbal_abi::BatchForwardOutput* output) {
        gimbal_backend_double_forward_batch(context, batch, input, output);
    }
    static void control_gradient(GimbalAbiContext* context,
                                 const gimbal_abi::GradientInput* input,
                                 gimbal_abi::GradientOutput* output) {
        gimbal_backend_double_control_gradient(context, input, output);
    }
    static void control_gradient_batch(GimbalAbiContext* context, std::size_t batch,
                                       const gimbal_abi::BatchGradientInput* input,
                                       gimbal_abi::BatchGradientOutput* output) {
        gimbal_backend_double_control_gradient_batch(context, batch, input, output);
    }
    static void parameter_sensitivity(GimbalAbiContext* context,
                                      const gimbal_abi::GradientInput* input,
                                      gimbal_abi::GradientOutput* output) {
        gimbal_backend_double_parameter_sensitivity(context, input, output);
    }
    static void parameter_sensitivity_batch(GimbalAbiContext* context, std::size_t batch,
                                            const gimbal_abi::BatchGradientInput* input,
                                            gimbal_abi::BatchGradientOutput* output) {
        gimbal_backend_double_parameter_sensitivity_batch(context, batch, input, output);
    }
};

template <>
struct BackendTraits<float> {
    static GimbalAbiContext* create(const gimbal_abi::ContextConfig& config) {
        return gimbal_backend_float_create(&config);
    }
    static void destroy(GimbalAbiContext* context) { gimbal_backend_float_destroy(context); }
    static void single_step(GimbalAbiContext* context, const gimbal_abi::SingleStepInput* input,
                            gimbal_abi::SingleStepOutput* output) {
        gimbal_backend_float_single_step(context, input, output);
    }
    static void forward(GimbalAbiContext* context, const gimbal_abi::ForwardInput* input,
                        gimbal_abi::ForwardOutput* output) {
        gimbal_backend_float_forward(context, input, output);
    }
    static void forward_batch(GimbalAbiContext* context, std::size_t batch,
                              const gimbal_abi::BatchGradientInput* input,
                              gimbal_abi::BatchForwardOutput* output) {
        gimbal_backend_float_forward_batch(context, batch, input, output);
    }
    static void control_gradient(GimbalAbiContext* context,
                                 const gimbal_abi::GradientInput* input,
                                 gimbal_abi::GradientOutput* output) {
        gimbal_backend_float_control_gradient(context, input, output);
    }
    static void control_gradient_batch(GimbalAbiContext* context, std::size_t batch,
                                       const gimbal_abi::BatchGradientInput* input,
                                       gimbal_abi::BatchGradientOutput* output) {
        gimbal_backend_float_control_gradient_batch(context, batch, input, output);
    }
    static void parameter_sensitivity(GimbalAbiContext* context,
                                      const gimbal_abi::GradientInput* input,
                                      gimbal_abi::GradientOutput* output) {
        gimbal_backend_float_parameter_sensitivity(context, input, output);
    }
    static void parameter_sensitivity_batch(GimbalAbiContext* context, std::size_t batch,
                                            const gimbal_abi::BatchGradientInput* input,
                                            gimbal_abi::BatchGradientOutput* output) {
        gimbal_backend_float_parameter_sensitivity_batch(context, batch, input, output);
    }
};

}  // namespace detail

// ---- 公开输入/输出结构体 ----

template <typename Real>
struct SingleStepInputT {
    ModelParametersT<Real> model;
    TerrainT<Real> terrain;
    BaseStateT<Real> base;
    math::Vector<Real, 4> angle = math::Vector<Real, 4>::zero();
    math::Vector<Real, 4> velocity = math::Vector<Real, 4>::zero();
    math::Vector<Real, 4> torque = math::Vector<Real, 4>::zero();
};

template <typename Real>
struct SingleStepOutputT {
    math::Vector<Real, 4> acceleration = math::Vector<Real, 4>::zero();
};

template <typename Real>
struct ForwardInputT {
    ModelParametersT<Real> model;
    TerrainT<Real> terrain;
    simulation::StateT<Real> initial;
    const BaseStateT<Real>* base = nullptr;    // num_steps
    const math::Vector<Real, 4>* torque = nullptr;  // num_steps
    std::size_t num_steps = 0;
    simulation::TrajectorySpecT<Real> spec;
    simulation::LossSpecT<Real> loss;
};

template <typename Real>
struct ForwardOutputT {
    Real loss = Real(0);
    std::vector<simulation::StateT<Real>> step_end;
    std::vector<simulation::StateT<Real>> stage_state;
    std::vector<BaseStateT<Real>> stage_base;
};

template <typename Real>
struct ControlGradientInputT {
    ModelParametersT<Real> model;
    TerrainT<Real> terrain;
    simulation::StateT<Real> initial;
    const BaseStateT<Real>* base = nullptr;
    const math::Vector<Real, 4>* torque = nullptr;
    std::size_t num_steps = 0;
    simulation::TrajectorySpecT<Real> spec;
    simulation::LossSpecT<Real> loss;
};

template <typename Real>
struct ControlGradientOutputT {
    Real loss = Real(0);
    simulation::StateT<Real> final_state;
    std::vector<math::Vector<Real, 4>> gradient;  // num_steps
};

template <typename Real>
struct ControlGradientBatchInputT {
    ModelParametersT<Real> model;
    TerrainT<Real> terrain;
    const simulation::StateT<Real>* initial = nullptr;  // batch
    const BaseStateT<Real>* base = nullptr;              // batch*num_steps
    const math::Vector<Real, 4>* torque = nullptr;       // batch*num_steps
    std::size_t num_steps = 0;
    simulation::TrajectorySpecT<Real> spec;
    const simulation::LossSpecT<Real>* loss = nullptr;   // batch
};

template <typename Real>
struct ControlGradientBatchOutputT {
    Real mean_loss = Real(0);
    std::vector<math::Vector<Real, 4>> gradient;  // batch*num_steps
    std::vector<simulation::StateT<Real>> final_state;  // batch
};

template <typename Real>
struct ForwardBatchOutputT {
    Real mean_loss = Real(0);
    std::vector<simulation::StateT<Real>> step_end;  // batch*num_steps
    std::vector<simulation::StateT<Real>> final_state;  // batch
};

template <typename Real>
struct ParameterSensitivityOutputT {
    Real loss = Real(0);
    simulation::StateT<Real> final_state;
    math::Vector<Real, parameters::kParameterCount> gradient =
        math::Vector<Real, parameters::kParameterCount>::zero();
};

template <typename Real>
struct ParameterSensitivityInputT {
    ModelParametersT<Real> model;
    TerrainT<Real> terrain;
    simulation::StateT<Real> initial;
    const BaseStateT<Real>* base = nullptr;
    const math::Vector<Real, 4>* torque = nullptr;
    std::size_t num_steps = 0;
    simulation::TrajectorySpecT<Real> spec;
    simulation::LossSpecT<Real> loss;
};

template <typename Real>
struct ParameterSensitivityBatchInputT {
    ModelParametersT<Real> model;
    TerrainT<Real> terrain;
    const simulation::StateT<Real>* initial = nullptr;
    const BaseStateT<Real>* base = nullptr;
    const math::Vector<Real, 4>* torque = nullptr;
    std::size_t num_steps = 0;
    simulation::TrajectorySpecT<Real> spec;
    const simulation::LossSpecT<Real>* loss = nullptr;
};

template <typename Real>
struct ParameterSensitivityBatchOutputT {
    Real mean_loss = Real(0);
    std::vector<math::Vector<Real, parameters::kParameterCount>> gradient;  // batch
    std::vector<simulation::StateT<Real>> final_state;  // batch
};

// ---- 处理器 ----

// 单步动力学处理器
template <typename Real>
class SingleStepDynamics {
public:
    explicit SingleStepDynamics(int thread_count) {
        gimbal_abi::ContextConfig config{thread_count};
        context_ = detail::BackendTraits<Real>::create(config);
        if (context_ == nullptr) {
            throw std::runtime_error("后端线程池创建失败");
        }
    }
    ~SingleStepDynamics() {
        if (context_ != nullptr) {
            detail::BackendTraits<Real>::destroy(context_);
        }
    }
    SingleStepDynamics(const SingleStepDynamics&) = delete;
    SingleStepDynamics& operator=(const SingleStepDynamics&) = delete;

    void run(const SingleStepInputT<Real>& input, SingleStepOutputT<Real>& output) const {
        gimbal_abi::SingleStepInput abi{};
        abi.model = &input.model;
        abi.terrain = &input.terrain;
        abi.base = &input.base;
        abi.angle = &input.angle;
        abi.velocity = &input.velocity;
        abi.torque = &input.torque;
        gimbal_abi::SingleStepOutput result{};
        result.acceleration = &output.acceleration;
        detail::BackendTraits<Real>::single_step(context_, &abi, &result);
    }

private:
    GimbalAbiContext* context_ = nullptr;
};

// 前向仿真处理器
template <typename Real>
class ForwardSimulation {
public:
    explicit ForwardSimulation(int thread_count) {
        gimbal_abi::ContextConfig config{thread_count};
        context_ = detail::BackendTraits<Real>::create(config);
        if (context_ == nullptr) {
            throw std::runtime_error("后端线程池创建失败");
        }
    }
    ~ForwardSimulation() {
        if (context_ != nullptr) {
            detail::BackendTraits<Real>::destroy(context_);
        }
    }
    ForwardSimulation(const ForwardSimulation&) = delete;
    ForwardSimulation& operator=(const ForwardSimulation&) = delete;

    void run(const ForwardInputT<Real>& input, ForwardOutputT<Real>& output) const {
        const std::size_t stages =
            input.num_steps * static_cast<std::size_t>(input.spec.refinement) * 4;
        output.step_end.resize(input.num_steps);
        output.stage_state.resize(stages);
        output.stage_base.resize(stages);
        gimbal_abi::ForwardInput abi{};
        abi.model = &input.model;
        abi.terrain = &input.terrain;
        abi.initial = &input.initial;
        abi.base = input.base;
        abi.torque = input.torque;
        abi.num_steps = input.num_steps;
        abi.step = static_cast<double>(input.spec.step);
        abi.refinement = input.spec.refinement;
        abi.loss = &input.loss;
        gimbal_abi::ForwardOutput result{};
        result.loss = &output.loss;
        result.step_end = output.step_end.data();
        result.stage_state = output.stage_state.data();
        result.stage_base = output.stage_base.data();
        detail::BackendTraits<Real>::forward(context_, &abi, &result);
    }

    void runBatch(std::size_t batch, const ControlGradientBatchInputT<Real>& input,
                  ForwardBatchOutputT<Real>& output) const {
        output.step_end.resize(batch * input.num_steps);
        output.final_state.resize(batch);
        gimbal_abi::BatchGradientInput abi{};
        abi.model = &input.model;
        abi.terrain = &input.terrain;
        abi.initial = input.initial;
        abi.base = input.base;
        abi.torque = input.torque;
        abi.num_steps = input.num_steps;
        abi.step = static_cast<double>(input.spec.step);
        abi.refinement = input.spec.refinement;
        abi.losses = input.loss;
        gimbal_abi::BatchForwardOutput result{};
        result.mean_loss = &output.mean_loss;
        result.step_end = output.step_end.data();
        result.final_state = output.final_state.data();
        detail::BackendTraits<Real>::forward_batch(context_, batch, &abi, &result);
    }

private:
    GimbalAbiContext* context_ = nullptr;
};

// 控制量梯度处理器
template <typename Real>
class ControlGradient {
public:
    explicit ControlGradient(int thread_count) {
        gimbal_abi::ContextConfig config{thread_count};
        context_ = detail::BackendTraits<Real>::create(config);
        if (context_ == nullptr) {
            throw std::runtime_error("后端线程池创建失败");
        }
    }
    ~ControlGradient() {
        if (context_ != nullptr) {
            detail::BackendTraits<Real>::destroy(context_);
        }
    }
    ControlGradient(const ControlGradient&) = delete;
    ControlGradient& operator=(const ControlGradient&) = delete;

    void run(const ControlGradientInputT<Real>& input, ControlGradientOutputT<Real>& output) const {
        output.gradient.resize(input.num_steps);
        gimbal_abi::GradientInput abi{};
        abi.model = &input.model;
        abi.terrain = &input.terrain;
        abi.initial = &input.initial;
        abi.base = input.base;
        abi.torque = input.torque;
        abi.num_steps = input.num_steps;
        abi.step = static_cast<double>(input.spec.step);
        abi.refinement = input.spec.refinement;
        abi.loss = &input.loss;
        gimbal_abi::GradientOutput result{};
        result.loss = &output.loss;
        result.final_state = &output.final_state;
        result.gradient = output.gradient.data();
        detail::BackendTraits<Real>::control_gradient(context_, &abi, &result);
    }

    void runBatch(std::size_t batch, const ControlGradientBatchInputT<Real>& input,
                  ControlGradientBatchOutputT<Real>& output) const {
        output.gradient.resize(batch * input.num_steps);
        output.final_state.resize(batch);
        gimbal_abi::BatchGradientInput abi{};
        abi.model = &input.model;
        abi.terrain = &input.terrain;
        abi.initial = input.initial;
        abi.base = input.base;
        abi.torque = input.torque;
        abi.num_steps = input.num_steps;
        abi.step = static_cast<double>(input.spec.step);
        abi.refinement = input.spec.refinement;
        abi.losses = input.loss;
        gimbal_abi::BatchGradientOutput result{};
        result.mean_loss = &output.mean_loss;
        result.gradient = output.gradient.data();
        result.final_state = output.final_state.data();
        detail::BackendTraits<Real>::control_gradient_batch(context_, batch, &abi, &result);
    }

private:
    GimbalAbiContext* context_ = nullptr;
};

// 参数灵敏度处理器
template <typename Real>
class ParameterSensitivity {
public:
    explicit ParameterSensitivity(int thread_count) {
        gimbal_abi::ContextConfig config{thread_count};
        context_ = detail::BackendTraits<Real>::create(config);
        if (context_ == nullptr) {
            throw std::runtime_error("后端线程池创建失败");
        }
    }
    ~ParameterSensitivity() {
        if (context_ != nullptr) {
            detail::BackendTraits<Real>::destroy(context_);
        }
    }
    ParameterSensitivity(const ParameterSensitivity&) = delete;
    ParameterSensitivity& operator=(const ParameterSensitivity&) = delete;

    void run(const ParameterSensitivityInputT<Real>& input,
             ParameterSensitivityOutputT<Real>& output) const {
        gimbal_abi::GradientInput abi{};
        abi.model = &input.model;
        abi.terrain = &input.terrain;
        abi.initial = &input.initial;
        abi.base = input.base;
        abi.torque = input.torque;
        abi.num_steps = input.num_steps;
        abi.step = static_cast<double>(input.spec.step);
        abi.refinement = input.spec.refinement;
        abi.loss = &input.loss;
        gimbal_abi::GradientOutput result{};
        result.loss = &output.loss;
        result.final_state = &output.final_state;
        result.gradient = &output.gradient;
        detail::BackendTraits<Real>::parameter_sensitivity(context_, &abi, &result);
    }

    void runBatch(std::size_t batch, const ParameterSensitivityBatchInputT<Real>& input,
                  ParameterSensitivityBatchOutputT<Real>& output) const {
        output.gradient.resize(batch);
        output.final_state.resize(batch);
        gimbal_abi::BatchGradientInput abi{};
        abi.model = &input.model;
        abi.terrain = &input.terrain;
        abi.initial = input.initial;
        abi.base = input.base;
        abi.torque = input.torque;
        abi.num_steps = input.num_steps;
        abi.step = static_cast<double>(input.spec.step);
        abi.refinement = input.spec.refinement;
        abi.losses = input.loss;
        gimbal_abi::BatchGradientOutput result{};
        result.mean_loss = &output.mean_loss;
        result.gradient = output.gradient.data();
        result.final_state = output.final_state.data();
        detail::BackendTraits<Real>::parameter_sensitivity_batch(context_, batch, &abi, &result);
    }

private:
    GimbalAbiContext* context_ = nullptr;
};

}  // namespace gimbal
