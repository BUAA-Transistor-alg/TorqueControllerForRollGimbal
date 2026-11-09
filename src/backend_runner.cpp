#include <algorithm>
#include <cstddef>
#include <vector>

#include "gimbal_dynamics/backend_abi.hpp"
#include "gimbal_dynamics/control_gradient.hpp"
#include "gimbal_dynamics/dynamics_model.hpp"
#include "gimbal_dynamics/parameter_gradient.hpp"
#include "gimbal_dynamics/simulation.hpp"
#include "thread_pool.hpp"

#ifndef GIMBAL_BACKEND_PREFIX
#error "GIMBAL_BACKEND_PREFIX 必须定义"
#endif

#define GIMBAL_ABI_JOIN2(a, b) a##b
#define GIMBAL_ABI_JOIN(a, b) GIMBAL_ABI_JOIN2(a, b)
#define GIMBAL_ABI_FN(name) GIMBAL_ABI_JOIN(GIMBAL_BACKEND_PREFIX, name)

using gimbal::Real;
namespace math = gimbal::math;

namespace {

// 每个精度后端各有一份自己的上下文类型，通过不透明指针暴露给处理器
struct BackendContext {
    explicit BackendContext(const gimbal_abi::ContextConfig& config)
        : pool(config.thread_count > 1 ? config.thread_count - 1 : 0),
          max_batch(config.max_batch == 0 ? 1 : config.max_batch) {}

    gimbal::detail::ThreadPool pool;
    std::size_t max_batch = 1;
    gimbal::control_gradient::Workspace control_single;
    gimbal::parameter_gradient::Workspace parameter_single;
    gimbal::control_gradient::BatchWorkspace control_batch;
    gimbal::parameter_gradient::BatchWorkspace parameter_batch;
};

BackendContext* as_context(GimbalAbiContext* pointer) {
    return reinterpret_cast<BackendContext*>(pointer);
}

GimbalAbiContext* as_abi(BackendContext* pointer) {
    return reinterpret_cast<GimbalAbiContext*>(pointer);
}

}  // namespace

namespace {

gimbal::ModelParameters from_abi_model(const void* pointer) {
    return *static_cast<const gimbal::ModelParameters*>(pointer);
}

gimbal::Terrain from_abi_terrain(const void* pointer) {
    return *static_cast<const gimbal::Terrain*>(pointer);
}

}  // namespace

extern "C" {

GimbalAbiContext* GIMBAL_ABI_FN(create)(const gimbal_abi::ContextConfig* config) {
    if (config == nullptr) {
        return nullptr;
    }
    try {
        return as_abi(new BackendContext(*config));
    } catch (...) {
        return nullptr;
    }
}

void GIMBAL_ABI_FN(destroy)(GimbalAbiContext* context) { delete as_context(context); }

void GIMBAL_ABI_FN(single_step)(GimbalAbiContext* context,
                                const gimbal_abi::SingleStepInput* input,
                                gimbal_abi::SingleStepOutput* output) {
    (void)context;
    gimbal::ModelParameters model = from_abi_model(input->model);
    gimbal::Terrain terrain = from_abi_terrain(input->terrain);
    const gimbal::BaseState& base = *static_cast<const gimbal::BaseState*>(input->base);
    const math::Vector<Real, 4>& angle = *static_cast<const math::Vector<Real, 4>*>(input->angle);
    const math::Vector<Real, 4>& velocity =
        *static_cast<const math::Vector<Real, 4>*>(input->velocity);
    const math::Vector<Real, 4>& torque =
        *static_cast<const math::Vector<Real, 4>*>(input->torque);
    *static_cast<math::Vector<Real, 4>*>(output->acceleration) =
        gimbal::forward_dynamics(model, terrain, base, angle, velocity, torque);
}

void GIMBAL_ABI_FN(forward)(GimbalAbiContext* context, const gimbal_abi::ForwardInput* input,
                            gimbal_abi::ForwardOutput* output) {
    (void)context;
    gimbal::ModelParameters model = from_abi_model(input->model);
    gimbal::Terrain terrain = from_abi_terrain(input->terrain);
    const gimbal::simulation::State& initial =
        *static_cast<const gimbal::simulation::State*>(input->initial);
    const gimbal::BaseState* base = static_cast<const gimbal::BaseState*>(input->base);
    const math::Vector<Real, 4>* torque =
        static_cast<const math::Vector<Real, 4>*>(input->torque);
    const gimbal::simulation::LossSpec& loss =
        *static_cast<const gimbal::simulation::LossSpec*>(input->loss);
    gimbal::simulation::TrajectorySpec spec{Real(input->step), input->refinement};
    gimbal::simulation::Trace trace;
    *static_cast<Real*>(output->loss) = gimbal::simulation::simulate(
        model, terrain, initial, base, torque, input->num_steps, spec, loss, trace);
    gimbal::simulation::State* step_end = static_cast<gimbal::simulation::State*>(output->step_end);
    for (std::size_t k = 0; k < input->num_steps; ++k) {
        step_end[k] = trace.step_end[k];
    }
    if (output->stage_state != nullptr) {
        gimbal::simulation::State* stage_state =
            static_cast<gimbal::simulation::State*>(output->stage_state);
        for (std::size_t i = 0; i < trace.stage_state.size(); ++i) {
            stage_state[i] = trace.stage_state[i];
        }
    }
    if (output->stage_base != nullptr) {
        gimbal::BaseState* stage_base = static_cast<gimbal::BaseState*>(output->stage_base);
        for (std::size_t i = 0; i < trace.stage_base.size(); ++i) {
            stage_base[i] = trace.stage_base[i];
        }
    }
}

void GIMBAL_ABI_FN(forward_batch)(GimbalAbiContext* context, std::size_t batch,
                                  const gimbal_abi::BatchGradientInput* input,
                                  gimbal_abi::BatchForwardOutput* output) {
    gimbal::ModelParameters model = from_abi_model(input->model);
    gimbal::Terrain terrain = from_abi_terrain(input->terrain);
    const gimbal::simulation::State* initial =
        static_cast<const gimbal::simulation::State*>(input->initial);
    const gimbal::BaseState* base = static_cast<const gimbal::BaseState*>(input->base);
    const math::Vector<Real, 4>* torque =
        static_cast<const math::Vector<Real, 4>*>(input->torque);
    const gimbal::simulation::LossSpec* losses =
        static_cast<const gimbal::simulation::LossSpec*>(input->losses);
    gimbal::simulation::TrajectorySpec spec{Real(input->step), input->refinement};
    gimbal::simulation::State* step_end = static_cast<gimbal::simulation::State*>(output->step_end);
    gimbal::simulation::State* final_state =
        static_cast<gimbal::simulation::State*>(output->final_state);
    std::vector<Real> loss_values(batch, Real(0));
    BackendContext* const ctx = as_context(context);
    gimbal::detail::ThreadPool& pool = ctx->pool;
    const std::size_t participants = static_cast<std::size_t>(pool.participant_count());
    const std::size_t chunk = std::max<std::size_t>(1, batch / participants);
    pool.parallel_dynamic(
        batch,
        [&](int, std::size_t sample) {
            const std::size_t offset = sample * input->num_steps;
            gimbal::simulation::Trace trace;
            loss_values[sample] = gimbal::simulation::simulate(
                model, terrain, initial[sample], base + offset, torque + offset,
                input->num_steps, spec, losses[sample], trace);
            for (std::size_t k = 0; k < input->num_steps; ++k) {
                step_end[offset + k] = trace.step_end[k];
            }
            final_state[sample] = trace.step_end.back();
        },
        chunk);
    Real total = Real(0);
    for (std::size_t sample = 0; sample < batch; ++sample) {
        total += loss_values[sample];
    }
    *static_cast<Real*>(output->mean_loss) = total / static_cast<Real>(batch);
}

void GIMBAL_ABI_FN(control_gradient)(GimbalAbiContext* context,
                                     const gimbal_abi::GradientInput* input,
                                     gimbal_abi::GradientOutput* output) {
    gimbal::ModelParameters model = from_abi_model(input->model);
    gimbal::Terrain terrain = from_abi_terrain(input->terrain);
    const gimbal::simulation::State& initial =
        *static_cast<const gimbal::simulation::State*>(input->initial);
    const gimbal::BaseState* base = static_cast<const gimbal::BaseState*>(input->base);
    const math::Vector<Real, 4>* torque =
        static_cast<const math::Vector<Real, 4>*>(input->torque);
    const gimbal::simulation::LossSpec& loss =
        *static_cast<const gimbal::simulation::LossSpec*>(input->loss);
    gimbal::simulation::TrajectorySpec spec{Real(input->step), input->refinement};
    BackendContext* const ctx = as_context(context);
    gimbal::simulation::State* final_state =
        static_cast<gimbal::simulation::State*>(output->final_state);
    *static_cast<Real*>(output->loss) = gimbal::control_gradient::simulateAndGradient(
        model, terrain, initial, base, torque, input->num_steps, spec, loss,
        ctx->control_single, static_cast<math::Vector<Real, 4>*>(output->gradient),
        final_state, &ctx->pool);
}

void GIMBAL_ABI_FN(control_gradient_batch)(GimbalAbiContext* context, std::size_t batch,
                                           const gimbal_abi::BatchGradientInput* input,
                                           gimbal_abi::BatchGradientOutput* output) {
    gimbal::ModelParameters model = from_abi_model(input->model);
    gimbal::Terrain terrain = from_abi_terrain(input->terrain);
    const gimbal::simulation::State* initial =
        static_cast<const gimbal::simulation::State*>(input->initial);
    const gimbal::BaseState* base = static_cast<const gimbal::BaseState*>(input->base);
    const math::Vector<Real, 4>* torque =
        static_cast<const math::Vector<Real, 4>*>(input->torque);
    const gimbal::simulation::LossSpec* losses =
        static_cast<const gimbal::simulation::LossSpec*>(input->losses);
    gimbal::simulation::TrajectorySpec spec{Real(input->step), input->refinement};
    BackendContext* const ctx = as_context(context);
    *static_cast<Real*>(output->mean_loss) = gimbal::control_gradient::simulateAndGradientBatch(
        model, terrain, initial, base, torque, batch, input->num_steps, spec, losses,
        ctx->control_batch, static_cast<math::Vector<Real, 4>*>(output->gradient),
        static_cast<gimbal::simulation::State*>(output->final_state), ctx->pool);
}

void GIMBAL_ABI_FN(parameter_sensitivity)(GimbalAbiContext* context,
                                          const gimbal_abi::GradientInput* input,
                                          gimbal_abi::GradientOutput* output) {
    gimbal::ModelParameters model = from_abi_model(input->model);
    gimbal::Terrain terrain = from_abi_terrain(input->terrain);
    const gimbal::simulation::State& initial =
        *static_cast<const gimbal::simulation::State*>(input->initial);
    const gimbal::BaseState* base = static_cast<const gimbal::BaseState*>(input->base);
    const math::Vector<Real, 4>* torque =
        static_cast<const math::Vector<Real, 4>*>(input->torque);
    const gimbal::simulation::LossSpec& loss =
        *static_cast<const gimbal::simulation::LossSpec*>(input->loss);
    gimbal::simulation::TrajectorySpec spec{Real(input->step), input->refinement};
    BackendContext* const ctx = as_context(context);
    gimbal::simulation::State* final_state =
        static_cast<gimbal::simulation::State*>(output->final_state);
    *static_cast<Real*>(output->loss) = gimbal::parameter_gradient::simulateAndSensitivity(
        model, terrain, initial, base, torque, input->num_steps, spec, loss,
        ctx->parameter_single, static_cast<gimbal::parameter_gradient::Gradient*>(
                                   output->gradient),
        final_state, &ctx->pool);
}

void GIMBAL_ABI_FN(parameter_sensitivity_batch)(GimbalAbiContext* context, std::size_t batch,
                                                const gimbal_abi::BatchGradientInput* input,
                                                gimbal_abi::BatchGradientOutput* output) {
    gimbal::ModelParameters model = from_abi_model(input->model);
    gimbal::Terrain terrain = from_abi_terrain(input->terrain);
    const gimbal::simulation::State* initial =
        static_cast<const gimbal::simulation::State*>(input->initial);
    const gimbal::BaseState* base = static_cast<const gimbal::BaseState*>(input->base);
    const math::Vector<Real, 4>* torque =
        static_cast<const math::Vector<Real, 4>*>(input->torque);
    const gimbal::simulation::LossSpec* losses =
        static_cast<const gimbal::simulation::LossSpec*>(input->losses);
    gimbal::simulation::TrajectorySpec spec{Real(input->step), input->refinement};
    BackendContext* const ctx = as_context(context);
    *static_cast<Real*>(output->mean_loss) =
        gimbal::parameter_gradient::simulateAndSensitivityBatch(
            model, terrain, initial, base, torque, batch, input->num_steps, spec, losses,
            ctx->parameter_batch,
            static_cast<gimbal::parameter_gradient::Gradient*>(output->gradient),
            static_cast<gimbal::simulation::State*>(output->final_state), ctx->pool);
}

}  // extern "C"
