#include <algorithm>
#include <cstddef>
#include <vector>

#include "gimbal_dynamics/backend_abi.hpp"
#include "gimbal_dynamics/control_gradient.hpp"
#include "gimbal_dynamics/dynamics_model.hpp"
#include "gimbal_dynamics/parameter_gradient.hpp"
#include "gimbal_dynamics/simulation.hpp"
#include "gimbal_dynamics/thread_pool.hpp"

#ifndef GIMBAL_BACKEND_PREFIX
#error "GIMBAL_BACKEND_PREFIX 必须定义"
#endif

#define GIMBAL_ABI_JOIN2(a, b) a##b
#define GIMBAL_ABI_JOIN(a, b) GIMBAL_ABI_JOIN2(a, b)
#define GIMBAL_ABI_FN(name) GIMBAL_ABI_JOIN(GIMBAL_BACKEND_PREFIX, name)

using tcrg::gimbal_dynamics::Real;
namespace math = tcrg::gimbal_dynamics::math;

namespace {

// 每个精度后端各有一份自己的上下文类型，通过不透明指针暴露给处理器
struct BackendContext {
    explicit BackendContext(const tcrg_abi::ContextConfig& config)
        : pool(config.thread_count > 1 ? config.thread_count - 1 : 0) {}

    tcrg::gimbal_dynamics::detail::ThreadPool pool;
    tcrg::gimbal_dynamics::control_gradient::Workspace control_single;
    tcrg::gimbal_dynamics::parameter_gradient::Workspace parameter_single;
    tcrg::gimbal_dynamics::control_gradient::BatchWorkspace control_batch;
    tcrg::gimbal_dynamics::parameter_gradient::BatchWorkspace parameter_batch;
};

BackendContext* as_context(GimbalAbiContext* pointer) {
    return reinterpret_cast<BackendContext*>(pointer);
}

GimbalAbiContext* as_abi(BackendContext* pointer) {
    return reinterpret_cast<GimbalAbiContext*>(pointer);
}

}  // namespace

namespace {

tcrg::gimbal_dynamics::ModelParameters from_abi_model(const void* pointer) {
    return *static_cast<const tcrg::gimbal_dynamics::ModelParameters*>(pointer);
}

tcrg::gimbal_dynamics::Terrain from_abi_terrain(const void* pointer) {
    return *static_cast<const tcrg::gimbal_dynamics::Terrain*>(pointer);
}

// 批量调用：共享权重与逐样本参考轨迹组合出每个样本的损失配置
std::vector<tcrg::gimbal_dynamics::simulation::LossSpec> build_batch_losses(
    const tcrg_abi::BatchGradientInput* input, std::size_t batch) {
    const auto* weights =
        static_cast<const tcrg::gimbal_dynamics::simulation::LossWeightsT<Real>*>(
            input->loss_weights);
    const auto* target_angle = static_cast<const math::Vector<Real, 4>*>(input->target_angle);
    const auto* target_velocity =
        static_cast<const math::Vector<Real, 4>*>(input->target_velocity);
    std::vector<tcrg::gimbal_dynamics::simulation::LossSpec> losses(batch);
    for (std::size_t sample = 0; sample < batch; ++sample) {
        losses[sample].angle_weights = weights->angle_weights;
        losses[sample].velocity_weights = weights->velocity_weights;
        losses[sample].torque_weights = weights->torque_weights;
        losses[sample].torque_rate_weights = weights->torque_rate_weights;
        losses[sample].limit_weights = weights->limit_weights;
        losses[sample].limit_lower = weights->limit_lower;
        losses[sample].limit_upper = weights->limit_upper;
        losses[sample].limit_margin = weights->limit_margin;
        if (target_angle != nullptr) {
            losses[sample].target_angle = target_angle + sample * input->num_steps;
        }
        if (target_velocity != nullptr) {
            losses[sample].target_velocity = target_velocity + sample * input->num_steps;
        }
    }
    return losses;
}

}  // namespace

extern "C" {

GimbalAbiContext* GIMBAL_ABI_FN(create)(const tcrg_abi::ContextConfig* config) {
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
                                const tcrg_abi::SingleStepInput* input,
                                tcrg_abi::SingleStepOutput* output) {
    (void)context;
    tcrg::gimbal_dynamics::ModelParameters model = from_abi_model(input->model);
    tcrg::gimbal_dynamics::Terrain terrain = from_abi_terrain(input->terrain);
    const tcrg::gimbal_dynamics::BaseState& base = *static_cast<const tcrg::gimbal_dynamics::BaseState*>(input->base);
    const math::Vector<Real, 4>& angle = *static_cast<const math::Vector<Real, 4>*>(input->angle);
    const math::Vector<Real, 4>& velocity =
        *static_cast<const math::Vector<Real, 4>*>(input->velocity);
    const math::Vector<Real, 4>& torque =
        *static_cast<const math::Vector<Real, 4>*>(input->torque);
    *static_cast<math::Vector<Real, 4>*>(output->acceleration) =
        tcrg::gimbal_dynamics::forward_dynamics(model, terrain, base, angle, velocity, torque);
}

void GIMBAL_ABI_FN(forward)(GimbalAbiContext* context, const tcrg_abi::ForwardInput* input,
                            tcrg_abi::ForwardOutput* output) {
    (void)context;
    tcrg::gimbal_dynamics::ModelParameters model = from_abi_model(input->model);
    tcrg::gimbal_dynamics::Terrain terrain = from_abi_terrain(input->terrain);
    const tcrg::gimbal_dynamics::simulation::State& initial =
        *static_cast<const tcrg::gimbal_dynamics::simulation::State*>(input->initial);
    const tcrg::gimbal_dynamics::BaseState* base = static_cast<const tcrg::gimbal_dynamics::BaseState*>(input->base);
    const math::Vector<Real, 4>* torque =
        static_cast<const math::Vector<Real, 4>*>(input->torque);
    const tcrg::gimbal_dynamics::simulation::LossSpec& loss =
        *static_cast<const tcrg::gimbal_dynamics::simulation::LossSpec*>(input->loss);
    tcrg::gimbal_dynamics::simulation::TrajectorySpec spec{Real(input->step), input->refinement};
    tcrg::gimbal_dynamics::simulation::Trace trace;
    *static_cast<Real*>(output->loss) = tcrg::gimbal_dynamics::simulation::simulate(
        model, terrain, initial, base, torque, input->num_steps, spec, loss, trace);
    tcrg::gimbal_dynamics::simulation::State* step_end = static_cast<tcrg::gimbal_dynamics::simulation::State*>(output->step_end);
    for (std::size_t k = 0; k < input->num_steps; ++k) {
        step_end[k] = trace.step_end[k];
    }
    if (output->stage_state != nullptr) {
        tcrg::gimbal_dynamics::simulation::State* stage_state =
            static_cast<tcrg::gimbal_dynamics::simulation::State*>(output->stage_state);
        for (std::size_t i = 0; i < trace.stage_state.size(); ++i) {
            stage_state[i] = trace.stage_state[i];
        }
    }
    if (output->stage_base != nullptr) {
        tcrg::gimbal_dynamics::BaseState* stage_base = static_cast<tcrg::gimbal_dynamics::BaseState*>(output->stage_base);
        for (std::size_t i = 0; i < trace.stage_base.size(); ++i) {
            stage_base[i] = trace.stage_base[i];
        }
    }
}

void GIMBAL_ABI_FN(forward_batch)(GimbalAbiContext* context, std::size_t batch,
                                  const tcrg_abi::BatchGradientInput* input,
                                  tcrg_abi::BatchForwardOutput* output) {
    tcrg::gimbal_dynamics::ModelParameters model = from_abi_model(input->model);
    const tcrg::gimbal_dynamics::Terrain* terrains =
        static_cast<const tcrg::gimbal_dynamics::Terrain*>(input->terrain);
    const tcrg::gimbal_dynamics::simulation::State* initial =
        static_cast<const tcrg::gimbal_dynamics::simulation::State*>(input->initial);
    const tcrg::gimbal_dynamics::BaseState* base = static_cast<const tcrg::gimbal_dynamics::BaseState*>(input->base);
    const math::Vector<Real, 4>* torque =
        static_cast<const math::Vector<Real, 4>*>(input->torque);
    const std::vector<tcrg::gimbal_dynamics::simulation::LossSpec> losses =
        build_batch_losses(input, batch);
    tcrg::gimbal_dynamics::simulation::TrajectorySpec spec{Real(input->step), input->refinement};
    tcrg::gimbal_dynamics::simulation::State* step_end = static_cast<tcrg::gimbal_dynamics::simulation::State*>(output->step_end);
    tcrg::gimbal_dynamics::simulation::State* final_state =
        static_cast<tcrg::gimbal_dynamics::simulation::State*>(output->final_state);
    std::vector<Real> loss_values(batch, Real(0));
    BackendContext* const ctx = as_context(context);
    tcrg::gimbal_dynamics::detail::ThreadPool& pool = ctx->pool;
    const std::size_t participants = static_cast<std::size_t>(pool.participant_count());
    const std::size_t chunk = std::max<std::size_t>(1, batch / participants);
    pool.parallel_dynamic(
        batch,
        [&](int, std::size_t sample) {
            const std::size_t offset = sample * input->num_steps;
            tcrg::gimbal_dynamics::simulation::Trace trace;
            loss_values[sample] = tcrg::gimbal_dynamics::simulation::simulate(
                model, terrains[sample], initial[sample], base + offset, torque + offset,
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
                                     const tcrg_abi::GradientInput* input,
                                     tcrg_abi::GradientOutput* output) {
    tcrg::gimbal_dynamics::ModelParameters model = from_abi_model(input->model);
    tcrg::gimbal_dynamics::Terrain terrain = from_abi_terrain(input->terrain);
    const tcrg::gimbal_dynamics::simulation::State& initial =
        *static_cast<const tcrg::gimbal_dynamics::simulation::State*>(input->initial);
    const tcrg::gimbal_dynamics::BaseState* base = static_cast<const tcrg::gimbal_dynamics::BaseState*>(input->base);
    const math::Vector<Real, 4>* torque =
        static_cast<const math::Vector<Real, 4>*>(input->torque);
    const tcrg::gimbal_dynamics::simulation::LossSpec& loss =
        *static_cast<const tcrg::gimbal_dynamics::simulation::LossSpec*>(input->loss);
    tcrg::gimbal_dynamics::simulation::TrajectorySpec spec{Real(input->step), input->refinement};
    BackendContext* const ctx = as_context(context);
    tcrg::gimbal_dynamics::simulation::State* final_state =
        static_cast<tcrg::gimbal_dynamics::simulation::State*>(output->final_state);
    *static_cast<Real*>(output->loss) = tcrg::gimbal_dynamics::control_gradient::simulateAndGradient(
        model, terrain, initial, base, torque, input->num_steps, spec, loss,
        ctx->control_single, static_cast<math::Vector<Real, 4>*>(output->gradient),
        final_state, &ctx->pool);
}

void GIMBAL_ABI_FN(control_gradient_batch)(GimbalAbiContext* context, std::size_t batch,
                                           const tcrg_abi::BatchGradientInput* input,
                                           tcrg_abi::BatchGradientOutput* output) {
    tcrg::gimbal_dynamics::ModelParameters model = from_abi_model(input->model);
    const tcrg::gimbal_dynamics::Terrain* terrains =
        static_cast<const tcrg::gimbal_dynamics::Terrain*>(input->terrain);
    const tcrg::gimbal_dynamics::simulation::State* initial =
        static_cast<const tcrg::gimbal_dynamics::simulation::State*>(input->initial);
    const tcrg::gimbal_dynamics::BaseState* base = static_cast<const tcrg::gimbal_dynamics::BaseState*>(input->base);
    const math::Vector<Real, 4>* torque =
        static_cast<const math::Vector<Real, 4>*>(input->torque);
    const std::vector<tcrg::gimbal_dynamics::simulation::LossSpec> losses =
        build_batch_losses(input, batch);
    tcrg::gimbal_dynamics::simulation::TrajectorySpec spec{Real(input->step), input->refinement};
    BackendContext* const ctx = as_context(context);
    *static_cast<Real*>(output->mean_loss) = tcrg::gimbal_dynamics::control_gradient::simulateAndGradientBatch(
        model, terrains, initial, base, torque, batch, input->num_steps, spec, losses.data(),
        ctx->control_batch, static_cast<math::Vector<Real, 4>*>(output->gradient),
        static_cast<tcrg::gimbal_dynamics::simulation::State*>(output->final_state), ctx->pool);
}

void GIMBAL_ABI_FN(parameter_sensitivity)(GimbalAbiContext* context,
                                          const tcrg_abi::GradientInput* input,
                                          tcrg_abi::GradientOutput* output) {
    tcrg::gimbal_dynamics::ModelParameters model = from_abi_model(input->model);
    tcrg::gimbal_dynamics::Terrain terrain = from_abi_terrain(input->terrain);
    const tcrg::gimbal_dynamics::simulation::State& initial =
        *static_cast<const tcrg::gimbal_dynamics::simulation::State*>(input->initial);
    const tcrg::gimbal_dynamics::BaseState* base = static_cast<const tcrg::gimbal_dynamics::BaseState*>(input->base);
    const math::Vector<Real, 4>* torque =
        static_cast<const math::Vector<Real, 4>*>(input->torque);
    const tcrg::gimbal_dynamics::simulation::LossSpec& loss =
        *static_cast<const tcrg::gimbal_dynamics::simulation::LossSpec*>(input->loss);
    tcrg::gimbal_dynamics::simulation::TrajectorySpec spec{Real(input->step), input->refinement};
    BackendContext* const ctx = as_context(context);
    tcrg::gimbal_dynamics::simulation::State* final_state =
        static_cast<tcrg::gimbal_dynamics::simulation::State*>(output->final_state);
    *static_cast<Real*>(output->loss) = tcrg::gimbal_dynamics::parameter_gradient::simulateAndSensitivity(
        model, terrain, initial, base, torque, input->num_steps, spec, loss,
        ctx->parameter_single, static_cast<tcrg::gimbal_dynamics::parameter_gradient::Gradient*>(
                                   output->gradient),
        final_state, &ctx->pool);
}

void GIMBAL_ABI_FN(parameter_sensitivity_batch)(GimbalAbiContext* context, std::size_t batch,
                                                const tcrg_abi::BatchGradientInput* input,
                                                tcrg_abi::BatchGradientOutput* output) {
    tcrg::gimbal_dynamics::ModelParameters model = from_abi_model(input->model);
    const tcrg::gimbal_dynamics::Terrain* terrains =
        static_cast<const tcrg::gimbal_dynamics::Terrain*>(input->terrain);
    const tcrg::gimbal_dynamics::simulation::State* initial =
        static_cast<const tcrg::gimbal_dynamics::simulation::State*>(input->initial);
    const tcrg::gimbal_dynamics::BaseState* base = static_cast<const tcrg::gimbal_dynamics::BaseState*>(input->base);
    const math::Vector<Real, 4>* torque =
        static_cast<const math::Vector<Real, 4>*>(input->torque);
    const std::vector<tcrg::gimbal_dynamics::simulation::LossSpec> losses =
        build_batch_losses(input, batch);
    tcrg::gimbal_dynamics::simulation::TrajectorySpec spec{Real(input->step), input->refinement};
    BackendContext* const ctx = as_context(context);
    *static_cast<Real*>(output->mean_loss) =
        tcrg::gimbal_dynamics::parameter_gradient::simulateAndSensitivityBatch(
            model, terrains, initial, base, torque, batch, input->num_steps, spec, losses.data(),
            ctx->parameter_batch,
            static_cast<tcrg::gimbal_dynamics::parameter_gradient::Gradient*>(output->gradient),
            static_cast<tcrg::gimbal_dynamics::simulation::State*>(output->final_state), ctx->pool);
}

}  // extern "C"
