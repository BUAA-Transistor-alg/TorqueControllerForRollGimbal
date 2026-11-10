#include "gimbal_dynamics/stage_data.hpp"

#include "gimbal_dynamics/model_evaluation.hpp"

namespace tcrg::gimbal_dynamics {
namespace detail {

void evaluate_stage_data(const ScalarModelParameters<Real>& model,
                         const TerrainFrame<Real>& frame, const BaseState& base,
                         const simulation::State& state, const math::Vector4d& torque,
                         StageData& out) {
    out.base = base;
    out.angle = state.angle;
    out.qdot = state.velocity;
    out.chain = evaluate_chain(model, frame, base, out.angle);
    const math::Matrix4d mass = evaluate_mass_matrix(model, out.chain, &out.mass_internals);
    const math::Vector4d bias =
        evaluate_bias(model, out.chain, base, out.qdot, &out.bias_internals, &out.mass_internals);
    const math::Vector4d right_hand_side = torque - elementwise(model.damping, out.qdot) - bias;
    out.factor.factorize(mass);
    if (!out.factor.valid) {
        throw std::runtime_error("惯量矩阵不是正定矩阵");
    }
    out.acceleration = out.factor.solve(right_hand_side);
}

}  // namespace detail
}  // namespace tcrg::gimbal_dynamics
