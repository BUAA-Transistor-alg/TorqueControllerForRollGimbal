#include "stage_data.hpp"

#include "model_evaluation.hpp"

namespace gimbal {
namespace detail {

void evaluate_stage_data(const ScalarModelParameters<Real>& model,
                         const TerrainFrame<Real>& frame, const BaseState& base,
                         const simulation::State& state, const math::Vector4d& torque,
                         StageData& out) {
    out.base = base;
    out.angle = state.angle;
    out.qdot = state.velocity;
    out.torque = torque;
    out.chain = evaluate_chain(model, frame, base, out.angle);
    out.mass = evaluate_mass_matrix(model, out.chain, &out.mass_internals);
    out.bias = evaluate_bias(model, out.chain, base, out.qdot, &out.bias_internals,
                             &out.mass_internals);
    out.right_hand_side = torque - elementwise(model.damping, out.qdot) - out.bias;
    out.factor.factorize(out.mass);
    if (!out.factor.valid) {
        throw std::runtime_error("惯量矩阵不是正定矩阵");
    }
    out.acceleration = out.factor.solve(out.right_hand_side);
}

}  // namespace detail
}  // namespace gimbal
