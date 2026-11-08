#include "stage_data.hpp"

#include "model_evaluation.hpp"

namespace gimbal {
namespace detail {

void evaluate_stage_data(const ScalarModelParameters<double>& model,
                         const ScalarTerrain<double>& terrain, const BaseState& base,
                         const simulation::State& state, const Vector4<double>& torque,
                         StageData& out) {
    out.base = base;
    out.angle = state.angle;
    out.qdot = state.velocity;
    out.torque = torque;
    out.chain = evaluate_chain(model, terrain, base, out.angle);
    out.mass = evaluate_mass_matrix(model, out.chain, &out.mass_internals);
    out.bias = evaluate_bias(model, out.chain, base, out.qdot, &out.bias_internals,
                             &out.mass_internals);
    out.right_hand_side = torque - model.damping.cwiseProduct(out.qdot) - out.bias;
    out.acceleration = solve_mass(out.mass, out.right_hand_side);
}

}  // namespace detail
}  // namespace gimbal
