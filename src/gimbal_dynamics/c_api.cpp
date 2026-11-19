#include "gimbal_dynamics/c_api.h"

#include <string>
#include <vector>

#include "gimbal_dynamics/handlers.hpp"
#include "gimbal_dynamics/model_evaluation.hpp"

namespace {

thread_local std::string g_last_error;

void set_error(const char* message) { g_last_error = message == nullptr ? "" : message; }

tcrg::gimbal_dynamics::ModelParametersT<double> convert_model(const TcrgModel& source) {
    using tcrg::gimbal_dynamics::BodyParametersT;
    tcrg::gimbal_dynamics::ModelParametersT<double> model;
    for (int i = 0; i < 4; ++i) {
        BodyParametersT<double>& body = model.bodies[i];
        body.mass = source.mass[i];
        for (int c = 0; c < 3; ++c) {
            body.com[c] = source.com[i][c];
            model.joint_offsets[i][c] = source.joint_offsets[i][c];
        }
        body.inertia(0, 0) = source.inertia[i][0];
        body.inertia(1, 1) = source.inertia[i][1];
        body.inertia(2, 2) = source.inertia[i][2];
        body.inertia(0, 1) = source.inertia[i][3];
        body.inertia(1, 0) = source.inertia[i][3];
        body.inertia(0, 2) = source.inertia[i][4];
        body.inertia(2, 0) = source.inertia[i][4];
        body.inertia(1, 2) = source.inertia[i][5];
        body.inertia(2, 1) = source.inertia[i][5];
    }
    for (int i = 0; i < 4; ++i) {
        model.damping[i] = source.damping[i];
    }
    model.gravity = source.gravity;
    return model;
}

tcrg::gimbal_dynamics::TerrainT<double> convert_terrain(const TcrgTerrain& source) {
    tcrg::gimbal_dynamics::TerrainT<double> terrain;
    terrain.slope_angle = source.slope_angle;
    terrain.slope_direction = source.slope_direction;
    return terrain;
}

tcrg::gimbal_dynamics::BaseStateT<double> convert_base(const TcrgBaseState& source) {
    tcrg::gimbal_dynamics::BaseStateT<double> base;
    for (int c = 0; c < 3; ++c) {
        base.acceleration[c] = source.acceleration[c];
    }
    base.yaw = source.yaw;
    base.yaw_rate = source.yaw_rate;
    base.yaw_acceleration = source.yaw_acceleration;
    return base;
}

tcrg::gimbal_dynamics::simulation::StateT<double> convert_state(const TcrgState& source) {
    tcrg::gimbal_dynamics::simulation::StateT<double> state;
    for (int i = 0; i < 4; ++i) {
        state.angle[i] = source.angle[i];
        state.velocity[i] = source.velocity[i];
    }
    return state;
}

void store_state(const tcrg::gimbal_dynamics::simulation::StateT<double>& state,
                 TcrgState* target) {
    for (int i = 0; i < 4; ++i) {
        target->angle[i] = state.angle[i];
        target->velocity[i] = state.velocity[i];
    }
}

bool valid(const TcrgModel* model, const TcrgTerrain* terrain, const TcrgBaseState* base,
           const TcrgState* state, const double* torque) {
    return model != nullptr && terrain != nullptr && base != nullptr && state != nullptr &&
           torque != nullptr;
}

}  // namespace

struct TcrgContext {
    explicit TcrgContext(int threads)
        : handler(threads), parameter_sensitivity(threads) {}
    tcrg::gimbal_dynamics::ForwardSimulation<double> handler;
    tcrg::gimbal_dynamics::ParameterSensitivity<double> parameter_sensitivity;
};

extern "C" {

TcrgContext* tcrg_context_create(int thread_count) {
    try {
        return new TcrgContext(thread_count < 1 ? 1 : thread_count);
    } catch (const std::exception& error) {
        set_error(error.what());
        return nullptr;
    }
}

void tcrg_context_destroy(TcrgContext* context) {
    delete context;
}

int tcrg_forward_step(const TcrgContext* context, const TcrgModel* model,
                      const TcrgTerrain* terrain, const TcrgBaseState* base,
                      const TcrgState* current, const double torque[4], double step,
                      int refinement, TcrgState* next) {
    if (context == nullptr || !valid(model, terrain, base, current, torque) || next == nullptr) {
        set_error("参数为空指针");
        return 1;
    }
    try {
        tcrg::gimbal_dynamics::ForwardInputT<double> input;
        input.model = convert_model(*model);
        input.terrain = convert_terrain(*terrain);
        input.initial = convert_state(*current);
        tcrg::gimbal_dynamics::BaseStateT<double> base_state = convert_base(*base);
        tcrg::gimbal_dynamics::math::Vector<double, 4> torque_vector;
        for (int i = 0; i < 4; ++i) {
            torque_vector[i] = torque[i];
        }
        input.base = &base_state;
        input.torque = &torque_vector;
        input.num_steps = 1;
        input.spec.step = step;
        input.spec.refinement = refinement;
        tcrg::gimbal_dynamics::ForwardOutputT<double> output;
        context->handler.run(input, output);
        store_state(output.step_end.back(), next);
        return 0;
    } catch (const std::exception& error) {
        set_error(error.what());
        return 2;
    }
}

int tcrg_forward_simulate(const TcrgContext* context, const TcrgModel* model,
                          const TcrgTerrain* terrain, const TcrgBaseState* base,
                          const TcrgState* initial, const double* torque, size_t num_steps,
                          double step, int refinement, TcrgState* final_state) {
    if (context == nullptr || !valid(model, terrain, base, initial, torque) ||
        final_state == nullptr) {
        set_error("参数为空指针");
        return 1;
    }
    if (num_steps == 0) {
        set_error("num_steps 必须大于 0");
        return 1;
    }
    try {
        tcrg::gimbal_dynamics::ForwardInputT<double> input;
        input.model = convert_model(*model);
        input.terrain = convert_terrain(*terrain);
        input.initial = convert_state(*initial);
        const tcrg::gimbal_dynamics::BaseStateT<double> base_state = convert_base(*base);
        std::vector<tcrg::gimbal_dynamics::BaseStateT<double>> bases(num_steps, base_state);
        std::vector<tcrg::gimbal_dynamics::math::Vector<double, 4>> torques(num_steps);
        for (size_t k = 0; k < num_steps; ++k) {
            for (int i = 0; i < 4; ++i) {
                torques[k][i] = torque[k * 4 + static_cast<size_t>(i)];
            }
        }
        input.base = bases.data();
        input.torque = torques.data();
        input.num_steps = num_steps;
        input.spec.step = step;
        input.spec.refinement = refinement;
        tcrg::gimbal_dynamics::ForwardOutputT<double> output;
        context->handler.run(input, output);
        store_state(output.step_end.back(), final_state);
        return 0;
    } catch (const std::exception& error) {
        set_error(error.what());
        return 2;
    }
}

int tcrg_bias_torque(const TcrgContext* context, const TcrgModel* model,
                     const TcrgTerrain* terrain, const TcrgBaseState* base,
                     const TcrgState* state, double bias[4]) {
    if (context == nullptr || model == nullptr || terrain == nullptr || base == nullptr ||
        state == nullptr || bias == nullptr) {
        set_error("参数为空指针");
        return 1;
    }
    try {
        using namespace tcrg::gimbal_dynamics;
        using namespace tcrg::gimbal_dynamics::detail;
        const ScalarModelParameters<double> scalar_model =
            convert_model<double>(convert_model(*model));
        const ScalarTerrain<double> scalar_terrain =
            convert_terrain<double>(convert_terrain(*terrain));
        const BaseStateT<double> base_state = convert_base(*base);
        const simulation::StateT<double> state_value = convert_state(*state);
        Vector4<double> q;
        for (int i = 0; i < 4; ++i) {
            q[i] = state_value.angle[i];
        }
        const Vector4<double> qdot = Vector4<double>::zero();
        const Chain<double> chain = evaluate_chain(scalar_model, scalar_terrain, base_state, q);
        MassInternals<double> mass_internals;
        evaluate_mass_matrix(scalar_model, chain, &mass_internals);
        const Vector4<double> bias_value =
            evaluate_bias(scalar_model, chain, base_state, qdot,
                          static_cast<BiasInternals<double>*>(nullptr), &mass_internals);
        for (int i = 0; i < 4; ++i) {
            bias[i] = bias_value[i];
        }
        return 0;
    } catch (const std::exception& error) {
        set_error(error.what());
        return 2;
    }
}

int tcrg_inverse_dynamics(const TcrgContext* context, const TcrgModel* model,
                          const TcrgTerrain* terrain, const TcrgBaseState* base,
                          const TcrgState* state, const double qdd[4], double torque[4]) {
    if (context == nullptr || model == nullptr || terrain == nullptr || base == nullptr ||
        state == nullptr || qdd == nullptr || torque == nullptr) {
        set_error("参数为空指针");
        return 1;
    }
    try {
        using namespace tcrg::gimbal_dynamics;
        using namespace tcrg::gimbal_dynamics::detail;
        const ScalarModelParameters<double> scalar_model =
            convert_model<double>(convert_model(*model));
        const ScalarTerrain<double> scalar_terrain =
            convert_terrain<double>(convert_terrain(*terrain));
        const BaseStateT<double> base_state = convert_base(*base);
        const simulation::StateT<double> state_value = convert_state(*state);
        Vector4<double> q;
        Vector4<double> qdot;
        Vector4<double> qdd_value;
        for (int i = 0; i < 4; ++i) {
            q[i] = state_value.angle[i];
            qdot[i] = state_value.velocity[i];
            qdd_value[i] = qdd[i];
        }
        const Chain<double> chain = evaluate_chain(scalar_model, scalar_terrain, base_state, q);
        MassInternals<double> mass_internals;
        const Matrix4<double> mass = evaluate_mass_matrix(scalar_model, chain, &mass_internals);
        const Vector4<double> bias_value = evaluate_bias(
            scalar_model, chain, base_state, qdot,
            static_cast<BiasInternals<double>*>(nullptr), &mass_internals);
        Vector4<double> result = bias_value;
        for (int r = 0; r < 4; ++r) {
            for (int c = 0; c < 4; ++c) {
                result[r] += mass(r, c) * qdd_value[c];
            }
            result[r] += scalar_model.damping[r] * qdot[r];
        }
        for (int i = 0; i < 4; ++i) {
            torque[i] = result[i];
        }
        return 0;
    } catch (const std::exception& error) {
        set_error(error.what());
        return 2;
    }
}

int tcrg_parameter_gradient(const TcrgContext* context, const TcrgModel* model,
                            const TcrgTerrain* terrain, const TcrgBaseState* base,
                            const TcrgState* initial, const double* torque,
                            size_t num_steps, double step, int refinement,
                            const TcrgLossSpec* loss_spec, double* loss,
                            TcrgState* final_state, double gradient[56]) {
    if (context == nullptr || model == nullptr || terrain == nullptr || base == nullptr ||
        initial == nullptr || torque == nullptr || loss_spec == nullptr || loss == nullptr ||
        final_state == nullptr || gradient == nullptr) {
        set_error("参数为空指针");
        return 1;
    }
    if (num_steps == 0) {
        set_error("num_steps 必须大于 0");
        return 1;
    }
    try {
        using namespace tcrg::gimbal_dynamics;
        ParameterSensitivityInputT<double> input;
        input.model = convert_model(*model);
        input.terrain = convert_terrain(*terrain);
        input.initial = convert_state(*initial);
        const BaseStateT<double> base_state = convert_base(*base);
        std::vector<math::Vector<double, 4>> torques(num_steps);
        std::vector<math::Vector<double, 4>> targets_angle(num_steps);
        std::vector<math::Vector<double, 4>> targets_velocity(num_steps);
        for (size_t k = 0; k < num_steps; ++k) {
            for (int i = 0; i < 4; ++i) {
                torques[k][i] = torque[k * 4 + static_cast<size_t>(i)];
            }
        }
        for (int i = 0; i < 4; ++i) {
            input.loss.angle_weights[i] = loss_spec->angle_weights[i];
            input.loss.velocity_weights[i] = loss_spec->velocity_weights[i];
            input.loss.torque_weights[i] = loss_spec->torque_weights[i];
        }
        if (loss_spec->target_angle != nullptr) {
            for (size_t k = 0; k < num_steps; ++k) {
                for (int i = 0; i < 4; ++i) {
                    targets_angle[k][i] = loss_spec->target_angle[k * 4 + static_cast<size_t>(i)];
                }
            }
            input.loss.target_angle = targets_angle.data();
        }
        if (loss_spec->target_velocity != nullptr) {
            for (size_t k = 0; k < num_steps; ++k) {
                for (int i = 0; i < 4; ++i) {
                    targets_velocity[k][i] =
                        loss_spec->target_velocity[k * 4 + static_cast<size_t>(i)];
                }
            }
            input.loss.target_velocity = targets_velocity.data();
        }
        std::vector<BaseStateT<double>> bases(num_steps, base_state);
        input.base = bases.data();
        input.torque = torques.data();
        input.num_steps = num_steps;
        input.spec.step = step;
        input.spec.refinement = refinement;
        ParameterSensitivityOutputT<double> output;
        context->parameter_sensitivity.run(input, output);
        *loss = output.loss;
        store_state(output.final_state, final_state);
        for (int i = 0; i < 56; ++i) {
            gradient[i] = output.gradient[i];
        }
        return 0;
    } catch (const std::exception& error) {
        set_error(error.what());
        return 2;
    }
}

int tcrg_inertial_basis(const TcrgContext* context, const TcrgModel* model,
                        const TcrgTerrain* terrain, const TcrgBaseState* base,
                        const TcrgState* state, const double qdd[4], double basis[44][4]) {
    if (context == nullptr || model == nullptr || terrain == nullptr || base == nullptr ||
        state == nullptr || qdd == nullptr || basis == nullptr) {
        set_error("参数为空指针");
        return 1;
    }
    try {
        using namespace tcrg::gimbal_dynamics;
        using namespace tcrg::gimbal_dynamics::detail;
        const ScalarModelParameters<double> template_model =
            convert_model<double>(convert_model(*model));
        const ScalarTerrain<double> scalar_terrain =
            convert_terrain<double>(convert_terrain(*terrain));
        const BaseStateT<double> base_state = convert_base(*base);
        const simulation::StateT<double> state_value = convert_state(*state);
        Vector4<double> q;
        Vector4<double> qdot;
        Vector4<double> qdd_value;
        for (int i = 0; i < 4; ++i) {
            q[i] = state_value.angle[i];
            qdot[i] = state_value.velocity[i];
            qdd_value[i] = qdd[i];
        }
        const Chain<double> chain = evaluate_chain(template_model, scalar_terrain, base_state, q);
        const auto torque_of = [&](const ScalarModelParameters<double>& candidate) {
            MassInternals<double> internals;
            const Matrix4<double> mass = evaluate_mass_matrix(candidate, chain, &internals);
            const Vector4<double> bias = evaluate_bias(
                candidate, chain, base_state, qdot,
                static_cast<BiasInternals<double>*>(nullptr), &internals);
            Vector4<double> result = bias;
            for (int r = 0; r < 4; ++r) {
                for (int c = 0; c < 4; ++c) {
                    result[r] += mass(r, c) * qdd_value[c];
                }
                result[r] += candidate.damping[r] * qdot[r];
            }
            return result;
        };
        ScalarModelParameters<double> zero;
        zero.joint_offsets = template_model.joint_offsets;
        zero.gravity = template_model.gravity;
        for (int body = 0; body < 4; ++body) {
            ScalarModelParameters<double> mass_model = zero;
            mass_model.bodies[body].mass = 1.0;
            const Vector4<double> mass_torque = torque_of(mass_model);
            for (int joint = 0; joint < 4; ++joint) {
                basis[body * 10][joint] = mass_torque[joint];
            }
            for (int axis = 0; axis < 3; ++axis) {
                ScalarModelParameters<double> shifted = zero;
                shifted.bodies[body].mass = 1.0;
                shifted.bodies[body].com = Vector3<double>::zero();
                shifted.bodies[body].com[axis] = 1.0;
                for (int r = 0; r < 3; ++r) {
                    for (int c = 0; c < 3; ++c) {
                        shifted.bodies[body].inertia(r, c) =
                            shifted.bodies[body].com[r] * shifted.bodies[body].com[c] -
                            (r == c ? 1.0 : 0.0);
                    }
                }
                const Vector4<double> value = torque_of(shifted);
                for (int joint = 0; joint < 4; ++joint) {
                    basis[body * 10 + 1 + axis][joint] =
                        value[joint] - mass_torque[joint];
                }
            }
            const int positions[6][2] = {{0, 0}, {1, 1}, {2, 2}, {0, 1}, {0, 2}, {1, 2}};
            for (int slot = 0; slot < 6; ++slot) {
                ScalarModelParameters<double> inertia_model = zero;
                inertia_model.bodies[body].inertia(positions[slot][0], positions[slot][1]) = 1.0;
                inertia_model.bodies[body].inertia(positions[slot][1], positions[slot][0]) = 1.0;
                const Vector4<double> value = torque_of(inertia_model);
                for (int joint = 0; joint < 4; ++joint) {
                    basis[body * 10 + 4 + slot][joint] = value[joint];
                }
            }
        }
        for (int joint = 0; joint < 4; ++joint) {
            for (int row = 0; row < 4; ++row) {
                basis[40 + joint][row] = (row == joint) ? qdot[joint] : 0.0;
            }
        }
        return 0;
    } catch (const std::exception& error) {
        set_error(error.what());
        return 2;
    }
}

const char* tcrg_last_error(void) { return g_last_error.c_str(); }

}  // extern "C"
