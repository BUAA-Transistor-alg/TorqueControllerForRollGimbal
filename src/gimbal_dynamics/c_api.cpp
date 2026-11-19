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
    explicit TcrgContext(int threads) : handler(threads) {}
    tcrg::gimbal_dynamics::ForwardSimulation<double> handler;
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

const char* tcrg_last_error(void) { return g_last_error.c_str(); }

}  // extern "C"
