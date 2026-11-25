// 校验：锁定 roll 轴三自由度模型的合并参数、前向动力学与控制量梯度。
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#include "gimbal_dynamics/model_evaluation.hpp"
#include "gimbal_dynamics/roll_locked.hpp"
#include "gimbal_dynamics/roll_locked_internal.hpp"

using namespace tcrg::gimbal_dynamics;

namespace {

using Vector3 = math::Vector<double, 3>;
using Matrix3 = math::Matrix<double, 3, 3>;

ModelParametersT<double> random_model(std::mt19937& rng) {
    std::uniform_real_distribution<double> unit(-1.0, 1.0);
    ModelParametersT<double> model = ModelParametersT<double>::demo();
    for (int body = 0; body < 4; ++body) {
        model.bodies[body].mass *= 1.0 + 0.2 * unit(rng);
        for (int component = 0; component < 3; ++component) {
            model.bodies[body].com[component] += 0.01 * unit(rng);
            model.joint_offsets[body][component] += 0.005 * unit(rng);
        }
    }
    for (int joint = 0; joint < 4; ++joint) {
        model.damping[joint] = 0.005 + 0.02 * (0.5 + 0.5 * unit(rng));
    }
    return model;
}

BaseStateT<double> random_base(std::mt19937& rng) {
    std::uniform_real_distribution<double> unit(-1.0, 1.0);
    BaseStateT<double> base;
    base.yaw = 3.0 * unit(rng);
    base.yaw_rate = 2.0 * unit(rng);
    base.yaw_acceleration = 5.0 * unit(rng);
    for (int component = 0; component < 3; ++component) {
        base.acceleration[component] = 3.0 * unit(rng);
    }
    return base;
}

TerrainT<double> random_terrain(std::mt19937& rng) {
    std::uniform_real_distribution<double> unit(-1.0, 1.0);
    TerrainT<double> terrain;
    terrain.slope_angle = 0.3 * unit(rng);
    terrain.slope_direction = 3.0 * unit(rng);
    return terrain;
}

Vector3 random_vector3(std::mt19937& rng, double scale) {
    std::uniform_real_distribution<double> unit(-1.0, 1.0);
    return Vector3::from(scale * unit(rng), scale * unit(rng), scale * unit(rng));
}

bool check_kinetic(std::mt19937& rng, int trials) {
    double max_error = 0.0;
    for (int trial = 0; trial < trials; ++trial) {
        const ModelParametersT<double> model = random_model(rng);
        std::uniform_real_distribution<double> angle_distribution(-3.141592653589793, 3.141592653589793);
        const double locked_angle = angle_distribution(rng);
        const roll_locked::ModelT merged = roll_locked::merge_model(model, locked_angle);
        const Matrix3 F = merged.fixed_rotations[1];
        const Vector3 g1 = model.bodies[0].com;
        const Vector3 g2 = model.joint_offsets[1] + F * model.bodies[1].com;
        const Vector3 g12 = merged.bodies[0].com;
        const Matrix3 inertia2_world = F * model.bodies[1].inertia * math::transpose(F);
        for (int sample = 0; sample < 20; ++sample) {
            const Vector3 reference_velocity = random_vector3(rng, 2.0);
            const Vector3 omega = random_vector3(rng, 2.0);
            const Vector3 v1 = reference_velocity + math::cross(omega, g1);
            const Vector3 v2 = reference_velocity + math::cross(omega, g2);
            const Vector3 vm = reference_velocity + math::cross(omega, g12);
            const double kinetic_full =
                0.5 * model.bodies[0].mass * math::dot(v1, v1) +
                0.5 * math::dot(omega, model.bodies[0].inertia * omega) +
                0.5 * model.bodies[1].mass * math::dot(v2, v2) +
                0.5 * math::dot(omega, inertia2_world * omega);
            const double kinetic_merged =
                0.5 * merged.bodies[0].mass * math::dot(vm, vm) +
                0.5 * math::dot(omega, merged.bodies[0].inertia * omega);
            max_error = std::max(max_error,
                                 std::fabs(kinetic_full - kinetic_merged) /
                                     std::max(1.0, std::fabs(kinetic_full)));
        }
        const double mass_error =
            std::fabs(merged.bodies[0].mass -
                      (model.bodies[0].mass + model.bodies[1].mass)) /
            std::max(1.0, merged.bodies[0].mass);
        max_error = std::max(max_error, mass_error);
    }
    printf("合并刚体动能与质量的最大相对偏差 = %.3e\n", max_error);
    return max_error < 1e-12;
}

Vector3 reference_acceleration(const ModelParametersT<double>& model, const TerrainT<double>& terrain,
                               const BaseStateT<double>& base, double locked_angle,
                               const Vector3& angle, const Vector3& velocity,
                               const Vector3& torque) {
    const detail::ScalarModelParameters<double> scalar = detail::convert_model<double>(model);
    const detail::ScalarTerrain<double> scalar_terrain = detail::convert_terrain<double>(terrain);
    const math::Vector<double, 4> full_angle =
        math::Vector<double, 4>::from(angle[0], locked_angle, angle[1], angle[2]);
    const math::Vector<double, 4> full_velocity =
        math::Vector<double, 4>::from(velocity[0], 0.0, velocity[1], velocity[2]);
    const detail::Chain<double> chain =
        detail::evaluate_chain(scalar, scalar_terrain, base, full_angle);
    detail::MassInternals<double> mass_internals;
    const math::Matrix<double, 4, 4> full_mass =
        detail::evaluate_mass_matrix(scalar, chain, &mass_internals);
    const math::Vector<double, 4> full_bias =
        detail::evaluate_bias(scalar, chain, base, full_velocity,
                              static_cast<detail::BiasInternals<double>*>(nullptr),
                              &mass_internals);
    const int indices[3] = {0, 2, 3};
    Matrix3 mass;
    Vector3 right_hand_side = Vector3::zero();
    for (int row = 0; row < 3; ++row) {
        for (int column = 0; column < 3; ++column) {
            mass(row, column) = full_mass(indices[row], indices[column]);
        }
        const int joint = indices[row];
        right_hand_side[row] =
            torque[row] - full_bias[joint] - scalar.damping[joint] * full_velocity[joint];
    }
    return roll_locked::detail::solve3(mass, right_hand_side);
}

bool check_forward(std::mt19937& rng, int trials) {
    double max_error = 0.0;
    for (int trial = 0; trial < trials; ++trial) {
        const ModelParametersT<double> model = random_model(rng);
        const TerrainT<double> terrain = random_terrain(rng);
        const BaseStateT<double> base = random_base(rng);
        std::uniform_real_distribution<double> angle_distribution(-3.141592653589793, 3.141592653589793);
        const double locked_angle = angle_distribution(rng);
        const Vector3 angle = random_vector3(rng, 1.5);
        const Vector3 velocity = random_vector3(rng, 2.0);
        const Vector3 torque = random_vector3(rng, 2.0);
        const roll_locked::ModelT merged = roll_locked::merge_model(model, locked_angle);
        const Vector3 actual =
            roll_locked::forward_dynamics(merged, terrain, base, angle, velocity, torque);
        const Vector3 reference =
            reference_acceleration(model, terrain, base, locked_angle, angle, velocity, torque);
        const double scale = std::max(1.0, std::max(std::fabs(reference[0]),
                                                    std::max(std::fabs(reference[1]),
                                                             std::fabs(reference[2]))));
        for (int component = 0; component < 3; ++component) {
            max_error = std::max(max_error, std::fabs(actual[component] - reference[component]) / scale);
        }
    }
    printf("前向动力学与四自由度约束参考的最大相对偏差 = %.3e\n", max_error);
    return max_error < 1e-11;
}


struct ReferenceStage {
    math::Vector<double, 4> acceleration;
    double roll_torque = 0.0;
    double acceleration_error = 0.0;
    double roll_acceleration = 0.0;
};

BaseStateT<double> advance_base_local(const BaseStateT<double>& base, double time) {
    BaseStateT<double> result = base;
    result.yaw = base.yaw + base.yaw_rate * time + 0.5 * base.yaw_acceleration * time * time;
    result.yaw_rate = base.yaw_rate + base.yaw_acceleration * time;
    return result;
}

// 在给定四自由度阶段状态上，用完整四自由度质量矩阵求锁定 roll 所需的准确力矩，
// 并检查把这个力矩和剩余三个力矩一起送入完整模型时，roll 加速度确实为零。
ReferenceStage reference_stage(const ModelParametersT<double>& model,
                               const TerrainT<double>& terrain,
                               const BaseStateT<double>& base,
                               const math::Vector<double, 4>& angle,
                               const math::Vector<double, 4>& velocity,
                               const Vector3& torque) {
    const detail::ScalarModelParameters<double> scalar = detail::convert_model<double>(model);
    const detail::ScalarTerrain<double> scalar_terrain = detail::convert_terrain<double>(terrain);
    const detail::TerrainFrame<double> frame = detail::evaluate_terrain_frame(scalar_terrain);
    const detail::Chain<double> chain = detail::evaluate_chain(scalar, frame, base, angle);
    detail::MassInternals<double> mass_internals;
    const math::Matrix<double, 4, 4> mass =
        detail::evaluate_mass_matrix(scalar, chain, &mass_internals);
    const math::Vector<double, 4> bias =
        detail::evaluate_bias(scalar, chain, base, velocity,
                              static_cast<detail::BiasInternals<double>*>(nullptr),
                              &mass_internals);
    const int indices[3] = {0, 2, 3};
    Matrix3 mass_sub;
    Vector3 right_hand_side = Vector3::zero();
    for (int row = 0; row < 3; ++row) {
        for (int column = 0; column < 3; ++column) {
            mass_sub(row, column) = mass(indices[row], indices[column]);
        }
        const int joint = indices[row];
        right_hand_side[row] =
            torque[row] - bias[joint] - scalar.damping[joint] * velocity[joint];
    }
    const Vector3 acceleration_sub = roll_locked::detail::solve3(mass_sub, right_hand_side);
    math::Vector<double, 4> acceleration = math::Vector<double, 4>::zero();
    acceleration[0] = acceleration_sub[0];
    acceleration[1] = 0.0;
    acceleration[2] = acceleration_sub[1];
    acceleration[3] = acceleration_sub[2];

    double roll_torque = 0.0;
    for (int joint = 0; joint < 4; ++joint) {
        roll_torque += mass(1, joint) * acceleration[joint];
    }
    roll_torque += scalar.damping[1] * velocity[1] + bias[1];

    math::Vector<double, 4> full_torque = math::Vector<double, 4>::zero();
    full_torque[0] = torque[0];
    full_torque[1] = roll_torque;
    full_torque[2] = torque[1];
    full_torque[3] = torque[2];
    const math::Vector<double, 4> checked =
        detail::evaluate_acceleration(scalar, frame, base, angle, velocity, full_torque);
    double acceleration_error = 0.0;
    for (int joint = 0; joint < 4; ++joint) {
        acceleration_error =
            std::max(acceleration_error, std::fabs(checked[joint] - acceleration[joint]));
    }
    ReferenceStage result;
    result.acceleration = acceleration;
    result.roll_torque = roll_torque;
    result.acceleration_error = acceleration_error;
    result.roll_acceleration = std::fabs(checked[1]);
    return result;
}

bool check_lock_torque_equivalence(std::mt19937& rng, int trials) {
    double max_state_error = 0.0;
    double max_stage_error = 0.0;
    double max_roll_acceleration = 0.0;
    double max_roll_torque = 0.0;
    for (int trial = 0; trial < trials; ++trial) {
        const ModelParametersT<double> model = random_model(rng);
        const TerrainT<double> terrain = random_terrain(rng);
        std::uniform_real_distribution<double> angle_distribution(-3.141592653589793,
                                                                  3.141592653589793);
        const double locked_angle = angle_distribution(rng);
        const std::size_t steps = 4;
        const double dt = 0.01;
        const int refinement = 2;
        const double h = dt / static_cast<double>(refinement);
        std::vector<BaseStateT<double>> bases(steps);
        std::vector<Vector3> torques(steps);
        for (std::size_t step = 0; step < steps; ++step) {
            bases[step] = random_base(rng);
            torques[step] = random_vector3(rng, 0.8);
        }
        const Vector3 initial_angle = random_vector3(rng, 0.6);
        const Vector3 initial_velocity = random_vector3(rng, 0.8);

        roll_locked::ForwardInputT input;
        input.model = model;
        input.locked_angle = locked_angle;
        input.terrain = terrain;
        input.initial.angle = initial_angle;
        input.initial.velocity = initial_velocity;
        input.base = bases.data();
        input.torque = torques.data();
        input.num_steps = steps;
        input.spec.step = dt;
        input.spec.refinement = refinement;
        roll_locked::ForwardOutputT output;
        roll_locked::simulate(input, output);

        math::Vector<double, 4> full_angle =
            math::Vector<double, 4>::from(initial_angle[0], locked_angle, initial_angle[1],
                                          initial_angle[2]);
        math::Vector<double, 4> full_velocity =
            math::Vector<double, 4>::from(initial_velocity[0], 0.0, initial_velocity[1],
                                          initial_velocity[2]);
        for (std::size_t step = 0; step < steps; ++step) {
            const Vector3 torque = torques[step];
            for (int substep = 0; substep < refinement; ++substep) {
                const double substep_time = static_cast<double>(substep) * h;
                const BaseStateT<double> stage_base[4] = {
                    advance_base_local(bases[step], substep_time),
                    advance_base_local(bases[step], substep_time + 0.5 * h),
                    advance_base_local(bases[step], substep_time + 0.5 * h),
                    advance_base_local(bases[step], substep_time + h),
                };
                math::Vector<double, 4> stage_angle[4];
                math::Vector<double, 4> stage_velocity[4];
                math::Vector<double, 4> angle_derivative[4];
                math::Vector<double, 4> velocity_derivative[4];
                stage_angle[0] = full_angle;
                stage_velocity[0] = full_velocity;
                for (int stage = 0; stage < 4; ++stage) {
                    if (stage > 0) {
                        const double advance = (stage == 3) ? h : 0.5 * h;
                        stage_angle[stage] = stage_angle[0] + angle_derivative[stage - 1] * advance;
                        stage_velocity[stage] =
                            stage_velocity[0] + velocity_derivative[stage - 1] * advance;
                    }
                    const ReferenceStage reference =
                        reference_stage(model, terrain, stage_base[stage],
                                        stage_angle[stage], stage_velocity[stage], torque);
                    angle_derivative[stage] = stage_velocity[stage];
                    velocity_derivative[stage] = reference.acceleration;
                    max_stage_error = std::max(max_stage_error, reference.acceleration_error);
                    max_roll_acceleration =
                        std::max(max_roll_acceleration, reference.roll_acceleration);
                    max_roll_torque = std::max(max_roll_torque, std::fabs(reference.roll_torque));
                }
                full_angle =
                    stage_angle[0] +
                    (angle_derivative[0] + angle_derivative[1] * 2.0 +
                     angle_derivative[2] * 2.0 + angle_derivative[3]) *
                        (h / 6.0);
                full_velocity =
                    stage_velocity[0] +
                    (velocity_derivative[0] + velocity_derivative[1] * 2.0 +
                     velocity_derivative[2] * 2.0 + velocity_derivative[3]) *
                        (h / 6.0);
            }
            const int indices[3] = {0, 2, 3};
            for (int joint = 0; joint < 3; ++joint) {
                max_state_error =
                    std::max(max_state_error,
                             std::fabs(full_angle[indices[joint]] -
                                       output.step_end[step].angle[joint]));
                max_state_error =
                    std::max(max_state_error,
                             std::fabs(full_velocity[indices[joint]] -
                                       output.step_end[step].velocity[joint]));
            }
        }
    }
    printf("准确锁定力矩下四自由度轨迹与三自由度轨迹最大偏差 = %.3e\n", max_state_error);
    printf("准确锁定力矩阶段的加速度复核最大偏差 = %.3e，roll 加速度最大绝对值 = %.3e\n",
           max_stage_error, max_roll_acceleration);
    printf("锁定所需 roll 力矩最大绝对值 = %.3e\n", max_roll_torque);
    return max_state_error < 1e-11 && max_stage_error < 1e-11 &&
           max_roll_acceleration < 1e-11;
}

struct Problem {
    roll_locked::ForwardInputT input;
    std::vector<BaseStateT<double>> base;
    std::vector<Vector3> torque;
    std::vector<Vector3> target_angle;
    std::vector<Vector3> target_velocity;
    std::size_t num_steps = 5;
    roll_locked::ForwardOutputT forward;
};

Problem make_problem(std::mt19937& rng) {
    Problem problem;
    problem.num_steps = 5;
    problem.input.model = random_model(rng);
    problem.input.locked_angle = 0.7 * (0.5 - std::uniform_real_distribution<double>(0.0, 1.0)(rng));
    problem.input.terrain = random_terrain(rng);
    problem.input.initial.angle = random_vector3(rng, 0.2);
    problem.input.initial.velocity = random_vector3(rng, 0.3);
    problem.base.resize(problem.num_steps);
    problem.torque.resize(problem.num_steps);
    problem.target_angle.resize(problem.num_steps);
    problem.target_velocity.resize(problem.num_steps);
    for (std::size_t step = 0; step < problem.num_steps; ++step) {
        problem.base[step] = random_base(rng);
        problem.torque[step] = random_vector3(rng, 0.5);
        problem.target_angle[step] = random_vector3(rng, 0.2);
        problem.target_velocity[step] = random_vector3(rng, 0.3);
    }
    problem.input.base = problem.base.data();
    problem.input.torque = problem.torque.data();
    problem.input.num_steps = problem.num_steps;
    problem.input.spec.step = 0.01;
    problem.input.spec.refinement = 2;
    problem.input.loss.angle_weights = Vector3::from(1.0, 2.0, 0.8);
    problem.input.loss.velocity_weights = Vector3::from(0.3, 0.5, 0.2);
    problem.input.loss.torque_weights = Vector3::from(0.05, 0.04, 0.03);
    problem.input.loss.torque_rate_weights = Vector3::from(0.7, 0.4, 0.6);
    problem.input.loss.limit_weights = Vector3::from(2.0, 2.5, 1.5);
    problem.input.loss.limit_lower = Vector3::from(-0.10, -0.08, -0.20);
    problem.input.loss.limit_upper = Vector3::from(0.10, 0.08, 0.20);
    problem.input.loss.limit_margin = Vector3::from(0.01, 0.01, 0.01);
    problem.input.loss.target_angle = problem.target_angle.data();
    problem.input.loss.target_velocity = problem.target_velocity.data();
    static Vector3 previous_torque = Vector3::from(0.10, -0.08, 0.02);
    problem.input.loss.previous_torque = &previous_torque;
    return problem;
}

bool check_simulation_and_gradient(std::mt19937& rng, int trials) {
    double max_loss_error = 0.0;
    double max_gradient_error = 0.0;
    for (int trial = 0; trial < trials; ++trial) {
        Problem problem = make_problem(rng);
        roll_locked::simulate(problem.input, problem.forward);
        roll_locked::GradientOutputT gradient_output;
        roll_locked::simulate_and_gradient(problem.input, gradient_output);
        const double loss_scale = std::max(1.0, std::fabs(problem.forward.loss));
        max_loss_error = std::max(max_loss_error,
                                  std::fabs(problem.forward.loss - gradient_output.loss) / loss_scale);
        for (int component = 0; component < 3; ++component) {
            max_loss_error = std::max(
                max_loss_error,
                std::fabs(problem.forward.step_end.back().angle[component] -
                          gradient_output.final_state.angle[component]) /
                    std::max(1.0, std::fabs(problem.forward.step_end.back().angle[component])));
        }

        const double epsilon = 1e-6;
        for (std::size_t step = 0; step < problem.num_steps; ++step) {
            for (int joint = 0; joint < 3; ++joint) {
                Problem plus = problem;
                Problem minus = problem;
                plus.input.base = plus.base.data();
                plus.input.torque = plus.torque.data();
                plus.input.loss.target_angle = plus.target_angle.data();
                plus.input.loss.target_velocity = plus.target_velocity.data();
                minus.input.base = minus.base.data();
                minus.input.torque = minus.torque.data();
                minus.input.loss.target_angle = minus.target_angle.data();
                minus.input.loss.target_velocity = minus.target_velocity.data();
                plus.torque[step][joint] += epsilon;
                minus.torque[step][joint] -= epsilon;
                roll_locked::ForwardOutputT plus_output;
                roll_locked::ForwardOutputT minus_output;
                roll_locked::simulate(plus.input, plus_output);
                roll_locked::simulate(minus.input, minus_output);
                const double difference =
                    (plus_output.loss - minus_output.loss) / (2.0 * epsilon);
                const double analytic = gradient_output.gradient[step][joint];
                max_gradient_error =
                    std::max(max_gradient_error,
                             std::fabs(difference - analytic) / std::max(1.0, std::fabs(analytic)));
            }
        }
    }
    printf("前向与梯度模式损失/末状态最大相对偏差 = %.3e\n", max_loss_error);
    printf("控制量梯度与中心差分最大相对偏差 = %.3e\n", max_gradient_error);
    return max_loss_error < 1e-12 && max_gradient_error < 2e-6;
}

bool check_boundaries(std::mt19937& rng) {
    const ModelParametersT<double> model = random_model(rng);
    const TerrainT<double> terrain = random_terrain(rng);
    const BaseStateT<double> base = random_base(rng);
    const Vector3 angle = random_vector3(rng, 1.0);
    const Vector3 velocity = random_vector3(rng, 1.0);
    const Vector3 torque = random_vector3(rng, 1.0);
    const double angles[4] = {0.0, 3.141592653589793 / 2.0, -3.141592653589793 / 2.0,
                              3.141592653589793};
    for (double locked_angle : angles) {
        const roll_locked::ModelT merged = roll_locked::merge_model(model, locked_angle);
        const Vector3 acceleration =
            roll_locked::forward_dynamics(merged, terrain, base, angle, velocity, torque);
        for (int component = 0; component < 3; ++component) {
            if (!std::isfinite(acceleration[component])) {
                printf("锁定角 %.6f 下的加速度不是有限值\n", locked_angle);
                return false;
            }
        }
    }
    return true;
}

bool check_invalid(std::mt19937& rng) {
    const ModelParametersT<double> model = random_model(rng);
    bool nan_threw = false;
    try {
        roll_locked::merge_model(model, std::nan(""));
    } catch (const std::invalid_argument&) {
        nan_threw = true;
    }
    ModelParametersT<double> invalid = model;
    invalid.bodies[0].inertia(0, 0) = -1.0;
    bool inertia_threw = false;
    try {
        roll_locked::merge_model(invalid, 0.3);
    } catch (const std::invalid_argument&) {
        inertia_threw = true;
    }
    return nan_threw && inertia_threw;
}

}  // namespace

int main() {
    std::mt19937 rng(20261125);
    bool ok = true;
    ok = check_kinetic(rng, 20) && ok;
    ok = check_forward(rng, 200) && ok;
    ok = check_lock_torque_equivalence(rng, 20) && ok;
    ok = check_simulation_and_gradient(rng, 20) && ok;
    ok = check_boundaries(rng) && ok;
    ok = check_invalid(rng) && ok;
    if (!ok) {
        printf("校验失败\n");
        return 1;
    }
    printf("校验通过\n");
    return 0;
}
