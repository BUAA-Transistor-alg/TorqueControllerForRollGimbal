#include "gimbal_dynamics/roll_locked_internal.hpp"

#include <cmath>
#include <stdexcept>

#include "gimbal_dynamics/model_evaluation.hpp"

namespace tcrg::gimbal_dynamics {
namespace roll_locked {
namespace detail {

Vector3 axis_local(int joint) {
    switch (joint) {
        case 0:
        case 1:
            return Vector3::unit(2);
        case 2:
            return Vector3::unit(0);
        default:
            throw std::out_of_range("关节编号超出范围");
    }
}

Matrix3 local_rotation(int joint, Real angle) {
    switch (joint) {
        case 0:
        case 1:
            return tcrg::gimbal_dynamics::detail::rotation_z<Real>(angle);
        case 2:
            return tcrg::gimbal_dynamics::detail::rotation_x<Real>(angle);
        default:
            throw std::out_of_range("关节编号超出范围");
    }
}

namespace {

Matrix3 outer_product(const Vector3& value) {
    Matrix3 result = Matrix3::zero();
    for (int row = 0; row < 3; ++row) {
        for (int column = 0; column < 3; ++column) {
            result(row, column) = value[row] * value[column];
        }
    }
    return result;
}

// 合并刚体的惯量平行轴项：m (|d|^2 I - d d^T)
Matrix3 parallel_axis(Real mass, const Vector3& d) {
    return (Matrix3::identity() * math::dot(d, d) - outer_product(d)) * mass;
}

void validate_merged(const ModelT& model) {
    const Real tolerance = std::is_same<Real, float>::value ? Real(1e-5) : Real(1e-9);
    const auto finite = [](Real value) { return std::isfinite(static_cast<double>(value)); };
    for (int body = 0; body < 3; ++body) {
        if (!finite(model.bodies[body].mass) || model.bodies[body].mass < Real(0)) {
            throw std::invalid_argument("合并刚体质量必须是非负有限值");
        }
        for (int component = 0; component < 3; ++component) {
            if (!finite(model.bodies[body].com[component]) ||
                !finite(model.joint_offsets[body][component])) {
                throw std::invalid_argument("合并刚体的质心或关节偏置包含非有限值");
            }
        }
        const Matrix3& inertia = model.bodies[body].inertia;
        for (int row = 0; row < 3; ++row) {
            for (int column = 0; column < 3; ++column) {
                if (!finite(inertia(row, column))) {
                    throw std::invalid_argument("合并后的惯量矩阵包含非有限值");
                }
                if (std::fabs(static_cast<double>(inertia(row, column) - inertia(column, row))) >
                    static_cast<double>(tolerance) *
                        (1.0 + std::fabs(static_cast<double>(inertia(row, column))))) {
                    throw std::invalid_argument("合并后的惯量矩阵必须对称");
                }
            }
        }
        const Real a = inertia(0, 0);
        const Real b = inertia(0, 1);
        const Real c = inertia(1, 1);
        const Real second_minor = a * c - b * b;
        const Real determinant =
            a * (inertia(1, 1) * inertia(2, 2) - inertia(1, 2) * inertia(2, 1)) -
            b * (inertia(1, 0) * inertia(2, 2) - inertia(1, 2) * inertia(2, 0)) +
            inertia(0, 2) *
                (inertia(1, 0) * inertia(2, 1) - inertia(1, 1) * inertia(2, 0));
        if (!(a > Real(0)) || !(second_minor > Real(0)) || !(determinant > Real(0))) {
            throw std::invalid_argument("合并后的惯量矩阵必须正定");
        }
    }
    for (int joint = 0; joint < 3; ++joint) {
        if (!finite(model.damping[joint]) || model.damping[joint] < Real(0)) {
            throw std::invalid_argument("合并后的粘性阻尼必须是非负有限值");
        }
    }
    if (!finite(model.gravity) || !finite(model.locked_angle)) {
        throw std::invalid_argument("重力加速度或锁定角度包含非有限值");
    }
}

}  // namespace

ChainT evaluate_chain(const ModelT& model, const Terrain& terrain, const BaseState& base,
                      const Vector3& angle) {
    ChainT chain;
    const Matrix3 terrain_rotation =
        tcrg::gimbal_dynamics::detail::rotation_z<Real>(terrain.slope_direction) *
        tcrg::gimbal_dynamics::detail::rotation_y<Real>(terrain.slope_angle) *
        tcrg::gimbal_dynamics::detail::rotation_z<Real>(-terrain.slope_direction);
    chain.normal = terrain_rotation * Vector3::unit(2);
    const Matrix3 base_rotation =
        terrain_rotation * tcrg::gimbal_dynamics::detail::rotation_z<Real>(base.yaw);
    Matrix3 parent_rotation = base_rotation;
    Vector3 parent_origin = Vector3::zero();
    for (int joint = 0; joint < 3; ++joint) {
        const Matrix3 fixed = model.fixed_rotations[joint];
        const Matrix3 rotation =
            parent_rotation * fixed * local_rotation(joint, angle[joint]);
        const Vector3 offset = parent_rotation * model.joint_offsets[joint];
        const Vector3 origin = parent_origin + offset;
        chain.offset[joint] = offset;
        chain.joint_origin[joint] = origin;
        chain.rotation[joint] = rotation;
        chain.axis[joint] = parent_rotation * fixed * axis_local(joint);
        chain.com[joint] = origin + rotation * model.bodies[joint].com;
        parent_rotation = rotation;
        parent_origin = origin;
    }
    return chain;
}

void evaluate_mass_matrix(const ModelT& model, const ChainT& chain, Matrix3& mass,
                          std::array<Matrix3, 3>* velocity_jacobian,
                          std::array<Matrix3, 3>* angular_jacobian,
                          std::array<Matrix3, 3>* inertia_world) {
    mass = Matrix3::zero();
    for (int body = 0; body < 3; ++body) {
        Matrix3 Jv = Matrix3::zero();
        Matrix3 Jw = Matrix3::zero();
        for (int joint = 0; joint <= body; ++joint) {
            const Vector3 arm = chain.com[body] - chain.joint_origin[joint];
            const Vector3 column = math::cross(chain.axis[joint], arm);
            for (int row = 0; row < 3; ++row) {
                Jv(row, joint) = column[row];
                Jw(row, joint) = chain.axis[joint][row];
            }
        }
        const Matrix3 Iworld =
            chain.rotation[body] * model.bodies[body].inertia * math::transpose(chain.rotation[body]);
        mass += math::transpose(Jv) * Jv * model.bodies[body].mass;
        mass += math::transpose(Jw) * Iworld * Jw;
        if (velocity_jacobian != nullptr) {
            (*velocity_jacobian)[body] = Jv;
        }
        if (angular_jacobian != nullptr) {
            (*angular_jacobian)[body] = Jw;
        }
        if (inertia_world != nullptr) {
            (*inertia_world)[body] = Iworld;
        }
    }
}

Vector3 evaluate_bias(const ModelT& model, const ChainT& chain, const BaseState& base,
                      const Vector3& velocity, BiasInternals* internals) {
    const Vector3 gravity_vector = Vector3::from(Real(0), Real(0), -model.gravity);
    const Vector3 base_omega = chain.normal * base.yaw_rate;
    const Vector3 base_alpha = chain.normal * base.yaw_acceleration;
    const Vector3 base_acceleration = base.acceleration;

    Vector3 previous_omega = base_omega;
    Vector3 previous_alpha = base_alpha;
    Vector3 previous_origin_acceleration = base_acceleration;
    std::array<Vector3, 3> omega{};
    std::array<Vector3, 3> alpha{};
    std::array<Vector3, 3> com_acceleration{};
    for (int body = 0; body < 3; ++body) {
        const Vector3& offset = chain.offset[body];
        const Vector3 origin_acceleration =
            previous_origin_acceleration + math::cross(previous_alpha, offset) +
            math::cross(previous_omega, math::cross(previous_omega, offset));
        const Vector3& axis = chain.axis[body];
        const Vector3 angular_velocity = previous_omega + axis * velocity[body];
        const Vector3 angular_acceleration =
            previous_alpha + math::cross(previous_omega, axis * velocity[body]);
        const Vector3 com_relative = chain.com[body] - chain.joint_origin[body];
        const Vector3 linear_acceleration =
            origin_acceleration + math::cross(angular_acceleration, com_relative) +
            math::cross(angular_velocity, math::cross(angular_velocity, com_relative));
        omega[body] = angular_velocity;
        alpha[body] = angular_acceleration;
        com_acceleration[body] = linear_acceleration;
        previous_omega = angular_velocity;
        previous_alpha = angular_acceleration;
        previous_origin_acceleration = origin_acceleration;
    }

    std::array<Vector3, 3> force{};
    std::array<Vector3, 3> couple{};
    for (int body = 0; body < 3; ++body) {
        const Matrix3 Iworld = chain.rotation[body] * model.bodies[body].inertia *
                               math::transpose(chain.rotation[body]);
        force[body] = (com_acceleration[body] - gravity_vector) * model.bodies[body].mass;
        couple[body] = Iworld * alpha[body] +
                       math::cross(omega[body], Iworld * omega[body]);
    }

    Vector3 bias = Vector3::zero();
    std::array<Vector3, 3> joint_force{};
    std::array<Vector3, 3> joint_moment{};
    Vector3 child_force = Vector3::zero();
    Vector3 child_moment = Vector3::zero();
    for (int body = 2; body >= 0; --body) {
        Vector3 moment = couple[body] + math::cross(chain.com[body] - chain.joint_origin[body],
                                                    force[body]);
        if (body < 2) {
            moment += child_moment +
                      math::cross(chain.joint_origin[body + 1] - chain.joint_origin[body],
                                  child_force);
        }
        bias[body] = math::dot(chain.axis[body], moment);
        joint_force[body] = force[body] + child_force;
        joint_moment[body] = moment;
        child_force = joint_force[body];
        child_moment = moment;
    }
    if (internals != nullptr) {
        internals->omega = omega;
        internals->alpha = alpha;
        internals->com_acceleration = com_acceleration;
        internals->force = force;
        internals->couple = couple;
        internals->joint_force = joint_force;
        internals->joint_moment = joint_moment;
    }
    return bias;
}

void Cholesky3::factorize(const Matrix3& mass) {
    valid = true;
    for (int row = 0; row < 3; ++row) {
        for (int column = 0; column <= row; ++column) {
            Real sum = mass(row, column);
            for (int k = 0; k < column; ++k) {
                sum -= lower[row][k] * lower[column][k];
            }
            if (row == column) {
                if (!(sum > Real(0))) {
                    valid = false;
                    return;
                }
                lower[row][column] = std::sqrt(sum);
            } else {
                lower[row][column] = sum / lower[column][column];
            }
        }
    }
}

Vector3 Cholesky3::solve(const Vector3& rhs) const {
    Real intermediate[3];
    for (int row = 0; row < 3; ++row) {
        Real sum = rhs[row];
        for (int k = 0; k < row; ++k) {
            sum -= lower[row][k] * intermediate[k];
        }
        intermediate[row] = sum / lower[row][row];
    }
    Vector3 result;
    for (int row = 2; row >= 0; --row) {
        Real sum = intermediate[row];
        for (int k = row + 1; k < 3; ++k) {
            sum -= lower[k][row] * result[k];
        }
        result[row] = sum / lower[row][row];
    }
    return result;
}

Vector3 solve3(const Matrix3& mass, const Vector3& rhs) {
    Cholesky3 factor;
    factor.factorize(mass);
    if (!factor.valid) {
        throw std::runtime_error("合并后的质量矩阵不是正定矩阵");
    }
    return factor.solve(rhs);
}

void evaluate_stage_data(const ModelT& model, const Terrain& terrain, const BaseState& base,
                         const StateT& state, const Vector3& torque, StageData& out) {
    out.base = base;
    out.angle = state.angle;
    out.velocity = state.velocity;
    out.chain = evaluate_chain(model, terrain, base, state.angle);
    evaluate_mass_matrix(model, out.chain, out.mass, &out.velocity_jacobian,
                         &out.angular_jacobian, &out.inertia_world);
    out.bias = evaluate_bias(model, out.chain, base, state.velocity, &out.bias_internals);
    const Vector3 right_hand_side =
        torque - math::elementwise(model.damping, state.velocity) - out.bias;
    out.factor.factorize(out.mass);
    if (!out.factor.valid) {
        throw std::runtime_error("合并后的质量矩阵不是正定矩阵");
    }
    out.acceleration = out.factor.solve(right_hand_side);
}

}  // namespace detail

ModelT merge_model(const ModelParameters& original, Real locked_angle) {
    if (!std::isfinite(static_cast<double>(locked_angle))) {
        throw std::invalid_argument("锁定角度必须是有限值");
    }
    const tcrg::gimbal_dynamics::detail::ScalarModelParameters<Real> scalar =
        tcrg::gimbal_dynamics::detail::convert_model<Real>(original);
    const Matrix3 F = tcrg::gimbal_dynamics::detail::rotation_y<Real>(locked_angle);

    const Real mass1 = scalar.bodies[0].mass;
    const Real mass2 = scalar.bodies[1].mass;
    const Real mass12 = mass1 + mass2;
    const Vector3& com1 = scalar.bodies[0].com;
    const Vector3& com2 = scalar.bodies[1].com;
    const Vector3& offset2 = scalar.joint_offsets[1];
    const Matrix3& inertia1 = scalar.bodies[0].inertia;
    const Matrix3& inertia2 = scalar.bodies[1].inertia;

    const Vector3 com12 =
        (mass12 > Real(0)) ? (com1 * mass1 + (offset2 + F * com2) * mass2) / mass12
                           : Vector3::zero();
    const Vector3 d1 = com1 - com12;
    const Vector3 d2 = offset2 + F * com2 - com12;
    const Matrix3 inertia12 = inertia1 + detail::parallel_axis(mass1, d1) +
                              F * inertia2 * math::transpose(F) +
                              detail::parallel_axis(mass2, d2);

    ModelT model;
    model.bodies[0].mass = mass12;
    model.bodies[0].com = com12;
    model.bodies[0].inertia = inertia12;
    model.bodies[1].mass = scalar.bodies[2].mass;
    model.bodies[1].com = scalar.bodies[2].com;
    model.bodies[1].inertia = scalar.bodies[2].inertia;
    model.bodies[2].mass = scalar.bodies[3].mass;
    model.bodies[2].com = scalar.bodies[3].com;
    model.bodies[2].inertia = scalar.bodies[3].inertia;
    model.joint_offsets[0] = scalar.joint_offsets[0];
    model.joint_offsets[1] = offset2 + F * scalar.joint_offsets[2];
    model.joint_offsets[2] = scalar.joint_offsets[3];
    model.damping = Vector3::from(scalar.damping[0], scalar.damping[2], scalar.damping[3]);
    model.gravity = scalar.gravity;
    model.fixed_rotations[0] = Matrix3::identity();
    model.fixed_rotations[1] = F;
    model.fixed_rotations[2] = Matrix3::identity();
    model.locked_angle = locked_angle;
    detail::validate_merged(model);
    return model;
}

Vector3 forward_dynamics(const ModelT& model, const Terrain& terrain, const BaseState& base,
                         const Vector3& angle, const Vector3& velocity, const Vector3& torque) {
    detail::StageData data;
    StateT state;
    state.angle = angle;
    state.velocity = velocity;
    detail::evaluate_stage_data(model, terrain, base, state, torque, data);
    return data.acceleration;
}

}  // namespace roll_locked
}  // namespace tcrg::gimbal_dynamics
