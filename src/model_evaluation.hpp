#pragma once

// 内部：标量泛化的动力学求值。
//
// 所有物理量都写成模板标量 Scalar 的形式，前向仿真与闭式单步实现共用同一套
// 求值代码。梯度路径对本文件的求值过程做解析方向导数（见 stage_derivatives.cpp），
// 闭式单步实现的质量矩阵偏导同样用链的方向导数求值（见 dynamics_model.cpp）。
//
// 质量矩阵按 DYNAMICS.md 7.2 的雅可比形式组装；偏置力 b 按第 8 节的逆动力学
// 递推形式计算（令 q̈ = 0）。递推形式只含四则运算与三角函数，对 q、q̇ 以及全部
// 模型参数一阶可微，因此不需要质量矩阵对 q 的二阶偏导。

#include <array>
#include <cmath>
#include <stdexcept>
#include <type_traits>

#include "gimbal_dynamics/dynamics_model.hpp"
#include "gimbal_dynamics/math_types.hpp"

namespace gimbal {
namespace detail {

template <typename Scalar>
using Vector3 = math::Vector<Scalar, 3>;
template <typename Scalar>
using Matrix3 = math::Matrix<Scalar, 3, 3>;
template <typename Scalar>
using Vector4 = math::Vector<Scalar, 4>;
template <typename Scalar>
using Matrix4 = math::Matrix<Scalar, 4, 4>;
template <typename Scalar>
using Matrix34 = math::Matrix<Scalar, 3, 4>;
template <typename Scalar>
using StateVector = math::Vector<Scalar, 8>;

template <typename Scalar>
Vector3<Scalar> cross_product(const Vector3<Scalar>& lhs, const Vector3<Scalar>& rhs) {
    return math::cross(lhs, rhs);
}

template <typename Scalar>
Scalar dot_product(const Vector3<Scalar>& lhs, const Vector3<Scalar>& rhs) {
    return math::dot(lhs, rhs);
}

template <typename Scalar>
Matrix3<Scalar> rotation_x(const Scalar& angle) {
    const Scalar cosine = std::cos(angle);
    const Scalar sine = std::sin(angle);
    Matrix3<Scalar> matrix;
    matrix.data[0][0] = Scalar(1);
    matrix.data[0][1] = Scalar(0);
    matrix.data[0][2] = Scalar(0);
    matrix.data[1][0] = Scalar(0);
    matrix.data[1][1] = cosine;
    matrix.data[1][2] = -sine;
    matrix.data[2][0] = Scalar(0);
    matrix.data[2][1] = sine;
    matrix.data[2][2] = cosine;
    return matrix;
}

template <typename Scalar>
Matrix3<Scalar> rotation_y(const Scalar& angle) {
    const Scalar cosine = std::cos(angle);
    const Scalar sine = std::sin(angle);
    Matrix3<Scalar> matrix;
    matrix.data[0][0] = cosine;
    matrix.data[0][1] = Scalar(0);
    matrix.data[0][2] = sine;
    matrix.data[1][0] = Scalar(0);
    matrix.data[1][1] = Scalar(1);
    matrix.data[1][2] = Scalar(0);
    matrix.data[2][0] = -sine;
    matrix.data[2][1] = Scalar(0);
    matrix.data[2][2] = cosine;
    return matrix;
}

template <typename Scalar>
Matrix3<Scalar> rotation_z(const Scalar& angle) {
    const Scalar cosine = std::cos(angle);
    const Scalar sine = std::sin(angle);
    Matrix3<Scalar> matrix;
    matrix.data[0][0] = cosine;
    matrix.data[0][1] = -sine;
    matrix.data[0][2] = Scalar(0);
    matrix.data[1][0] = sine;
    matrix.data[1][1] = cosine;
    matrix.data[1][2] = Scalar(0);
    matrix.data[2][0] = Scalar(0);
    matrix.data[2][1] = Scalar(0);
    matrix.data[2][2] = Scalar(1);
    return matrix;
}

// 关节轴在父坐标系中的方向：a1=z, a2=y, a3=z, a4=x
template <typename Scalar>
Vector3<Scalar> axis_local(int joint) {
    switch (joint) {
        case 0:
            return Vector3<Scalar>::unit(2);
        case 1:
            return Vector3<Scalar>::unit(1);
        case 2:
            return Vector3<Scalar>::unit(2);
        case 3:
            return Vector3<Scalar>::unit(0);
        default:
            throw std::invalid_argument("关节编号超出范围");
    }
}

template <typename Scalar>
Matrix3<Scalar> local_rotation(int joint, const Scalar& angle) {
    switch (joint) {
        case 0:
            return rotation_z(angle);
        case 1:
            return rotation_y(angle);
        case 2:
            return rotation_z(angle);
        case 3:
            return rotation_x(angle);
        default:
            throw std::invalid_argument("关节编号超出范围");
    }
}

// 标量泛化的模型参数。字段名与 ModelParameters 一致，
// 从而 gimbal::parameters::add_parameter 对两者通用。
template <typename Scalar>
struct ScalarBodyParameters {
    Scalar mass = Scalar(0);
    Vector3<Scalar> com = Vector3<Scalar>::zero();
    Matrix3<Scalar> inertia = Matrix3<Scalar>::zero();
};

template <typename Scalar>
struct ScalarModelParameters {
    std::array<ScalarBodyParameters<Scalar>, 4> bodies{};
    std::array<Vector3<Scalar>, 4> joint_offsets{};
    Vector4<Scalar> damping = Vector4<Scalar>::zero();
    Scalar gravity = Scalar(0);
};

template <typename Scalar>
ScalarModelParameters<Scalar> convert_model(const ModelParameters& model) {
    ScalarModelParameters<Scalar> result;
    for (int i = 0; i < 4; ++i) {
        result.bodies[i].mass = Scalar(model.bodies[i].mass);
        for (int row = 0; row < 3; ++row) {
            result.bodies[i].com[row] = Scalar(model.bodies[i].com[row]);
            result.joint_offsets[i][row] = Scalar(model.joint_offsets[i][row]);
            for (int column = 0; column < 3; ++column) {
                result.bodies[i].inertia(row, column) =
                    Scalar(model.bodies[i].inertia(row, column));
            }
        }
    }
    for (int joint = 0; joint < 4; ++joint) {
        result.damping[joint] = Scalar(model.damping[joint]);
    }
    result.gravity = Scalar(model.gravity);
    return result;
}

template <typename Scalar>
struct ScalarTerrain {
    Scalar slope_angle = Scalar(0);
    Scalar slope_direction = Scalar(0);
};

template <typename Scalar>
ScalarTerrain<Scalar> convert_terrain(const Terrain& terrain) {
    ScalarTerrain<Scalar> result;
    result.slope_angle = Scalar(terrain.slope_angle);
    result.slope_direction = Scalar(terrain.slope_direction);
    return result;
}

// 链上各刚体的姿态、关节原点、质心与关节轴。位置均相对底盘参考点。
template <typename Scalar>
struct Chain {
    Vector3<Scalar> normal;
    std::array<Matrix3<Scalar>, 4> rotation{};
    std::array<Vector3<Scalar>, 4> joint_origin{};
    std::array<Vector3<Scalar>, 4> offset{};  // r_i = R_{i-1} o_i
    std::array<Vector3<Scalar>, 4> com{};     // g_i，相对底盘参考点
    std::array<Vector3<Scalar>, 4> axis{};
};

// 地形坐标系与法向量，整个仿真只求值一次
template <typename Scalar>
struct TerrainFrame {
    Matrix3<Scalar> rotation;
    Vector3<Scalar> normal;
};

template <typename Scalar>
TerrainFrame<Scalar> evaluate_terrain_frame(const ScalarTerrain<Scalar>& terrain) {
    TerrainFrame<Scalar> frame;
    frame.rotation = rotation_z(terrain.slope_direction) * rotation_y(terrain.slope_angle) *
                     rotation_z(-terrain.slope_direction);
    frame.normal = frame.rotation * Vector3<Scalar>::unit(2);
    return frame;
}

template <typename Scalar>
Chain<Scalar> evaluate_chain(const ScalarModelParameters<Scalar>& model,
                             const TerrainFrame<Scalar>& frame, const BaseState& base,
                             const Vector4<Scalar>& q) {
    Chain<Scalar> chain;
    const Matrix3<Scalar> base_rotation = frame.rotation * rotation_z(Scalar(base.yaw));
    chain.normal = frame.normal;
    Matrix3<Scalar> parent_rotation = base_rotation;
    Vector3<Scalar> parent_origin = Vector3<Scalar>::zero();
    for (int joint = 0; joint < 4; ++joint) {
        const Vector3<Scalar> offset = parent_rotation * model.joint_offsets[joint];
        const Vector3<Scalar> origin = parent_origin + offset;
        const Matrix3<Scalar> rotation = parent_rotation * local_rotation(joint, q[joint]);
        chain.offset[joint] = offset;
        chain.joint_origin[joint] = origin;
        chain.rotation[joint] = rotation;
        chain.axis[joint] = parent_rotation * axis_local<Scalar>(joint);
        chain.com[joint] = origin + rotation * model.bodies[joint].com;
        parent_rotation = rotation;
        parent_origin = origin;
    }
    return chain;
}

template <typename Scalar>
Chain<Scalar> evaluate_chain(const ScalarModelParameters<Scalar>& model,
                             const ScalarTerrain<Scalar>& terrain, const BaseState& base,
                             const Vector4<Scalar>& q) {
    return evaluate_chain(model, evaluate_terrain_frame(terrain), base, q);
}

// 质量矩阵求值的中间量，供导数阶段复用
template <typename Scalar>
struct MassInternals {
    std::array<Matrix34<Scalar>, 4> velocity_jacobian;
    std::array<Matrix34<Scalar>, 4> angular_jacobian;
    std::array<Matrix3<Scalar>, 4> inertia_world;
};

// 质量矩阵 M（DYNAMICS.md 7.2）
template <typename Scalar>
Matrix4<Scalar> evaluate_mass_matrix(const ScalarModelParameters<Scalar>& model,
                                     const Chain<Scalar>& chain,
                                     MassInternals<Scalar>* internals = nullptr) {
    Matrix4<Scalar> mass = Matrix4<Scalar>::zero();
    for (int i = 0; i < 4; ++i) {
        Matrix34<Scalar> velocity_jacobian = Matrix34<Scalar>::zero();
        Matrix34<Scalar> angular_jacobian = Matrix34<Scalar>::zero();
        for (int joint = 0; joint <= i; ++joint) {
            const Vector3<Scalar> arm = chain.com[i] - chain.joint_origin[joint];
            const Vector3<Scalar> velocity_column = cross_product(chain.axis[joint], arm);
            for (int row = 0; row < 3; ++row) {
                velocity_jacobian(row, joint) = velocity_column[row];
                angular_jacobian(row, joint) = chain.axis[joint][row];
            }
        }
        const Matrix3<Scalar> inertia_world =
            chain.rotation[i] * model.bodies[i].inertia * transpose(chain.rotation[i]);
        mass += model.bodies[i].mass * (transpose(velocity_jacobian) * velocity_jacobian);
        mass += transpose(angular_jacobian) * inertia_world * angular_jacobian;
        if (internals != nullptr) {
            internals->velocity_jacobian[i] = velocity_jacobian;
            internals->angular_jacobian[i] = angular_jacobian;
            internals->inertia_world[i] = inertia_world;
        }
    }
    return mass;
}

// 偏置力递推的中间量，供导数阶段复用
template <typename Scalar>
struct BiasInternals {
    std::array<Vector3<Scalar>, 4> omega;
    std::array<Vector3<Scalar>, 4> alpha;
    std::array<Vector3<Scalar>, 4> origin_acceleration;
    std::array<Vector3<Scalar>, 4> com_acceleration;
    std::array<Vector3<Scalar>, 4> force;
    std::array<Vector3<Scalar>, 4> couple;
    std::array<Vector3<Scalar>, 4> joint_force;
    std::array<Vector3<Scalar>, 4> joint_moment;
};

// 偏置力 b：逆动力学递推（DYNAMICS.md 第 8 节），令 q̈ = 0。
// mass_internals 非空时直接复用其中已算好的惯量世界矩阵，避免重复求值。
template <typename Scalar>
Vector4<Scalar> evaluate_bias(const ScalarModelParameters<Scalar>& model,
                              const Chain<Scalar>& chain, const BaseState& base,
                              const Vector4<Scalar>& qdot,
                              BiasInternals<Scalar>* internals = nullptr,
                              const MassInternals<Scalar>* mass_internals = nullptr) {
    BiasInternals<Scalar> data;
    Vector3<Scalar> gravity_vector = Vector3<Scalar>::zero();
    gravity_vector[2] = -model.gravity;
    const Vector3<Scalar> base_omega = chain.normal * Scalar(base.yaw_rate);
    const Vector3<Scalar> base_alpha = chain.normal * Scalar(base.yaw_acceleration);
    Vector3<Scalar> base_acceleration;
    for (int i = 0; i < 3; ++i) {
        base_acceleration[i] = Scalar(base.acceleration[i]);
    }

    Vector3<Scalar> previous_omega = base_omega;
    Vector3<Scalar> previous_alpha = base_alpha;
    Vector3<Scalar> previous_origin_acceleration = base_acceleration;
    for (int i = 0; i < 4; ++i) {
        const Vector3<Scalar> offset = chain.offset[i];
        const Vector3<Scalar> origin_acceleration =
            previous_origin_acceleration + cross_product(previous_alpha, offset) +
            cross_product(previous_omega, cross_product(previous_omega, offset));
        const Vector3<Scalar> axis = chain.axis[i];
        const Vector3<Scalar> angular_velocity = previous_omega + axis * qdot[i];
        // q̈ = 0，因此关节角加速度只来自上一刚体的角速度与关节轴的耦合项
        const Vector3<Scalar> angular_acceleration =
            previous_alpha + cross_product(previous_omega, axis * qdot[i]);
        const Vector3<Scalar> com_relative = chain.com[i] - chain.joint_origin[i];
        const Vector3<Scalar> linear_acceleration =
            origin_acceleration + cross_product(angular_acceleration, com_relative) +
            cross_product(angular_velocity, cross_product(angular_velocity, com_relative));
        data.omega[i] = angular_velocity;
        data.alpha[i] = angular_acceleration;
        data.origin_acceleration[i] = origin_acceleration;
        data.com_acceleration[i] = linear_acceleration;
        previous_omega = angular_velocity;
        previous_alpha = angular_acceleration;
        previous_origin_acceleration = origin_acceleration;
    }

    for (int i = 0; i < 4; ++i) {
        const Matrix3<Scalar> inertia_world =
            (mass_internals != nullptr)
                ? (*mass_internals).inertia_world[i]
                : chain.rotation[i] * model.bodies[i].inertia * transpose(chain.rotation[i]);
        data.force[i] = model.bodies[i].mass * (data.com_acceleration[i] - gravity_vector);
        data.couple[i] = inertia_world * data.alpha[i] +
                         cross_product(data.omega[i], inertia_world * data.omega[i]);
    }

    Vector4<Scalar> bias = Vector4<Scalar>::zero();
    Vector3<Scalar> child_force = Vector3<Scalar>::zero();
    Vector3<Scalar> child_moment = Vector3<Scalar>::zero();
    for (int i = 3; i >= 0; --i) {
        Vector3<Scalar> moment =
            data.couple[i] + cross_product(chain.com[i] - chain.joint_origin[i], data.force[i]);
        if (i < 3) {
            moment += child_moment +
                      cross_product(chain.joint_origin[i + 1] - chain.joint_origin[i], child_force);
        }
        bias[i] = dot_product(chain.axis[i], moment);
        data.joint_force[i] = data.force[i] + child_force;
        data.joint_moment[i] = moment;
        child_force = data.joint_force[i];
        child_moment = moment;
    }
    if (internals != nullptr) {
        *internals = data;
    }
    return bias;
}

// 求解 M q̈ = rhs（Cholesky 分解）
inline Vector4<Real> solve_mass(const Matrix4<Real>& mass, const Vector4<Real>& rhs) {
    return math::solve4(mass, rhs);
}

// 状态导数 y' = f(y, u)，y = [q; q̇]，u = τ
template <typename Scalar>
StateVector<Scalar> evaluate_derivative(const ScalarModelParameters<Scalar>& model,
                                        const ScalarTerrain<Scalar>& terrain,
                                        const BaseState& base, const StateVector<Scalar>& state,
                                        const Vector4<Scalar>& torque) {
    Vector4<Scalar> q;
    Vector4<Scalar> qdot;
    for (int i = 0; i < 4; ++i) {
        q[i] = state[i];
        qdot[i] = state[4 + i];
    }
    const Chain<Scalar> chain = evaluate_chain(model, terrain, base, q);
    MassInternals<Scalar> mass_internals;
    const Matrix4<Scalar> mass = evaluate_mass_matrix(model, chain, &mass_internals);
    const Vector4<Scalar> bias = evaluate_bias(
        model, chain, base, qdot, static_cast<BiasInternals<Scalar>*>(nullptr), &mass_internals);
    const Vector4<Scalar> rhs = torque - elementwise(model.damping, qdot) - bias;
    const Vector4<Scalar> qdd = solve_mass(mass, rhs);
    StateVector<Scalar> derivative;
    for (int i = 0; i < 4; ++i) {
        derivative[i] = qdot[i];
        derivative[4 + i] = qdd[i];
    }
    return derivative;
}

}  // namespace detail
}  // namespace gimbal

