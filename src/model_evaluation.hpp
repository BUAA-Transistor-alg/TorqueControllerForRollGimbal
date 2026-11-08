#pragma once

// 内部：标量泛化的动力学求值。
//
// 所有物理量都转换成模板标量 Scalar：double 用于前向仿真；std::complex<double>
// 用于单步闭式实现（dynamics_model.cpp）的复步长求导。仿真的梯度路径直接对
// 本文件的求值过程做解析方向导数（见 stage_derivatives.cpp），不走复步长。
//
// 质量矩阵按 DYNAMICS.md 7.2 的雅可比形式组装；偏置力 b 按第 8 节的逆动力学
// 递推形式计算（令 q̈ = 0）。递推形式只含四则运算与三角函数，对 q、q̇ 以及全部
// 模型参数一阶可微，因此不需要质量矩阵对 q 的二阶偏导。

#include <array>
#include <cmath>
#include <complex>
#include <stdexcept>
#include <type_traits>

#include <Eigen/Cholesky>
#include <Eigen/Core>

#include "gimbal_dynamics/dynamics_model.hpp"

namespace gimbal {
namespace detail {

using Complex = std::complex<double>;

// 复步长的虚部增量。虚部按线性方式参与运算，不出现减法抵消。
inline constexpr double kComplexStep = 1e-20;

template <typename Scalar>
using Vector3 = Eigen::Matrix<Scalar, 3, 1>;
template <typename Scalar>
using Matrix3 = Eigen::Matrix<Scalar, 3, 3>;
template <typename Scalar>
using Vector4 = Eigen::Matrix<Scalar, 4, 1>;
template <typename Scalar>
using Matrix4 = Eigen::Matrix<Scalar, 4, 4>;
template <typename Scalar>
using StateVector = Eigen::Matrix<Scalar, 8, 1>;

// 叉乘与点积显式展开：Eigen 的 cross 只接受三维向量，dot 对复数会取共轭。
template <typename LhsDerived, typename RhsDerived>
Eigen::Matrix<typename LhsDerived::Scalar, 3, 1> cross_product(
    const Eigen::MatrixBase<LhsDerived>& lhs, const Eigen::MatrixBase<RhsDerived>& rhs) {
    using Scalar = typename LhsDerived::Scalar;
    static_assert(std::is_same<Scalar, typename RhsDerived::Scalar>::value,
                  "叉乘两端的标量类型必须一致");
    return Eigen::Matrix<Scalar, 3, 1>(lhs.y() * rhs.z() - lhs.z() * rhs.y(),
                                       lhs.z() * rhs.x() - lhs.x() * rhs.z(),
                                       lhs.x() * rhs.y() - lhs.y() * rhs.x());
}

template <typename LhsDerived, typename RhsDerived>
typename LhsDerived::Scalar dot_product(const Eigen::MatrixBase<LhsDerived>& lhs,
                                        const Eigen::MatrixBase<RhsDerived>& rhs) {
    using Scalar = typename LhsDerived::Scalar;
    static_assert(std::is_same<Scalar, typename RhsDerived::Scalar>::value,
                  "点积两端的标量类型必须一致");
    return (lhs.transpose() * rhs)(0, 0);
}

template <typename Scalar>
Matrix3<Scalar> rotation_x(const Scalar& angle) {
    const Scalar cosine = std::cos(angle);
    const Scalar sine = std::sin(angle);
    Matrix3<Scalar> matrix;
    matrix << Scalar(1), Scalar(0), Scalar(0),
              Scalar(0), cosine, -sine,
              Scalar(0), sine, cosine;
    return matrix;
}

template <typename Scalar>
Matrix3<Scalar> rotation_y(const Scalar& angle) {
    const Scalar cosine = std::cos(angle);
    const Scalar sine = std::sin(angle);
    Matrix3<Scalar> matrix;
    matrix << cosine, Scalar(0), sine,
              Scalar(0), Scalar(1), Scalar(0),
              -sine, Scalar(0), cosine;
    return matrix;
}

template <typename Scalar>
Matrix3<Scalar> rotation_z(const Scalar& angle) {
    const Scalar cosine = std::cos(angle);
    const Scalar sine = std::sin(angle);
    Matrix3<Scalar> matrix;
    matrix << cosine, -sine, Scalar(0),
              sine, cosine, Scalar(0),
              Scalar(0), Scalar(0), Scalar(1);
    return matrix;
}

// 关节轴在父坐标系中的方向：a1=z, a2=y, a3=z, a4=x
template <typename Scalar>
Vector3<Scalar> axis_local(int joint) {
    switch (joint) {
        case 0:
            return Vector3<Scalar>::UnitZ();
        case 1:
            return Vector3<Scalar>::UnitY();
        case 2:
            return Vector3<Scalar>::UnitZ();
        case 3:
            return Vector3<Scalar>::UnitX();
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
    Vector3<Scalar> com = Vector3<Scalar>::Zero();
    Matrix3<Scalar> inertia = Matrix3<Scalar>::Zero();
};

template <typename Scalar>
struct ScalarModelParameters {
    std::array<ScalarBodyParameters<Scalar>, 4> bodies{};
    std::array<Vector3<Scalar>, 4> joint_offsets{};
    Vector4<Scalar> damping = Vector4<Scalar>::Zero();
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

// 标量泛化模型之间的转换（数值保持不变）
template <typename ToScalar, typename FromScalar>
ScalarModelParameters<ToScalar> convert_scalar_model(const ScalarModelParameters<FromScalar>& model) {
    ScalarModelParameters<ToScalar> result;
    for (int i = 0; i < 4; ++i) {
        result.bodies[i].mass = ToScalar(model.bodies[i].mass);
        for (int row = 0; row < 3; ++row) {
            result.bodies[i].com[row] = ToScalar(model.bodies[i].com[row]);
            result.joint_offsets[i][row] = ToScalar(model.joint_offsets[i][row]);
            for (int column = 0; column < 3; ++column) {
                result.bodies[i].inertia(row, column) =
                    ToScalar(model.bodies[i].inertia(row, column));
            }
        }
    }
    for (int joint = 0; joint < 4; ++joint) {
        result.damping[joint] = ToScalar(model.damping[joint]);
    }
    result.gravity = ToScalar(model.gravity);
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

template <typename Scalar>
Chain<Scalar> evaluate_chain(const ScalarModelParameters<Scalar>& model,
                             const ScalarTerrain<Scalar>& terrain, const BaseState& base,
                             const Vector4<Scalar>& q) {
    Chain<Scalar> chain;
    const Matrix3<Scalar> terrain_frame = rotation_z(terrain.slope_direction) *
                                         rotation_y(terrain.slope_angle) *
                                         rotation_z(-terrain.slope_direction);
    const Matrix3<Scalar> base_rotation = terrain_frame * rotation_z(Scalar(base.yaw));
    chain.normal = terrain_frame * Vector3<Scalar>::UnitZ();
    Matrix3<Scalar> parent_rotation = base_rotation;
    Vector3<Scalar> parent_origin = Vector3<Scalar>::Zero();
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

// 质量矩阵求值的中间量，供导数阶段复用
template <typename Scalar>
struct MassInternals {
    std::array<Eigen::Matrix<Scalar, 3, 4>, 4> velocity_jacobian{};
    std::array<Eigen::Matrix<Scalar, 3, 4>, 4> angular_jacobian{};
    std::array<Matrix3<Scalar>, 4> inertia_world{};
};

// 质量矩阵 M（DYNAMICS.md 7.2）
template <typename Scalar>
Matrix4<Scalar> evaluate_mass_matrix(const ScalarModelParameters<Scalar>& model,
                                     const Chain<Scalar>& chain,
                                     MassInternals<Scalar>* internals = nullptr) {
    Matrix4<Scalar> mass = Matrix4<Scalar>::Zero();
    for (int i = 0; i < 4; ++i) {
        Eigen::Matrix<Scalar, 3, 4> velocity_jacobian = Eigen::Matrix<Scalar, 3, 4>::Zero();
        Eigen::Matrix<Scalar, 3, 4> angular_jacobian = Eigen::Matrix<Scalar, 3, 4>::Zero();
        for (int joint = 0; joint <= i; ++joint) {
            velocity_jacobian.col(joint) =
                cross_product(chain.axis[joint], chain.com[i] - chain.joint_origin[joint]);
            angular_jacobian.col(joint) = chain.axis[joint];
        }
        const Matrix3<Scalar> inertia_world =
            chain.rotation[i] * model.bodies[i].inertia * chain.rotation[i].transpose();
        mass += model.bodies[i].mass * velocity_jacobian.transpose() * velocity_jacobian;
        mass += angular_jacobian.transpose() * inertia_world * angular_jacobian;
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
    std::array<Vector3<Scalar>, 4> omega{};
    std::array<Vector3<Scalar>, 4> alpha{};
    std::array<Vector3<Scalar>, 4> origin_acceleration{};
    std::array<Vector3<Scalar>, 4> com_acceleration{};
    std::array<Vector3<Scalar>, 4> force{};
    std::array<Vector3<Scalar>, 4> couple{};
    std::array<Vector3<Scalar>, 4> joint_force{};
    std::array<Vector3<Scalar>, 4> joint_moment{};
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
    const Vector3<Scalar> gravity_vector(Scalar(0), Scalar(0), -model.gravity);
    const Vector3<Scalar> base_omega = Scalar(base.yaw_rate) * chain.normal;
    const Vector3<Scalar> base_alpha = Scalar(base.yaw_acceleration) * chain.normal;
    const Vector3<Scalar> base_acceleration(
        Scalar(base.acceleration.x()), Scalar(base.acceleration.y()),
        Scalar(base.acceleration.z()));

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
                : chain.rotation[i] * model.bodies[i].inertia * chain.rotation[i].transpose();
        data.force[i] = model.bodies[i].mass * (data.com_acceleration[i] - gravity_vector);
        data.couple[i] = inertia_world * data.alpha[i] +
                         cross_product(data.omega[i], inertia_world * data.omega[i]);
    }

    Vector4<Scalar> bias = Vector4<Scalar>::Zero();
    Vector3<Scalar> child_force = Vector3<Scalar>::Zero();
    Vector3<Scalar> child_moment = Vector3<Scalar>::Zero();
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

// 求解 M q̈ = rhs。double 用 LLT；复数用实部 LLT 加一阶虚部修正，
// 保证复步长求得的导数精确。
inline Vector4<double> solve_mass(const Matrix4<double>& mass, const Vector4<double>& rhs) {
    const Eigen::LLT<Matrix4<double>> decomposition(mass);
    if (decomposition.info() != Eigen::Success) {
        throw std::runtime_error("惯量矩阵不是正定矩阵");
    }
    return decomposition.solve(rhs);
}

inline Vector4<Complex> solve_mass(const Matrix4<Complex>& mass, const Vector4<Complex>& rhs) {
    const Eigen::LLT<Matrix4<double>> decomposition(Matrix4<double>(mass.real()));
    if (decomposition.info() != Eigen::Success) {
        throw std::runtime_error("惯量矩阵不是正定矩阵");
    }
    const Vector4<double> real_solution = decomposition.solve(Vector4<double>(rhs.real()));
    const Vector4<double> imag_rhs = Vector4<double>(rhs.imag()) - Matrix4<double>(mass.imag()) * real_solution;
    const Vector4<double> imag_solution = decomposition.solve(imag_rhs);
    Vector4<Complex> solution;
    for (int i = 0; i < 4; ++i) {
        solution[i] = Complex(real_solution[i], imag_solution[i]);
    }
    return solution;
}

// 状态导数 y' = f(y, u)，y = [q; q̇]，u = τ
template <typename Scalar>
StateVector<Scalar> evaluate_derivative(const ScalarModelParameters<Scalar>& model,
                                        const ScalarTerrain<Scalar>& terrain,
                                        const BaseState& base, const StateVector<Scalar>& state,
                                        const Vector4<Scalar>& torque) {
    const Vector4<Scalar> q = state.template head<4>();
    const Vector4<Scalar> qdot = state.template tail<4>();
    const Chain<Scalar> chain = evaluate_chain(model, terrain, base, q);
    MassInternals<Scalar> mass_internals;
    const Matrix4<Scalar> mass = evaluate_mass_matrix(model, chain, &mass_internals);
    const Vector4<Scalar> bias = evaluate_bias(
        model, chain, base, qdot, static_cast<BiasInternals<Scalar>*>(nullptr), &mass_internals);
    const Vector4<Scalar> rhs = torque - model.damping.cwiseProduct(qdot) - bias;
    const Vector4<Scalar> qdd = solve_mass(mass, rhs);
    StateVector<Scalar> derivative;
    derivative.template head<4>() = qdot;
    derivative.template tail<4>() = qdd;
    return derivative;
}

}  // namespace detail
}  // namespace gimbal
