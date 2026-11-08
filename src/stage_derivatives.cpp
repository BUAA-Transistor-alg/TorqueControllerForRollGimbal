#include "stage_jacobians.hpp"

#include <array>
#include <stdexcept>

#include <Eigen/Cholesky>

#include "model_evaluation.hpp"

namespace gimbal {
namespace detail {
namespace {

// 方向编号约定：
//   0..3   q_0..q_3
//   4..7   q̇_0..q̇_3
//   8..19  刚体质心：8 + 3*body + component
//   20..31 关节安装偏置：20 + 3*joint + component
constexpr int kStateDirectionCount = 8;
constexpr int kParameterDirectionCount = 32;

using Vector3d = Vector3<double>;
using Matrix3d = Matrix3<double>;
using Vector4d = Vector4<double>;
using Matrix4d = Matrix4<double>;
using Array3 = std::array<Vector3d, 4>;
using ArrayMatrix = std::array<Matrix3d, 4>;

Matrix3d skew_matrix(const Vector3d& vector) {
    Matrix3d matrix;
    matrix << 0.0, -vector.z(), vector.y(),
              vector.z(), 0.0, -vector.x(),
              -vector.y(), vector.x(), 0.0;
    return matrix;
}

Array3 zero_vectors() {
    Array3 result;
    for (int i = 0; i < 4; ++i) {
        result[i].setZero();
    }
    return result;
}

// 各刚体角速度、角加速度、质心加速度的方向导数
struct KinematicsDirection {
    Array3 omega;
    Array3 alpha;
    Array3 com_acceleration;

    KinematicsDirection() : omega(zero_vectors()), alpha(zero_vectors()), com_acceleration(zero_vectors()) {}
};

// 全量运动学方向导数：用于 q 方向与 q̇ 方向
KinematicsDirection propagate_kinematics(const StageData& stage,
                                         const std::array<double, 4>& delta_velocity,
                                         const Array3& delta_offset, const Array3& delta_axis,
                                         const Array3& delta_com_relative) {
    const Chain<double>& chain = stage.chain;
    KinematicsDirection result;

    Vector3d previous_omega = stage.base.yaw_rate * chain.normal;
    Vector3d previous_alpha = stage.base.yaw_acceleration * chain.normal;
    Vector3d previous_origin_acceleration = stage.base.acceleration;
    Vector3d delta_previous_omega = Vector3d::Zero();
    Vector3d delta_previous_alpha = Vector3d::Zero();
    Vector3d delta_previous_origin_acceleration = Vector3d::Zero();
    for (int i = 0; i < 4; ++i) {
        const Vector3d& offset = chain.offset[i];
        const Vector3d& axis = chain.axis[i];
        const Vector3d& delta_offset_i = delta_offset[i];
        const Vector3d& delta_axis_i = delta_axis[i];
        const double velocity = stage.qdot[i];
        const double delta_velocity_i = delta_velocity[i];

        const Vector3d delta_omega =
            delta_previous_omega + delta_axis_i * velocity + axis * delta_velocity_i;
        const Vector3d delta_alpha =
            delta_previous_alpha + cross_product(delta_previous_omega, axis * velocity) +
            cross_product(previous_omega, delta_axis_i * velocity + axis * delta_velocity_i);
        const Vector3d origin_acceleration =
            previous_origin_acceleration + cross_product(previous_alpha, offset) +
            cross_product(previous_omega, cross_product(previous_omega, offset));
        const Vector3d delta_origin_acceleration =
            delta_previous_origin_acceleration + cross_product(delta_previous_alpha, offset) +
            cross_product(previous_alpha, delta_offset_i) +
            cross_product(delta_previous_omega, cross_product(previous_omega, offset)) +
            cross_product(previous_omega, cross_product(delta_previous_omega, offset)) +
            cross_product(previous_omega, cross_product(previous_omega, delta_offset_i));
        const Vector3d com_relative = chain.com[i] - chain.joint_origin[i];
        const Vector3d& delta_com_relative_i = delta_com_relative[i];
        const Vector3d angular_velocity = previous_omega + axis * velocity;
        const Vector3d angular_acceleration =
            previous_alpha + cross_product(previous_omega, axis * velocity);
        const Vector3d delta_com_acceleration =
            delta_origin_acceleration + cross_product(delta_alpha, com_relative) +
            cross_product(angular_acceleration, delta_com_relative_i) +
            cross_product(delta_omega, cross_product(angular_velocity, com_relative)) +
            cross_product(angular_velocity, cross_product(delta_omega, com_relative)) +
            cross_product(angular_velocity, cross_product(angular_velocity, delta_com_relative_i));

        result.omega[i] = delta_omega;
        result.alpha[i] = delta_alpha;
        result.com_acceleration[i] = delta_com_acceleration;
        previous_omega = angular_velocity;
        previous_alpha = angular_acceleration;
        previous_origin_acceleration = origin_acceleration;
        delta_previous_omega = delta_omega;
        delta_previous_alpha = delta_alpha;
        delta_previous_origin_acceleration = delta_origin_acceleration;
    }
    return result;
}

// 刚体质心参数方向：只有质心加速度的方向导数非零
KinematicsDirection com_kinematics(const StageData& stage, int body,
                                   const Vector3d& delta_com_relative) {
    KinematicsDirection result;
    const Vector3d& omega = stage.bias_internals.omega[body];
    const Vector3d& alpha = stage.bias_internals.alpha[body];
    result.com_acceleration[body] = cross_product(alpha, delta_com_relative) +
                                    cross_product(omega, cross_product(omega, delta_com_relative));
    return result;
}

// 关节安装偏置方向：各刚体质心加速度方向导数等于原点加速度方向导数
KinematicsDirection offset_kinematics(const StageData& stage, int joint,
                                      const Vector3d& delta_offset) {
    KinematicsDirection result;
    const Vector3d previous_omega =
        (joint == 0) ? Vector3d(stage.base.yaw_rate * stage.chain.normal)
                     : stage.bias_internals.omega[joint - 1];
    const Vector3d previous_alpha =
        (joint == 0) ? Vector3d(stage.base.yaw_acceleration * stage.chain.normal)
                     : stage.bias_internals.alpha[joint - 1];
    const Vector3d origin_acceleration =
        cross_product(previous_alpha, delta_offset) +
        cross_product(previous_omega, cross_product(previous_omega, delta_offset));
    for (int i = joint; i < 4; ++i) {
        result.com_acceleration[i] = origin_acceleration;
    }
    return result;
}

ArrayMatrix zero_matrices() {
    ArrayMatrix result;
    for (int i = 0; i < 4; ++i) {
        result[i].setZero();
    }
    return result;
}

// 由各刚体的加速度、角速度、角加速度与惯量方向导数求偏置力方向导数
Vector4d evaluate_bias_direction(const ScalarModelParameters<double>& model, const StageData& stage,
                                 const KinematicsDirection& kinematics,
                                 const ArrayMatrix& delta_inertia_world,
                                 const Array3& delta_com_relative, const Array3& delta_offset,
                                 const Array3& delta_axis) {
    const BiasInternals<double>& internals = stage.bias_internals;
    Array3 delta_force = zero_vectors();
    Array3 delta_couple = zero_vectors();
    for (int i = 0; i < 4; ++i) {
        const Matrix3d& inertia_world = stage.mass_internals.inertia_world[i];
        delta_force[i] = model.bodies[i].mass * kinematics.com_acceleration[i];
        delta_couple[i] = delta_inertia_world[i] * internals.alpha[i] +
                          inertia_world * kinematics.alpha[i] +
                          cross_product(kinematics.omega[i], inertia_world * internals.omega[i]) +
                          cross_product(internals.omega[i],
                                        delta_inertia_world[i] * internals.omega[i] +
                                            inertia_world * kinematics.omega[i]);
    }

    Vector4d result = Vector4d::Zero();
    Vector3d child_force = Vector3d::Zero();
    Vector3d child_moment = Vector3d::Zero();
    for (int i = 3; i >= 0; --i) {
        const Vector3d& com_relative = stage.chain.com[i] - stage.chain.joint_origin[i];
        Vector3d moment = delta_couple[i] +
                          cross_product(delta_com_relative[i], internals.force[i]) +
                          cross_product(com_relative, delta_force[i]);
        if (i < 3) {
            moment += child_moment +
                      cross_product(delta_offset[i + 1], internals.joint_force[i + 1]) +
                      cross_product(stage.chain.offset[i + 1], child_force);
        }
        result[i] = dot_product(delta_axis[i], internals.joint_moment[i]) +
                    dot_product(stage.chain.axis[i], moment);
        child_force = delta_force[i] + child_force;
        child_moment = moment;
    }
    return result;
}

// 全量质量矩阵方向导数（q 方向）
Matrix4d evaluate_mass_direction(const StageData& stage, const ScalarModelParameters<double>& model,
                                 const Array3& delta_axis, const Array3& delta_com,
                                 const Array3& delta_origin, const ArrayMatrix& delta_rotation) {
    Matrix4d mass_derivative = Matrix4d::Zero();
    for (int i = 0; i < 4; ++i) {
        const Eigen::Matrix<double, 3, 4>& velocity_jacobian =
            stage.mass_internals.velocity_jacobian[i];
        const Eigen::Matrix<double, 3, 4>& angular_jacobian =
            stage.mass_internals.angular_jacobian[i];
        const Matrix3d& inertia_world = stage.mass_internals.inertia_world[i];

        Eigen::Matrix<double, 3, 4> velocity_jacobian_derivative =
            Eigen::Matrix<double, 3, 4>::Zero();
        Eigen::Matrix<double, 3, 4> angular_jacobian_derivative =
            Eigen::Matrix<double, 3, 4>::Zero();
        for (int joint = 0; joint <= i; ++joint) {
            velocity_jacobian_derivative.col(joint) =
                cross_product(delta_axis[joint], stage.chain.com[i] - stage.chain.joint_origin[joint]) +
                cross_product(stage.chain.axis[joint], delta_com[i] - delta_origin[joint]);
            angular_jacobian_derivative.col(joint) = delta_axis[joint];
        }
        const Matrix3d& inertia = model.bodies[i].inertia;
        const Matrix3d delta_inertia_world =
            delta_rotation[i] * inertia * stage.chain.rotation[i].transpose() +
            stage.chain.rotation[i] * inertia * delta_rotation[i].transpose();
        mass_derivative +=
            model.bodies[i].mass *
            (velocity_jacobian_derivative.transpose() * velocity_jacobian +
             velocity_jacobian.transpose() * velocity_jacobian_derivative);
        mass_derivative +=
            angular_jacobian_derivative.transpose() * inertia_world * angular_jacobian;
        mass_derivative += angular_jacobian.transpose() * delta_inertia_world * angular_jacobian;
        mass_derivative +=
            angular_jacobian.transpose() * inertia_world * angular_jacobian_derivative;
    }
    return mass_derivative;
}

// 刚体质心参数方向的质量矩阵方向导数（只有该刚体的雅可比变化）
Matrix4d evaluate_mass_direction_com(const StageData& stage, const ScalarModelParameters<double>& model,
                                     int body, const Vector3d& delta_com_relative) {
    const Eigen::Matrix<double, 3, 4>& velocity_jacobian =
        stage.mass_internals.velocity_jacobian[body];
    Eigen::Matrix<double, 3, 4> velocity_jacobian_derivative = Eigen::Matrix<double, 3, 4>::Zero();
    for (int joint = 0; joint <= body; ++joint) {
        velocity_jacobian_derivative.col(joint) =
            cross_product(stage.chain.axis[joint], delta_com_relative);
    }
    return model.bodies[body].mass *
           (velocity_jacobian_derivative.transpose() * velocity_jacobian +
            velocity_jacobian.transpose() * velocity_jacobian_derivative);
}

// 关节安装偏置方向的质量矩阵方向导数：i ≥ joint 且 j < joint 的雅可比项变化
Matrix4d evaluate_mass_direction_offset(const StageData& stage,
                                        const ScalarModelParameters<double>& model, int joint,
                                        const Vector3d& delta_offset) {
    Matrix4d mass_derivative = Matrix4d::Zero();
    for (int i = joint; i < 4; ++i) {
        const Eigen::Matrix<double, 3, 4>& velocity_jacobian =
            stage.mass_internals.velocity_jacobian[i];
        Eigen::Matrix<double, 3, 4> velocity_jacobian_derivative = Eigen::Matrix<double, 3, 4>::Zero();
        for (int smaller = 0; smaller < joint; ++smaller) {
            velocity_jacobian_derivative.col(smaller) =
                cross_product(stage.chain.axis[smaller], delta_offset);
        }
        mass_derivative +=
            model.bodies[i].mass *
            (velocity_jacobian_derivative.transpose() * velocity_jacobian +
             velocity_jacobian.transpose() * velocity_jacobian_derivative);
    }
    return mass_derivative;
}

// 偏置力对力/力偶方向导数的响应（质量、惯量参数方向使用）
Vector4d bias_direction_from_internal(const StageData& stage,
                                      const std::array<Vector3d, 4>& delta_force,
                                      const std::array<Vector3d, 4>& delta_couple) {
    Vector4d result = Vector4d::Zero();
    Vector3d child_force = Vector3d::Zero();
    Vector3d child_moment = Vector3d::Zero();
    for (int i = 3; i >= 0; --i) {
        Vector3d moment = delta_couple[i] +
            cross_product(stage.chain.com[i] - stage.chain.joint_origin[i], delta_force[i]);
        if (i < 3) {
            moment += child_moment + cross_product(stage.chain.offset[i + 1], child_force);
        }
        result[i] = dot_product(stage.chain.axis[i], moment);
        child_force = delta_force[i] + child_force;
        child_moment = moment;
    }
    return result;
}

// 质量参数：dM/dm_i = Jv_i^T Jv_i，dF_i = a_g,i - g_W
void mass_parameter_direction(const StageData& stage, const ScalarModelParameters<double>& model,
                              int body, Matrix4d& mass_derivative, Vector4d& bias_derivative) {
    const Eigen::Matrix<double, 3, 4>& velocity_jacobian =
        stage.mass_internals.velocity_jacobian[body];
    mass_derivative = velocity_jacobian.transpose() * velocity_jacobian;

    std::array<Vector3d, 4> delta_force;
    std::array<Vector3d, 4> delta_couple;
    for (int i = 0; i < 4; ++i) {
        delta_force[i].setZero();
        delta_couple[i].setZero();
    }
    delta_force[body] = stage.bias_internals.com_acceleration[body] -
                        Vector3d(0.0, 0.0, -model.gravity);
    bias_derivative = bias_direction_from_internal(stage, delta_force, delta_couple);
}

// 惯量参数：对称分量 (row,column)
void inertia_parameter_direction(const StageData& stage, int body, int entry,
                                 Matrix4d& mass_derivative, Vector4d& bias_derivative) {
    static constexpr int kRow[6] = {0, 1, 2, 0, 0, 1};
    static constexpr int kColumn[6] = {0, 1, 2, 1, 2, 2};
    const int row = kRow[entry];
    const int column = kColumn[entry];
    Matrix3d inertia_direction = Matrix3d::Zero();
    inertia_direction(row, column) += 1.0;
    if (row != column) {
        inertia_direction(column, row) += 1.0;
    }
    const Matrix3d& rotation = stage.chain.rotation[body];
    const Matrix3d inertia_world_direction = rotation * inertia_direction * rotation.transpose();

    const Eigen::Matrix<double, 3, 4>& angular_jacobian =
        stage.mass_internals.angular_jacobian[body];
    mass_derivative = angular_jacobian.transpose() * inertia_world_direction * angular_jacobian;

    std::array<Vector3d, 4> delta_force;
    std::array<Vector3d, 4> delta_couple;
    for (int i = 0; i < 4; ++i) {
        delta_force[i].setZero();
        delta_couple[i].setZero();
    }
    const Vector3d& omega = stage.bias_internals.omega[body];
    const Vector3d& alpha = stage.bias_internals.alpha[body];
    delta_couple[body] = inertia_world_direction * alpha +
                         cross_product(omega, inertia_world_direction * omega);
    bias_derivative = bias_direction_from_internal(stage, delta_force, delta_couple);
}

Matrix4d invert_mass(const Matrix4d& mass) {
    const Eigen::LLT<Matrix4d> decomposition(mass);
    if (decomposition.info() != Eigen::Success) {
        throw std::runtime_error("惯量矩阵不是正定矩阵");
    }
    return decomposition.solve(Matrix4d::Identity());
}

template <int N>
struct DirectionDerivatives {
    std::array<Matrix4d, N> mass{};
    std::array<Vector4d, N> bias{};
};

// 按方向类型分别求值：q 与 q̇ 方向做全量运动学传播，
// 质心与偏置方向走只含必要项的短路径。
template <int N>
DirectionDerivatives<N> evaluate_directions(const StageData& stage,
                                            const ScalarModelParameters<double>& model) {
    const Chain<double>& chain = stage.chain;
    DirectionDerivatives<N> result;
    for (int d = 0; d < N; ++d) {
        result.mass[d].setZero();
        result.bias[d].setZero();
    }
    const Array3 zero_vectors_value = zero_vectors();
    const ArrayMatrix zero_matrices_value = zero_matrices();
    const std::array<double, 4> zero_angles = {0.0, 0.0, 0.0, 0.0};

    // 广义坐标方向 0..3
    for (int k = 0; k < 4 && k < N; ++k) {
        const Matrix3d axis_skew = skew_matrix(axis_local<double>(k));
        ArrayMatrix delta_rotation = zero_matrices_value;
        for (int i = k; i < 4; ++i) {
            delta_rotation[i] = chain.rotation[k] * axis_skew * chain.rotation[k].transpose() *
                                chain.rotation[i];
        }
        Array3 delta_offset = zero_vectors_value;
        for (int m = 1; m < 4; ++m) {
            if (k < m) {
                delta_offset[m] = delta_rotation[m - 1] * model.joint_offsets[m];
            }
        }
        Array3 delta_origin = zero_vectors_value;
        for (int i = 0; i < 4; ++i) {
            for (int m = 0; m <= i; ++m) {
                delta_origin[i] += delta_offset[m];
            }
        }
        Array3 delta_axis = zero_vectors_value;
        for (int j = 1; j < 4; ++j) {
            delta_axis[j] = delta_rotation[j - 1] * axis_local<double>(j);
        }
        Array3 delta_com_relative = zero_vectors_value;
        Array3 delta_com = zero_vectors_value;
        for (int i = 0; i < 4; ++i) {
            delta_com_relative[i] = delta_rotation[i] * model.bodies[i].com;
            delta_com[i] = delta_origin[i] + delta_com_relative[i];
        }
        const KinematicsDirection kinematics =
            propagate_kinematics(stage, zero_angles, delta_offset, delta_axis, delta_com_relative);
        ArrayMatrix delta_inertia_world = zero_matrices_value;
        for (int i = 0; i < 4; ++i) {
            delta_inertia_world[i] =
                delta_rotation[i] * model.bodies[i].inertia * chain.rotation[i].transpose() +
                chain.rotation[i] * model.bodies[i].inertia * delta_rotation[i].transpose();
        }
        result.bias[k] = evaluate_bias_direction(model, stage, kinematics, delta_inertia_world,
                                                 delta_com_relative, delta_offset, delta_axis);
        result.mass[k] = evaluate_mass_direction(stage, model, delta_axis, delta_com, delta_origin,
                                                 delta_rotation);
    }

    // 广义速度方向 4..7：链不变，质量矩阵导数为零
    for (int k = 0; k < 4 && 4 + k < N; ++k) {
        std::array<double, 4> delta_velocity = zero_angles;
        delta_velocity[k] = 1.0;
        const KinematicsDirection kinematics =
            propagate_kinematics(stage, delta_velocity, zero_vectors_value, zero_vectors_value,
                                 zero_vectors_value);
        result.bias[4 + k] = evaluate_bias_direction(model, stage, kinematics, zero_matrices_value,
                                                     zero_vectors_value, zero_vectors_value,
                                                     zero_vectors_value);
    }

    if constexpr (N > kStateDirectionCount) {
        // 刚体质心方向 8..19
        for (int body = 0; body < 4; ++body) {
            for (int component = 0; component < 3; ++component) {
                const int direction = kStateDirectionCount + 3 * body + component;
                Vector3d unit = Vector3d::Zero();
                unit[component] = 1.0;
                const Vector3d delta_com_relative = chain.rotation[body] * unit;
                Array3 delta_com = zero_vectors_value;
                delta_com[body] = delta_com_relative;
                const KinematicsDirection kinematics = com_kinematics(stage, body, delta_com_relative);
                result.bias[direction] =
                    evaluate_bias_direction(model, stage, kinematics, zero_matrices_value, delta_com,
                                            zero_vectors_value, zero_vectors_value);
                result.mass[direction] =
                    evaluate_mass_direction_com(stage, model, body, delta_com_relative);
            }
        }
        // 关节安装偏置方向 20..31
        const Matrix3d base_rotation =
            chain.rotation[0] * local_rotation(0, stage.angle[0]).transpose();
        for (int joint = 0; joint < 4; ++joint) {
            const Matrix3d& parent_rotation =
                (joint == 0) ? base_rotation : chain.rotation[joint - 1];
            for (int component = 0; component < 3; ++component) {
                const int direction = 20 + 3 * joint + component;
                Vector3d unit = Vector3d::Zero();
                unit[component] = 1.0;
                const Vector3d delta_offset_value = parent_rotation * unit;
                Array3 delta_offset = zero_vectors_value;
                delta_offset[joint] = delta_offset_value;
                const KinematicsDirection kinematics =
                    offset_kinematics(stage, joint, delta_offset_value);
                result.bias[direction] =
                    evaluate_bias_direction(model, stage, kinematics, zero_matrices_value,
                                            zero_vectors_value, delta_offset, zero_vectors_value);
                result.mass[direction] =
                    evaluate_mass_direction_offset(stage, model, joint, delta_offset_value);
            }
        }
    }
    return result;
}

template <int N>
void fill_state_jacobian(const StageData& stage, const ScalarModelParameters<double>& model,
                         const Matrix4d& inverse_mass,
                         const std::array<Matrix4d, N>& mass_derivative,
                         const std::array<Vector4d, N>& bias_derivative,
                         Eigen::Matrix<double, 8, 8>& state_jacobian) {
    state_jacobian.setZero();
    state_jacobian.block<4, 4>(0, 4).setIdentity();
    for (int k = 0; k < 4; ++k) {
        state_jacobian.block<4, 1>(4, k) =
            inverse_mass * (-bias_derivative[k] - mass_derivative[k] * stage.acceleration);
        Vector4d damping_column = Vector4d::Zero();
        damping_column[k] = model.damping[k];
        state_jacobian.block<4, 1>(4, 4 + k) =
            inverse_mass * (-damping_column - bias_derivative[4 + k]);
    }
}

}  // namespace

StageJacobians evaluate_stage_jacobians(const StageData& stage,
                                        const ScalarModelParameters<double>& model) {
    const DirectionDerivatives<kStateDirectionCount> directional =
        evaluate_directions<kStateDirectionCount>(stage, model);
    const Matrix4d inverse_mass = invert_mass(stage.mass);

    StageJacobians result;
    fill_state_jacobian<kStateDirectionCount>(stage, model, inverse_mass, directional.mass,
                                              directional.bias, result.state);
    result.input.setZero();
    result.input.block<4, 4>(4, 0) = inverse_mass;
    return result;
}

ParameterJacobian evaluate_parameter_jacobian(const StageData& stage,
                                              const ScalarModelParameters<double>& model,
                                              Eigen::Matrix<double, 8, 8>& state_jacobian) {
    const DirectionDerivatives<kParameterDirectionCount> directional =
        evaluate_directions<kParameterDirectionCount>(stage, model);
    const Matrix4d inverse_mass = invert_mass(stage.mass);
    fill_state_jacobian<kParameterDirectionCount>(stage, model, inverse_mass, directional.mass,
                                                  directional.bias, state_jacobian);

    ParameterJacobian result;
    result.setZero();
    for (int index = 0; index < parameters::kParameterCount; ++index) {
        Matrix4d delta_mass = Matrix4d::Zero();
        Vector4d delta_bias = Vector4d::Zero();
        Vector4d delta_damping_production = Vector4d::Zero();
        if (index < parameters::kBodyCount * parameters::kBodyParameterCount) {
            const int body = index / parameters::kBodyParameterCount;
            const int slot = index % parameters::kBodyParameterCount;
            if (slot == 0) {
                mass_parameter_direction(stage, model, body, delta_mass, delta_bias);
            } else if (slot <= 3) {
                const int direction = kStateDirectionCount + 3 * body + (slot - 1);
                delta_mass = directional.mass[direction];
                delta_bias = directional.bias[direction];
            } else {
                inertia_parameter_direction(stage, body, slot - 4, delta_mass, delta_bias);
            }
        } else if (index < parameters::kBodyCount * parameters::kBodyParameterCount +
                               parameters::kOffsetParameterCount) {
            const int local = index - parameters::kBodyCount * parameters::kBodyParameterCount;
            const int direction = 20 + local;
            delta_mass = directional.mass[direction];
            delta_bias = directional.bias[direction];
        } else {
            const int joint = index - (parameters::kParameterCount -
                                       parameters::kDampingParameterCount);
            delta_damping_production[joint] = stage.qdot[joint];
        }
        result.block<4, 1>(4, index) = inverse_mass * (-delta_damping_production - delta_bias -
                                                       delta_mass * stage.acceleration);
    }
    return result;
}

}  // namespace detail
}  // namespace gimbal



