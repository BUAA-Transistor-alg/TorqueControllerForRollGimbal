#include "stage_jacobians.hpp"

#include <array>
#include <stdexcept>

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

using Vector3d = math::Vector3d;
using Matrix3d = math::Matrix3d;
using Vector4d = math::Vector4d;
using Matrix4d = math::Matrix4d;
using Matrix34d = math::Matrix34d;
using Array3 = std::array<Vector3d, 4>;
using ArrayMatrix = std::array<Matrix3d, 4>;

Matrix3d skew_matrix(const Vector3d& vector) {
    Matrix3d matrix;
    matrix.data[0][0] = 0.0;
    matrix.data[0][1] = -vector[2];
    matrix.data[0][2] = vector[1];
    matrix.data[1][0] = vector[2];
    matrix.data[1][1] = 0.0;
    matrix.data[1][2] = -vector[0];
    matrix.data[2][0] = -vector[1];
    matrix.data[2][1] = vector[0];
    matrix.data[2][2] = 0.0;
    return matrix;
}

Array3 zero_vectors() {
    Array3 result;
    for (int i = 0; i < 4; ++i) {
        result[i].setZero();
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

// ---- 方向维批处理：数据按 [刚体][分量][方向] 存放，方向维连续 ----

// out = a × v（a 为方向批次，v 为固定向量）
template <int N>
void cross_batch_vector(const Real a[3][N], const Vector3d& v, Real out[3][N]) {
    for (int d = 0; d < N; ++d) {
        out[0][d] = a[1][d] * v[2] - a[2][d] * v[1];
        out[1][d] = a[2][d] * v[0] - a[0][d] * v[2];
        out[2][d] = a[0][d] * v[1] - a[1][d] * v[0];
    }
}

// out = v × a（a 为方向批次，v 为固定向量）
template <int N>
void cross_vector_batch(const Vector3d& v, const Real a[3][N], Real out[3][N]) {
    for (int d = 0; d < N; ++d) {
        out[0][d] = v[1] * a[2][d] - v[2] * a[1][d];
        out[1][d] = v[2] * a[0][d] - v[0] * a[2][d];
        out[2][d] = v[0] * a[1][d] - v[1] * a[0][d];
    }
}

// out += a
template <int N>
void add_batch(Real out[3][N], const Real a[3][N]) {
    for (int c = 0; c < 3; ++c) {
        for (int d = 0; d < N; ++d) {
            out[c][d] += a[c][d];
        }
    }
}

// out = a + b + c（三项求和，减少临时量写法）
template <int N>
void sum3_batch(const Real a[3][N], const Real b[3][N], const Real c[3][N],
                Real out[3][N]) {
    for (int component = 0; component < 3; ++component) {
        for (int d = 0; d < N; ++d) {
            out[component][d] = a[component][d] + b[component][d] + c[component][d];
        }
    }
}

// out = v * row（外积）
template <int N>
void outer_batch(const Vector3d& v, const Real row[N], Real out[3][N]) {
    for (int component = 0; component < 3; ++component) {
        for (int d = 0; d < N; ++d) {
            out[component][d] = v[component] * row[d];
        }
    }
}

// out = s * a
template <int N>
void scale_batch(const Real a[3][N], Real scale, Real out[3][N]) {
    for (int component = 0; component < 3; ++component) {
        for (int d = 0; d < N; ++d) {
            out[component][d] = scale * a[component][d];
        }
    }
}

// 方向种子批次
template <int N>
struct SeedBatch {
    Real axis[4][3][N];
    Real offset[4][3][N];
    Real origin[4][3][N];
    Real com[4][3][N];
    Real com_relative[4][3][N];
    Real velocity[4][N];
    Real inertia_alpha[4][3][N];
    Real inertia_omega[4][3][N];
    Real inertia_world[4][4][3][3];  // [body][q 方向][3][3]
};

// 广义坐标方向（列 0..3）与惯量矩阵方向导数由构造过程完整写入，
// 这里只需要清零其余方向列。
template <int N>
void zero_seed_batch(SeedBatch<N>& seed) {
    for (int body = 0; body < 4; ++body) {
        for (int component = 0; component < 3; ++component) {
            for (int d = 4; d < N; ++d) {
                seed.axis[body][component][d] = 0.0;
                seed.offset[body][component][d] = 0.0;
                seed.origin[body][component][d] = 0.0;
                seed.com[body][component][d] = 0.0;
                seed.com_relative[body][component][d] = 0.0;
                seed.inertia_alpha[body][component][d] = 0.0;
                seed.inertia_omega[body][component][d] = 0.0;
            }
        }
        for (int d = 0; d < N; ++d) {
            seed.velocity[body][d] = 0.0;
        }
    }
}

// 构造方向种子批次
template <int N>
SeedBatch<N> build_seed_batch(const StageData& stage,
                              const ScalarModelParameters<Real>& model) {
    const Chain<Real>& chain = stage.chain;
    const BiasInternals<Real>& internals = stage.bias_internals;
    SeedBatch<N> seed;
    zero_seed_batch(seed);
    const Matrix3d base_rotation =
        chain.rotation[0] * transpose(local_rotation(0, stage.angle[0]));

    // 广义坐标方向 0..3
    for (int k = 0; k < 4; ++k) {
        const Matrix3d axis_skew = skew_matrix(axis_local<Real>(k));
        const Matrix3d rotation_derivative =
            chain.rotation[k] * axis_skew * transpose(chain.rotation[k]);
        ArrayMatrix delta_rotation = zero_matrices();
        for (int i = k; i < 4; ++i) {
            delta_rotation[i] = rotation_derivative * chain.rotation[i];
        }
        Array3 delta_offset = zero_vectors();
        for (int m = 1; m < 4; ++m) {
            if (k < m) {
                delta_offset[m] = delta_rotation[m - 1] * model.joint_offsets[m];
            }
        }
        Array3 delta_origin = zero_vectors();
        for (int i = 0; i < 4; ++i) {
            for (int m = 0; m <= i; ++m) {
                delta_origin[i] += delta_offset[m];
            }
        }
        Array3 delta_axis = zero_vectors();
        for (int j = 1; j < 4; ++j) {
            delta_axis[j] = delta_rotation[j - 1] * axis_local<Real>(j);
        }
        Array3 delta_com_relative = zero_vectors();
        Array3 delta_com = zero_vectors();
        for (int i = 0; i < 4; ++i) {
            delta_com_relative[i] = delta_rotation[i] * model.bodies[i].com;
            delta_com[i] = delta_origin[i] + delta_com_relative[i];
        }
        for (int i = 0; i < 4; ++i) {
            for (int component = 0; component < 3; ++component) {
                seed.axis[i][component][k] = delta_axis[i][component];
                seed.offset[i][component][k] = delta_offset[i][component];
                seed.origin[i][component][k] = delta_origin[i][component];
                seed.com[i][component][k] = delta_com[i][component];
                seed.com_relative[i][component][k] = delta_com_relative[i][component];
            }
            const Matrix3d delta_inertia_world =
                delta_rotation[i] * model.bodies[i].inertia *
                    transpose(chain.rotation[i]) +
                chain.rotation[i] * model.bodies[i].inertia *
                    transpose(delta_rotation[i]);
            for (int row = 0; row < 3; ++row) {
                for (int column = 0; column < 3; ++column) {
                    seed.inertia_world[i][k][row][column] = delta_inertia_world(row, column);
                }
                Real alpha = 0.0;
                Real omega = 0.0;
                for (int l = 0; l < 3; ++l) {
                    alpha += delta_inertia_world(row, l) * internals.alpha[i][l];
                    omega += delta_inertia_world(row, l) * internals.omega[i][l];
                }
                seed.inertia_alpha[i][row][k] = alpha;
                seed.inertia_omega[i][row][k] = omega;
            }
        }
    }

    // 广义速度方向 4..7
    for (int i = 0; i < 4 && 4 + i < N; ++i) {
        seed.velocity[i][4 + i] = 1.0;
    }

    if constexpr (N > kStateDirectionCount) {
        // 刚体质心方向 8..19
        for (int body = 0; body < 4; ++body) {
            for (int component = 0; component < 3; ++component) {
                const int direction = kStateDirectionCount + 3 * body + component;
                Vector3d unit = Vector3d::zero();
                unit[component] = 1.0;
                const Vector3d derivative = chain.rotation[body] * unit;
                for (int c = 0; c < 3; ++c) {
                    seed.com_relative[body][c][direction] = derivative[c];
                    seed.com[body][c][direction] = derivative[c];
                }
            }
        }
        // 关节安装偏置方向 20..31
        for (int joint = 0; joint < 4; ++joint) {
            const Matrix3d& parent_rotation =
                (joint == 0) ? base_rotation : chain.rotation[joint - 1];
            for (int component = 0; component < 3; ++component) {
                const int direction = 20 + 3 * joint + component;
                Vector3d unit = Vector3d::zero();
                unit[component] = 1.0;
                const Vector3d derivative = parent_rotation * unit;
                for (int c = 0; c < 3; ++c) {
                    seed.offset[joint][c][direction] = derivative[c];
                    for (int i = joint; i < 4; ++i) {
                        seed.origin[i][c][direction] = derivative[c];
                        seed.com[i][c][direction] = derivative[c];
                    }
                }
            }
        }
    }
    return seed;
}

// 各刚体角速度、角加速度、质心加速度的方向导数批次
template <int N>
struct KinematicsBatch {
    Real omega[4][3][N];
    Real alpha[4][3][N];
    Real com_acceleration[4][3][N];
};

// 全量运动学方向导数：逐方向独立计算，按方向维展开为连续的浮点运算
template <int N>
KinematicsBatch<N> propagate_kinematics(const StageData& stage, const SeedBatch<N>& seed) {
    const Chain<Real>& chain = stage.chain;
    const BiasInternals<Real>& internals = stage.bias_internals;
    KinematicsBatch<N> result;

    Vector3d previous_omega = chain.normal * stage.base.yaw_rate;
    Vector3d previous_alpha = chain.normal * stage.base.yaw_acceleration;
    Vector3d previous_origin_acceleration = stage.base.acceleration;
    Real delta_previous_omega[3][N] = {};
    Real delta_previous_alpha[3][N] = {};
    Real delta_previous_origin_acceleration[3][N] = {};
    Real temp1[3][N];
    Real temp2[3][N];
    Real temp3[3][N];

    for (int i = 0; i < 4; ++i) {
        const Vector3d& offset = chain.offset[i];
        const Vector3d& axis = chain.axis[i];
        const Real velocity = stage.qdot[i];

        // axis_direction = δa_i q̇_i + a_i δq̇_i
        Real axis_direction[3][N];
        for (int component = 0; component < 3; ++component) {
            for (int d = 0; d < N; ++d) {
                axis_direction[component][d] =
                    seed.axis[i][component][d] * velocity + axis[component] * seed.velocity[i][d];
            }
        }
        // δω_i = δω_{i-1} + axis_direction
        for (int component = 0; component < 3; ++component) {
            for (int d = 0; d < N; ++d) {
                result.omega[i][component][d] =
                    delta_previous_omega[component][d] + axis_direction[component][d];
            }
        }
        // δα_i = δα_{i-1} + δω_{i-1} × (a_i q̇_i) + ω_{i-1} × axis_direction
        cross_batch_vector(delta_previous_omega, axis * velocity, temp1);
        cross_vector_batch(previous_omega, axis_direction, temp2);
        sum3_batch(delta_previous_alpha, temp1, temp2, result.alpha[i]);

        // 基础量递推（固定向量）
        const Vector3d origin_acceleration =
            previous_origin_acceleration + cross_product(previous_alpha, offset) +
            cross_product(previous_omega, cross_product(previous_omega, offset));
        const Vector3d omega_cross_offset = cross_product(previous_omega, offset);

        // δa_p = δa_p_{i-1} + δα_{i-1} × r + α_{i-1} × δr
        //        + δω_{i-1} × (ω × r) + ω × (δω_{i-1} × r) + ω × (ω × δr)
        cross_batch_vector(delta_previous_alpha, offset, temp1);
        cross_vector_batch(previous_alpha, seed.offset[i], temp2);
        Real delta_origin_acceleration[3][N];
        sum3_batch(delta_previous_origin_acceleration, temp1, temp2, delta_origin_acceleration);
        cross_batch_vector(delta_previous_omega, omega_cross_offset, temp1);
        cross_batch_vector(delta_previous_omega, offset, temp2);
        cross_vector_batch(previous_omega, temp2, temp3);
        add_batch(delta_origin_acceleration, temp1);
        add_batch(delta_origin_acceleration, temp3);
        cross_vector_batch(previous_omega, seed.offset[i], temp2);
        cross_vector_batch(previous_omega, temp2, temp3);
        add_batch(delta_origin_acceleration, temp3);

        // δa_g = δa_p + δα_i × c + α_i × δc
        //        + δω_i × (ω_i × c) + ω_i × (δω_i × c) + ω_i × (ω_i × δc)
        const Vector3d com_relative = chain.com[i] - chain.joint_origin[i];
        const Vector3d omega_cross_com = cross_product(internals.omega[i], com_relative);
        cross_batch_vector(result.alpha[i], com_relative, temp1);
        cross_vector_batch(internals.alpha[i], seed.com_relative[i], temp2);
        sum3_batch(delta_origin_acceleration, temp1, temp2, result.com_acceleration[i]);
        cross_batch_vector(result.omega[i], omega_cross_com, temp1);
        cross_batch_vector(result.omega[i], com_relative, temp2);
        cross_vector_batch(internals.omega[i], temp2, temp3);
        add_batch(result.com_acceleration[i], temp1);
        add_batch(result.com_acceleration[i], temp3);
        cross_vector_batch(internals.omega[i], seed.com_relative[i], temp2);
        cross_vector_batch(internals.omega[i], temp2, temp3);
        add_batch(result.com_acceleration[i], temp3);

        previous_omega = internals.omega[i];
        previous_alpha = internals.alpha[i];
        previous_origin_acceleration = origin_acceleration;
        for (int component = 0; component < 3; ++component) {
            for (int d = 0; d < N; ++d) {
                delta_previous_omega[component][d] = result.omega[i][component][d];
                delta_previous_alpha[component][d] = result.alpha[i][component][d];
                delta_previous_origin_acceleration[component][d] =
                    delta_origin_acceleration[component][d];
            }
        }
    }
    return result;
}

// 偏置力方向导数的批处理：由运动学方向导数批次求各关节方向导数
template <int N>
void evaluate_bias_direction_batch(const ScalarModelParameters<Real>& model, const StageData& stage,
                                   const SeedBatch<N>& seed, const KinematicsBatch<N>& kinematics,
                                   Real delta_bias[4][N]) {
    const BiasInternals<Real>& internals = stage.bias_internals;
    Real delta_force[4][3][N];
    Real delta_couple[4][3][N];
    Real temp1[3][N];
    Real temp2[3][N];
    Real temp3[3][N];

    for (int i = 0; i < 4; ++i) {
        const Matrix3d& inertia_world = stage.mass_internals.inertia_world[i];
        const Real body_mass = model.bodies[i].mass;
        for (int component = 0; component < 3; ++component) {
            for (int d = 0; d < N; ++d) {
                delta_force[i][component][d] = body_mass * kinematics.com_acceleration[i][component][d];
            }
        }
        // δN = δI_w α + I_w δα + δω × (I_w ω) + ω × (δI_w ω + I_w δω)
        const Vector3d inertia_world_omega = inertia_world * internals.omega[i];
        cross_batch_vector(kinematics.omega[i], inertia_world_omega, temp1);
        Real inertia_alpha[3][N];
        Real inertia_omega[3][N];
        for (int component = 0; component < 3; ++component) {
            for (int d = 0; d < N; ++d) {
                Real alpha_value = 0.0;
                Real omega_value = 0.0;
                for (int l = 0; l < 3; ++l) {
                    alpha_value += inertia_world(component, l) * kinematics.alpha[i][l][d];
                    omega_value += inertia_world(component, l) * kinematics.omega[i][l][d];
                }
                inertia_alpha[component][d] = alpha_value;
                inertia_omega[component][d] = omega_value;
            }
        }
        for (int component = 0; component < 3; ++component) {
            for (int d = 0; d < N; ++d) {
                temp2[component][d] = seed.inertia_omega[i][component][d] + inertia_omega[component][d];
            }
        }
        cross_vector_batch(internals.omega[i], temp2, temp3);
        for (int component = 0; component < 3; ++component) {
            for (int d = 0; d < N; ++d) {
                delta_couple[i][component][d] = seed.inertia_alpha[i][component][d] +
                                                inertia_alpha[component][d] +
                                                temp1[component][d] + temp3[component][d];
            }
        }
    }

    Real child_force[3][N] = {};
    Real child_moment[3][N] = {};
    for (int i = 3; i >= 0; --i) {
        const Vector3d com_relative = stage.chain.com[i] - stage.chain.joint_origin[i];
        // δm = δN + δc × F + c × δF
        cross_batch_vector(seed.com_relative[i], internals.force[i], temp1);
        cross_vector_batch(com_relative, delta_force[i], temp2);
        Real moment[3][N];
        for (int component = 0; component < 3; ++component) {
            for (int d = 0; d < N; ++d) {
                moment[component][d] = delta_couple[i][component][d] + temp1[component][d] +
                                       temp2[component][d];
            }
        }
        if (i < 3) {
            cross_batch_vector(seed.offset[i + 1], internals.joint_force[i + 1], temp1);
            cross_vector_batch(stage.chain.offset[i + 1], child_force, temp2);
            for (int component = 0; component < 3; ++component) {
                for (int d = 0; d < N; ++d) {
                    moment[component][d] += child_moment[component][d] + temp1[component][d] +
                                            temp2[component][d];
                }
            }
        }
        const Vector3d& axis = stage.chain.axis[i];
        const Vector3d& joint_moment = internals.joint_moment[i];
        for (int d = 0; d < N; ++d) {
            Real value = 0.0;
            for (int component = 0; component < 3; ++component) {
                value += seed.axis[i][component][d] * joint_moment[component] +
                         axis[component] * moment[component][d];
            }
            delta_bias[i][d] = value;
        }
        for (int component = 0; component < 3; ++component) {
            for (int d = 0; d < N; ++d) {
                child_force[component][d] += delta_force[i][component][d];
                child_moment[component][d] = moment[component][d];
            }
        }
    }
}

// 从批次中取出第 direction 个方向的 3 维向量组
template <int N>
Array3 extract_vectors(const Real field[4][3][N], int direction) {
    Array3 result;
    for (int i = 0; i < 4; ++i) {
        result[i] = Vector3d::from(field[i][0][direction], field[i][1][direction],
                                   field[i][2][direction]);
    }
    return result;
}

template <int N>
Vector3d extract_vector(const Real field[3][N], int direction) {
    return Vector3d::from(field[0][direction], field[1][direction], field[2][direction]);
}

// 全量质量矩阵方向导数（q 方向）
Matrix4d evaluate_mass_direction(const StageData& stage, const ScalarModelParameters<Real>& model,
                                 const Array3& delta_axis, const Array3& delta_com,
                                 const Array3& delta_origin,
                                 const ArrayMatrix& delta_inertia_world) {
    Matrix4d mass_derivative = Matrix4d::zero();
    for (int i = 0; i < 4; ++i) {
        const Matrix34d& velocity_jacobian = stage.mass_internals.velocity_jacobian[i];
        const Matrix34d& angular_jacobian = stage.mass_internals.angular_jacobian[i];
        const Matrix3d& inertia_world = stage.mass_internals.inertia_world[i];
        Matrix34d velocity_jacobian_derivative = Matrix34d::zero();
        Matrix34d angular_jacobian_derivative = Matrix34d::zero();
        for (int joint = 0; joint <= i; ++joint) {
            const Vector3d column =
                cross_product(delta_axis[joint],
                              stage.chain.com[i] - stage.chain.joint_origin[joint]) +
                cross_product(stage.chain.axis[joint], delta_com[i] - delta_origin[joint]);
            for (int row = 0; row < 3; ++row) {
                velocity_jacobian_derivative(row, joint) = column[row];
                angular_jacobian_derivative(row, joint) = delta_axis[joint][row];
            }
        }
        mass_derivative += model.bodies[i].mass *
                           (transpose(velocity_jacobian_derivative) * velocity_jacobian +
                            transpose(velocity_jacobian) * velocity_jacobian_derivative);
        mass_derivative += transpose(angular_jacobian_derivative) * inertia_world * angular_jacobian;
        mass_derivative +=
            transpose(angular_jacobian) * delta_inertia_world[i] * angular_jacobian;
        mass_derivative += transpose(angular_jacobian) * inertia_world * angular_jacobian_derivative;
    }
    return mass_derivative;
}

// 后缀加权的速度雅可比和：suffix[joint] = Σ_{i ≥ joint} m_i Jv_i
std::array<Matrix34d, 4> evaluate_weighted_jacobian_sums(const StageData& stage,
                                                        const ScalarModelParameters<Real>& model) {
    std::array<Matrix34d, 4> suffix;
    for (int joint = 0; joint < 4; ++joint) {
        suffix[joint] = Matrix34d::zero();
    }
    for (int i = 3; i >= 0; --i) {
        const Matrix34d& velocity_jacobian = stage.mass_internals.velocity_jacobian[i];
        for (int joint = 0; joint <= i; ++joint) {
            suffix[joint] = suffix[joint] + velocity_jacobian * model.bodies[i].mass;
        }
    }
    return suffix;
}

// 刚体质心参数方向的质量矩阵方向导数：只有该刚体的雅可比列变化
Matrix4d evaluate_mass_direction_com(const StageData& stage,
                                     const ScalarModelParameters<Real>& model, int body,
                                     const Vector3d& delta_com_relative) {
    const Matrix34d& velocity_jacobian = stage.mass_internals.velocity_jacobian[body];
    const Real body_mass = model.bodies[body].mass;
    Vector3d axis_cross[4];
    for (int joint = 0; joint <= body; ++joint) {
        axis_cross[joint] = cross_product(stage.chain.axis[joint], delta_com_relative);
    }
    Matrix4d mass_derivative = Matrix4d::zero();
    for (int row = 0; row <= body; ++row) {
        for (int column = 0; column <= body; ++column) {
            Real value = 0.0;
            for (int component = 0; component < 3; ++component) {
                value += axis_cross[row][component] * velocity_jacobian(component, column) +
                         velocity_jacobian(component, row) * axis_cross[column][component];
            }
            mass_derivative(row, column) = body_mass * value;
        }
    }
    return mass_derivative;
}

// 关节安装偏置方向的质量矩阵方向导数：i ≥ joint 且 j < joint 的雅可比项变化
Matrix4d evaluate_mass_direction_offset(const StageData& stage, int joint,
                                        const Vector3d& delta_offset,
                                        const std::array<Matrix34d, 4>& weighted_sums) {
    const Matrix34d& weighted = weighted_sums[joint];
    Vector3d axis_cross[4];
    for (int smaller = 0; smaller < joint; ++smaller) {
        axis_cross[smaller] = cross_product(stage.chain.axis[smaller], delta_offset);
    }
    Matrix4d mass_derivative = Matrix4d::zero();
    for (int row = 0; row < 4; ++row) {
        for (int column = 0; column < 4; ++column) {
            Real value = 0.0;
            if (row < joint) {
                for (int component = 0; component < 3; ++component) {
                    value += axis_cross[row][component] * weighted(component, column);
                }
            }
            if (column < joint) {
                for (int component = 0; component < 3; ++component) {
                    value += weighted(component, row) * axis_cross[column][component];
                }
            }
            mass_derivative(row, column) = value;
        }
    }
    return mass_derivative;
}

// 28 个参数方向（质量 4 个、惯量 24 个）的质量矩阵与偏置力方向导数。
// 方向编号：0..3 为按刚体的质量，4 + 6*body + entry 为惯量的对称分量。
constexpr int kParameterBatchCount = 28;

void evaluate_parameter_directions(const StageData& stage,
                                   const ScalarModelParameters<Real>& model,
                                   Real mass_derivative[kParameterBatchCount][16],
                                   Real bias_derivative[kParameterBatchCount][4]) {
    static constexpr int kRow[6] = {0, 1, 2, 0, 0, 1};
    static constexpr int kColumn[6] = {0, 1, 2, 1, 2, 2};

    // 质量参数：dM/dm_i = Jv_i^T Jv_i
    for (int body = 0; body < 4; ++body) {
        const Matrix34d& velocity_jacobian = stage.mass_internals.velocity_jacobian[body];
        for (int row = 0; row < 4; ++row) {
            for (int column = 0; column < 4; ++column) {
                Real sum = 0.0;
                for (int component = 0; component < 3; ++component) {
                    sum += velocity_jacobian(component, row) *
                           velocity_jacobian(component, column);
                }
                mass_derivative[body][row * 4 + column] = sum;
            }
        }
    }
    // 惯量参数：dM/dE = A^T E A，A = R^T Jw
    for (int body = 0; body < 4; ++body) {
        const Matrix3d& rotation = stage.chain.rotation[body];
        const Matrix34d& angular_jacobian = stage.mass_internals.angular_jacobian[body];
        const Matrix34d local_jacobian = transpose(rotation) * angular_jacobian;
        for (int entry = 0; entry < 6; ++entry) {
            const int direction = 4 + 6 * body + entry;
            const int row_index = kRow[entry];
            const int column_index = kColumn[entry];
            for (int row = 0; row < 4; ++row) {
                for (int column = 0; column < 4; ++column) {
                    Real sum = local_jacobian(row_index, row) *
                                 local_jacobian(column_index, column);
                    if (row_index != column_index) {
                        sum += local_jacobian(column_index, row) *
                               local_jacobian(row_index, column);
                    }
                    mass_derivative[direction][row * 4 + column] = sum;
                }
            }
        }
    }

    // 偏置力：质量参数给力，惯量参数给力偶
    Real delta_force[4][3][kParameterBatchCount] = {};
    Real delta_couple[4][3][kParameterBatchCount] = {};
    for (int body = 0; body < 4; ++body) {
        for (int component = 0; component < 3; ++component) {
            delta_force[body][component][body] =
                stage.bias_internals.com_acceleration[body][component];
        }
        delta_force[body][2][body] += model.gravity;
        const Matrix3d& rotation = stage.chain.rotation[body];
        const Vector3d& omega = stage.bias_internals.omega[body];
        const Vector3d& alpha = stage.bias_internals.alpha[body];
        for (int entry = 0; entry < 6; ++entry) {
            const int direction = 4 + 6 * body + entry;
            Matrix3d inertia_direction = Matrix3d::zero();
            inertia_direction(kRow[entry], kColumn[entry]) += 1.0;
            if (kRow[entry] != kColumn[entry]) {
                inertia_direction(kColumn[entry], kRow[entry]) += 1.0;
            }
            const Matrix3d inertia_world_direction =
                rotation * inertia_direction * transpose(rotation);
            const Vector3d couple = inertia_world_direction * alpha +
                                    cross_product(omega, inertia_world_direction * omega);
            for (int component = 0; component < 3; ++component) {
                delta_couple[body][component][direction] = couple[component];
            }
        }
    }

    // 反向递推：δm_i = δN_i + c_i × δF_i + δm_{i+1} + r_{i+1} × δf_{i+1}，
    //           δb_i = a_i · δm_i
    Real child_force[3][kParameterBatchCount] = {};
    Real child_moment[3][kParameterBatchCount] = {};
    for (int i = 3; i >= 0; --i) {
        const Vector3d& axis = stage.chain.axis[i];
        const Vector3d com_relative = stage.chain.com[i] - stage.chain.joint_origin[i];
        for (int direction = 0; direction < kParameterBatchCount; ++direction) {
            Real moment[3];
            for (int component = 0; component < 3; ++component) {
                moment[component] = delta_couple[i][component][direction];
            }
            moment[0] += com_relative[1] * delta_force[i][2][direction] -
                         com_relative[2] * delta_force[i][1][direction];
            moment[1] += com_relative[2] * delta_force[i][0][direction] -
                         com_relative[0] * delta_force[i][2][direction];
            moment[2] += com_relative[0] * delta_force[i][1][direction] -
                         com_relative[1] * delta_force[i][0][direction];
            if (i < 3) {
                const Vector3d& offset = stage.chain.offset[i + 1];
                moment[0] += child_moment[0][direction] + offset[1] * child_force[2][direction] -
                             offset[2] * child_force[1][direction];
                moment[1] += child_moment[1][direction] + offset[2] * child_force[0][direction] -
                             offset[0] * child_force[2][direction];
                moment[2] += child_moment[2][direction] + offset[0] * child_force[1][direction] -
                             offset[1] * child_force[0][direction];
            }
            Real value = 0.0;
            for (int component = 0; component < 3; ++component) {
                value += axis[component] * moment[component];
            }
            bias_derivative[direction][i] = value;
            for (int component = 0; component < 3; ++component) {
                child_force[component][direction] += delta_force[i][component][direction];
                child_moment[component][direction] = moment[component];
            }
        }
    }
}

// 复用阶段数据里已有的 Cholesky 分解求逆矩阵的四列
Matrix4d invert_mass(const StageData& stage) {
    Matrix4d inverse = Matrix4d::zero();
    for (int column = 0; column < 4; ++column) {
        Vector4d unit = Vector4d::zero();
        unit[column] = 1.0;
        const Vector4d solution = stage.factor.solve(unit);
        for (int row = 0; row < 4; ++row) {
            inverse(row, column) = solution[row];
        }
    }
    return inverse;
}

template <int N>
struct DirectionDerivatives {
    std::array<Matrix4d, N> mass{};
    std::array<Vector4d, N> bias{};
};

// 由批次取出第 direction 个方向的质量矩阵方向导数
template <int N>
Matrix4d evaluate_direction_mass(const StageData& stage, const ScalarModelParameters<Real>& model,
                                 const SeedBatch<N>& seed,
                                 const std::array<Matrix34d, 4>& weighted_sums, int direction) {
    if (direction < 4) {
        const Array3 delta_axis = extract_vectors(seed.axis, direction);
        const Array3 delta_com = extract_vectors(seed.com, direction);
        const Array3 delta_origin = extract_vectors(seed.origin, direction);
        ArrayMatrix delta_inertia_world = zero_matrices();
        for (int i = 0; i < 4; ++i) {
            for (int row = 0; row < 3; ++row) {
                for (int column = 0; column < 3; ++column) {
                    delta_inertia_world[i](row, column) =
                        seed.inertia_world[i][direction][row][column];
                }
            }
        }
        return evaluate_mass_direction(stage, model, delta_axis, delta_com, delta_origin,
                                       delta_inertia_world);
    }
    if (direction < kStateDirectionCount) {
        return Matrix4d::zero();
    }
    if (direction < 20) {
        const int body = (direction - kStateDirectionCount) / 3;
        const Vector3d delta_com_relative = extract_vector(seed.com_relative[body], direction);
        return evaluate_mass_direction_com(stage, model, body, delta_com_relative);
    }
    const int joint = (direction - 20) / 3;
    const Vector3d delta_offset = extract_vector(seed.offset[joint], direction);
    return evaluate_mass_direction_offset(stage, joint, delta_offset, weighted_sums);
}

// 批量求全部方向的质量矩阵与偏置力方向导数
template <int N>
DirectionDerivatives<N> evaluate_directions(const StageData& stage,
                                            const ScalarModelParameters<Real>& model) {
    const SeedBatch<N> seed = build_seed_batch<N>(stage, model);
    const KinematicsBatch<N> kinematics = propagate_kinematics<N>(stage, seed);
    Real delta_bias[4][N];
    evaluate_bias_direction_batch<N>(model, stage, seed, kinematics, delta_bias);
    const std::array<Matrix34d, 4> weighted_sums = evaluate_weighted_jacobian_sums(stage, model);

    DirectionDerivatives<N> result;
    for (int direction = 0; direction < N; ++direction) {
        result.bias[direction] =
            Vector4d::from(delta_bias[0][direction], delta_bias[1][direction],
                           delta_bias[2][direction], delta_bias[3][direction]);
        result.mass[direction] =
            evaluate_direction_mass<N>(stage, model, seed, weighted_sums, direction);
    }
    return result;
}

template <int N>
void fill_state_jacobian(const StageData& stage, const ScalarModelParameters<Real>& model,
                         const Matrix4d& inverse_mass,
                         const std::array<Matrix4d, N>& mass_derivative,
                         const std::array<Vector4d, N>& bias_derivative,
                         math::Matrix8d& state_jacobian) {
    state_jacobian.setZero();
    for (int i = 0; i < 4; ++i) {
        state_jacobian(i, 4 + i) = 1.0;
    }
    for (int k = 0; k < 4; ++k) {
        const Vector4d acceleration_column =
            inverse_mass * (-bias_derivative[k] - mass_derivative[k] * stage.acceleration);
        Vector4d damping_column = Vector4d::zero();
        damping_column[k] = model.damping[k];
        const Vector4d velocity_column =
            inverse_mass * (-damping_column - bias_derivative[4 + k]);
        for (int row = 0; row < 4; ++row) {
            state_jacobian(4 + row, k) = acceleration_column[row];
            state_jacobian(4 + row, 4 + k) = velocity_column[row];
        }
    }
}

}  // namespace

StageJacobians evaluate_stage_jacobians(const StageData& stage,
                                        const ScalarModelParameters<Real>& model) {
    const DirectionDerivatives<kStateDirectionCount> directional =
        evaluate_directions<kStateDirectionCount>(stage, model);
    const Matrix4d inverse_mass = invert_mass(stage);

    StageJacobians result;
    fill_state_jacobian<kStateDirectionCount>(stage, model, inverse_mass, directional.mass,
                                              directional.bias, result.state);
    result.input.setZero();
    for (int row = 0; row < 4; ++row) {
        for (int column = 0; column < 4; ++column) {
            result.input(4 + row, column) = inverse_mass(row, column);
        }
    }
    return result;
}

ParameterJacobian evaluate_parameter_jacobian(const StageData& stage,
                                              const ScalarModelParameters<Real>& model,
                                              math::Matrix8d& state_jacobian) {
    const DirectionDerivatives<kParameterDirectionCount> directional =
        evaluate_directions<kParameterDirectionCount>(stage, model);
    const Matrix4d inverse_mass = invert_mass(stage);
    fill_state_jacobian<kParameterDirectionCount>(stage, model, inverse_mass, directional.mass,
                                                  directional.bias, state_jacobian);

    Real parameter_mass[kParameterBatchCount][16];
    Real parameter_bias[kParameterBatchCount][4];
    evaluate_parameter_directions(stage, model, parameter_mass, parameter_bias);

    ParameterJacobian result;
    result.setZero();
    for (int index = 0; index < parameters::kParameterCount; ++index) {
        Matrix4d delta_mass = Matrix4d::zero();
        Vector4d delta_bias = Vector4d::zero();
        Vector4d delta_damping_production = Vector4d::zero();
        int parameter_direction = -1;
        if (index < parameters::kBodyCount * parameters::kBodyParameterCount) {
            const int body = index / parameters::kBodyParameterCount;
            const int slot = index % parameters::kBodyParameterCount;
            if (slot == 0) {
                parameter_direction = body;
            } else if (slot <= 3) {
                const int direction = kStateDirectionCount + 3 * body + (slot - 1);
                delta_mass = directional.mass[direction];
                delta_bias = directional.bias[direction];
            } else {
                parameter_direction = 4 + 6 * body + (slot - 4);
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
        if (parameter_direction >= 0) {
            for (int row = 0; row < 4; ++row) {
                for (int column = 0; column < 4; ++column) {
                    delta_mass(row, column) =
                        parameter_mass[parameter_direction][row * 4 + column];
                }
                delta_bias[row] = parameter_bias[parameter_direction][row];
            }
        }
        const Vector4d column = inverse_mass * (-delta_damping_production - delta_bias -
                                                delta_mass * stage.acceleration);
        for (int row = 0; row < 4; ++row) {
            result(4 + row, index) = column[row];
        }
    }
    return result;
}

}  // namespace detail
}  // namespace gimbal




