#include "gimbal_dynamics/dynamics_model.hpp"

#include <array>
#include <stdexcept>

#include "model_evaluation.hpp"

namespace gimbal {
namespace {

template <typename Scalar>
using Vector3 = detail::Vector3<Scalar>;
template <typename Scalar>
using Matrix3 = detail::Matrix3<Scalar>;
template <typename Scalar>
using Vector4 = detail::Vector4<Scalar>;
template <typename Scalar>
using Matrix4 = detail::Matrix4<Scalar>;
template <typename Scalar>
using Matrix34 = detail::Matrix34<Scalar>;

using Vector3d = Vector3<double>;
using Matrix3d = Matrix3<double>;
using Vector4d = Vector4<double>;
using Matrix4d = Matrix4<double>;
using Matrix34d = Matrix34<double>;

math::Vector3d vec3(double x, double y, double z) {
    math::Vector3d value;
    value[0] = x;
    value[1] = y;
    value[2] = z;
    return value;
}

math::Matrix3d mat3(double xx, double xy, double xz, double yx, double yy, double yz, double zx,
                    double zy, double zz) {
    math::Matrix3d value;
    value(0, 0) = xx;
    value(0, 1) = xy;
    value(0, 2) = xz;
    value(1, 0) = yx;
    value(1, 1) = yy;
    value(1, 2) = yz;
    value(2, 0) = zx;
    value(2, 1) = zy;
    value(2, 2) = zz;
    return value;
}

// M、M_θψ、M_ψψ 与 J_g（DYNAMICS.md 7.2）
template <typename Scalar>
struct DynamicsTerms {
    Matrix4<Scalar> mass = Matrix4<Scalar>::zero();
    Vector4<Scalar> coupling = Vector4<Scalar>::zero();
    Scalar base_inertia = Scalar(0);
    Matrix34<Scalar> momentum_jacobian = Matrix34<Scalar>::zero();
};

template <typename Scalar>
DynamicsTerms<Scalar> evaluate_terms(const detail::ScalarModelParameters<Scalar>& model,
                                     const detail::Chain<Scalar>& chain) {
    const Vector3<Scalar>& normal = chain.normal;
    DynamicsTerms<Scalar> terms;
    for (int i = 0; i < 4; ++i) {
        const Scalar body_mass = model.bodies[i].mass;
        const Matrix3<Scalar> inertia_world =
            chain.rotation[i] * model.bodies[i].inertia * transpose(chain.rotation[i]);
        Matrix34<Scalar> velocity_jacobian = Matrix34<Scalar>::zero();
        Matrix34<Scalar> angular_jacobian = Matrix34<Scalar>::zero();
        for (int joint = 0; joint <= i; ++joint) {
            const Vector3<Scalar> column = detail::cross_product(
                chain.axis[joint], chain.com[i] - chain.joint_origin[joint]);
            for (int row = 0; row < 3; ++row) {
                velocity_jacobian(row, joint) = column[row];
                angular_jacobian(row, joint) = chain.axis[joint][row];
            }
        }
        terms.mass += body_mass * (transpose(velocity_jacobian) * velocity_jacobian);
        terms.mass += transpose(angular_jacobian) * inertia_world * angular_jacobian;
        terms.coupling +=
            body_mass * (transpose(velocity_jacobian) * detail::cross_product(normal, chain.com[i]));
        terms.coupling += transpose(angular_jacobian) * inertia_world * normal;
        const Vector3<Scalar> normal_cross_com = detail::cross_product(normal, chain.com[i]);
        terms.base_inertia += body_mass * detail::dot_product(normal_cross_com, normal_cross_com) +
                              detail::dot_product(normal, inertia_world * normal);
        terms.momentum_jacobian += velocity_jacobian * body_mass;
    }
    return terms;
}

// 质量矩阵各量对广义坐标的偏导，用链的方向导数解析求值
struct Derivatives {
    DynamicsTerms<double> terms;
    std::array<Matrix4d, 4> mass_derivative{};
    std::array<Vector4d, 4> coupling_derivative{};
    Vector4d base_inertia_derivative = Vector4d::zero();
};

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

Derivatives evaluate_derivatives(const ModelParameters& model, const Terrain& terrain,
                                 const BaseState& base, const Vector4d& q) {
    const detail::ScalarModelParameters<double> scalar_model = detail::convert_model<double>(model);
    const detail::ScalarTerrain<double> scalar_terrain = detail::convert_terrain<double>(terrain);
    const detail::Chain<double>& chain =
        detail::evaluate_chain(scalar_model, scalar_terrain, base, q);

    Derivatives result;
    result.terms = evaluate_terms<double>(scalar_model, chain);
    const Vector3d& normal = chain.normal;

    // 各刚体的速度/角速度雅可比与惯量世界矩阵
    Matrix34d velocity_jacobian[4];
    Matrix34d angular_jacobian[4];
    Matrix3d inertia_world[4];
    for (int i = 0; i < 4; ++i) {
        velocity_jacobian[i] = Matrix34d::zero();
        angular_jacobian[i] = Matrix34d::zero();
        for (int joint = 0; joint <= i; ++joint) {
            const Vector3d column = detail::cross_product(
                chain.axis[joint], chain.com[i] - chain.joint_origin[joint]);
            for (int row = 0; row < 3; ++row) {
                velocity_jacobian[i](row, joint) = column[row];
                angular_jacobian[i](row, joint) = chain.axis[joint][row];
            }
        }
        inertia_world[i] =
            chain.rotation[i] * scalar_model.bodies[i].inertia * transpose(chain.rotation[i]);
    }

    for (int k = 0; k < 4; ++k) {
        const Matrix3d axis_skew = skew_matrix(detail::axis_local<double>(k));
        Matrix3d delta_rotation[4];
        for (int i = 0; i < 4; ++i) {
            delta_rotation[i] = Matrix3d::zero();
        }
        const Matrix3d rotation_derivative =
            chain.rotation[k] * axis_skew * transpose(chain.rotation[k]);
        for (int i = k; i < 4; ++i) {
            delta_rotation[i] = rotation_derivative * chain.rotation[i];
        }
        Vector3d delta_offset[4];
        Vector3d delta_origin[4];
        for (int i = 0; i < 4; ++i) {
            delta_offset[i] = Vector3d::zero();
            delta_origin[i] = Vector3d::zero();
        }
        for (int m = 1; m < 4; ++m) {
            if (k < m) {
                delta_offset[m] = delta_rotation[m - 1] * scalar_model.joint_offsets[m];
            }
        }
        for (int i = 0; i < 4; ++i) {
            for (int m = 0; m <= i; ++m) {
                delta_origin[i] = delta_origin[i] + delta_offset[m];
            }
        }
        Vector3d delta_axis[4];
        for (int j = 0; j < 4; ++j) {
            delta_axis[j] = Vector3d::zero();
        }
        for (int j = 1; j < 4; ++j) {
            delta_axis[j] = delta_rotation[j - 1] * detail::axis_local<double>(j);
        }
        Vector3d delta_com[4];
        for (int i = 0; i < 4; ++i) {
            delta_com[i] = delta_origin[i] + delta_rotation[i] * scalar_model.bodies[i].com;
        }

        Matrix4d delta_mass = Matrix4d::zero();
        Vector4d delta_coupling = Vector4d::zero();
        double delta_base_inertia = 0.0;
        for (int i = 0; i < 4; ++i) {
            Matrix34d delta_velocity_jacobian = Matrix34d::zero();
            Matrix34d delta_angular_jacobian = Matrix34d::zero();
            for (int joint = 0; joint <= i; ++joint) {
                const Vector3d column =
                    detail::cross_product(delta_axis[joint],
                                          chain.com[i] - chain.joint_origin[joint]) +
                    detail::cross_product(chain.axis[joint], delta_com[i] - delta_origin[joint]);
                for (int row = 0; row < 3; ++row) {
                    delta_velocity_jacobian(row, joint) = column[row];
                    delta_angular_jacobian(row, joint) = delta_axis[joint][row];
                }
            }
            const Matrix3d delta_inertia_world =
                delta_rotation[i] * scalar_model.bodies[i].inertia * transpose(chain.rotation[i]) +
                chain.rotation[i] * scalar_model.bodies[i].inertia * transpose(delta_rotation[i]);
            const Vector3d normal_cross_com = detail::cross_product(normal, chain.com[i]);
            const Vector3d normal_cross_delta_com = detail::cross_product(normal, delta_com[i]);
            delta_mass = delta_mass + scalar_model.bodies[i].mass *
                                          (transpose(delta_velocity_jacobian) * velocity_jacobian[i] +
                                           transpose(velocity_jacobian[i]) * delta_velocity_jacobian);
            delta_mass = delta_mass +
                         transpose(delta_angular_jacobian) * inertia_world[i] * angular_jacobian[i] +
                         transpose(angular_jacobian[i]) * delta_inertia_world * angular_jacobian[i] +
                         transpose(angular_jacobian[i]) * inertia_world[i] * delta_angular_jacobian;
            delta_coupling = delta_coupling +
                             scalar_model.bodies[i].mass *
                                 (transpose(delta_velocity_jacobian) * normal_cross_com +
                                  transpose(velocity_jacobian[i]) * normal_cross_delta_com) +
                             transpose(delta_angular_jacobian) * inertia_world[i] * normal +
                             transpose(angular_jacobian[i]) * delta_inertia_world * normal;
            delta_base_inertia +=
                2.0 * scalar_model.bodies[i].mass *
                    detail::dot_product(normal_cross_com, normal_cross_delta_com) +
                detail::dot_product(normal, delta_inertia_world * normal);
        }
        result.mass_derivative[k] = delta_mass;
        result.coupling_derivative[k] = delta_coupling;
        result.base_inertia_derivative[k] = delta_base_inertia;
    }
    return result;
}

// 偏置力 b 与质量矩阵（DYNAMICS.md 7.3）
struct BiasResult {
    DynamicsTerms<double> terms;
    Vector4d bias;
};

BiasResult evaluate_bias(const ModelParameters& model, const Terrain& terrain, const BaseState& base,
                         const Vector4d& q, const Vector4d& qdot) {
    const Derivatives derivatives = evaluate_derivatives(model, terrain, base, q);
    const DynamicsTerms<double>& terms = derivatives.terms;
    Vector4d christoffel = Vector4d::zero();
    for (int j = 0; j < 4; ++j) {
        for (int k = 0; k < 4; ++k) {
            for (int l = 0; l < 4; ++l) {
                christoffel[j] +=
                    (derivatives.mass_derivative[k](j, l) -
                     0.5 * derivatives.mass_derivative[j](k, l)) * qdot[k] * qdot[l];
            }
        }
    }
    Vector4d yaw_coupling = Vector4d::zero();
    for (int j = 0; j < 4; ++j) {
        for (int k = 0; k < 4; ++k) {
            yaw_coupling[j] +=
                (derivatives.coupling_derivative[k][j] -
                 derivatives.coupling_derivative[j][k]) * qdot[k];
        }
    }
    yaw_coupling = yaw_coupling * base.yaw_rate;
    const Vector3d gravity_vector = vec3(0.0, 0.0, -model.gravity);
    const Vector4d bias =
        -(transpose(terms.momentum_jacobian) * gravity_vector) +
        transpose(terms.momentum_jacobian) * base.acceleration +
        terms.coupling * base.yaw_acceleration -
        derivatives.base_inertia_derivative * (base.yaw_rate * base.yaw_rate) * 0.5 + christoffel +
        yaw_coupling;
    return {terms, bias};
}

}  // namespace

ModelParameters ModelParameters::demo() {
    ModelParameters model;
    model.bodies = {{
        BodyParameters{0.90, vec3(0.004, 0.002, 0.018),
                       mat3(0.00140, 0.00002, 0.00003,
                            0.00002, 0.00155, 0.00001,
                            0.00003, 0.00001, 0.00110)},
        BodyParameters{0.42, vec3(0.012, 0.003, 0.032),
                       mat3(0.00028, 0.00001, 0.00002,
                            0.00001, 0.00033, 0.00001,
                            0.00002, 0.00001, 0.00021)},
        BodyParameters{0.25, vec3(-0.004, -0.003, 0.022),
                       mat3(0.00011, 0.000005, 0.000004,
                            0.000005, 0.00014, 0.000003,
                            0.000004, 0.000003, 0.00009)},
        BodyParameters{0.55, vec3(0.008, 0.001, 0.026),
                       mat3(0.00034, 0.00002, 0.00001,
                            0.00002, 0.00039, 0.000015,
                            0.00001, 0.000015, 0.00026)},
    }};
    model.joint_offsets = {{
        vec3(0.0, 0.0, 0.045),
        vec3(0.028, 0.0, 0.095),
        vec3(0.0, 0.0, 0.072),
        vec3(0.0, 0.0, 0.058),
    }};
    model.damping[0] = 0.020;
    model.damping[1] = 0.015;
    model.damping[2] = 0.010;
    model.damping[3] = 0.008;
    model.gravity = 9.81;
    return model;
}

math::Vector4d forward_dynamics(const ModelParameters& model, const Terrain& terrain,
                                const BaseState& base, const math::Vector4d& q,
                                const math::Vector4d& qdot, const math::Vector4d& torque) {
    const BiasResult bias = evaluate_bias(model, terrain, base, q, qdot);
    const Vector4d right_hand_side = torque - elementwise(model.damping, qdot) - bias.bias;
    return detail::solve_mass(bias.terms.mass, right_hand_side);
}

}  // namespace gimbal
