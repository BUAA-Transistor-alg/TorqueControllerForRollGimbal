#include "gimbal_dynamics/dynamics_model.hpp"

#include <array>
#include <complex>
#include <stdexcept>

#include <Eigen/Cholesky>

#include "model_evaluation.hpp"

namespace gimbal {
namespace {

using Complex = std::complex<double>;

template <typename Scalar>
using Vector3 = detail::Vector3<Scalar>;
template <typename Scalar>
using Matrix3 = detail::Matrix3<Scalar>;
template <typename Scalar>
using Vector4 = detail::Vector4<Scalar>;
template <typename Scalar>
using Matrix4 = detail::Matrix4<Scalar>;

using Vector3d = Vector3<double>;
using Matrix3d = Matrix3<double>;
using Vector4d = Vector4<double>;
using Matrix4d = Matrix4<double>;

// M、M_θψ、M_ψψ 与 J_g（DYNAMICS.md 7.2）
template <typename Scalar>
struct DynamicsTerms {
    Matrix4<Scalar> mass = Matrix4<Scalar>::Zero();
    Vector4<Scalar> coupling = Vector4<Scalar>::Zero();
    Scalar base_inertia = Scalar(0);
    Eigen::Matrix<Scalar, 3, 4> momentum_jacobian = Eigen::Matrix<Scalar, 3, 4>::Zero();
};

template <typename Scalar>
DynamicsTerms<Scalar> evaluate_terms(const detail::ScalarModelParameters<Scalar>& model,
                                     const detail::Chain<Scalar>& chain) {
    const Vector3<Scalar>& normal = chain.normal;
    DynamicsTerms<Scalar> terms;
    for (int i = 0; i < 4; ++i) {
        const Scalar body_mass = model.bodies[i].mass;
        const Matrix3<Scalar> inertia_world =
            chain.rotation[i] * model.bodies[i].inertia * chain.rotation[i].transpose();
        Eigen::Matrix<Scalar, 3, 4> velocity_jacobian = Eigen::Matrix<Scalar, 3, 4>::Zero();
        Eigen::Matrix<Scalar, 3, 4> angular_jacobian = Eigen::Matrix<Scalar, 3, 4>::Zero();
        for (int joint = 0; joint <= i; ++joint) {
            velocity_jacobian.col(joint) =
                detail::cross_product(chain.axis[joint], chain.com[i] - chain.joint_origin[joint]);
            angular_jacobian.col(joint) = chain.axis[joint];
        }
        terms.mass += body_mass * velocity_jacobian.transpose() * velocity_jacobian;
        terms.mass += angular_jacobian.transpose() * inertia_world * angular_jacobian;
        terms.coupling += body_mass * velocity_jacobian.transpose() *
                          detail::cross_product(normal, chain.com[i]);
        terms.coupling += angular_jacobian.transpose() * inertia_world * normal;
        const Vector3<Scalar> normal_cross_com = detail::cross_product(normal, chain.com[i]);
        terms.base_inertia += body_mass * detail::dot_product(normal_cross_com, normal_cross_com) +
                              detail::dot_product(normal, inertia_world * normal);
        terms.momentum_jacobian += body_mass * velocity_jacobian;
    }
    return terms;
}

// 质量矩阵各量对广义坐标的偏导，复步长精确求值
struct Derivatives {
    DynamicsTerms<double> terms;
    std::array<Matrix4d, 4> mass_derivative{};
    std::array<Vector4d, 4> coupling_derivative{};
    Vector4d base_inertia_derivative = Vector4d::Zero();
};

Derivatives evaluate_derivatives(const ModelParameters& model, const Terrain& terrain,
                                 const BaseState& base, const Vector4d& q) {
    const detail::ScalarModelParameters<double> scalar_model = detail::convert_model<double>(model);
    const detail::ScalarTerrain<double> scalar_terrain = detail::convert_terrain<double>(terrain);
    const detail::ScalarModelParameters<Complex> complex_model = detail::convert_model<Complex>(model);
    const detail::ScalarTerrain<Complex> complex_terrain = detail::convert_terrain<Complex>(terrain);

    Derivatives result;
    result.terms = evaluate_terms<double>(
        scalar_model, detail::evaluate_chain(scalar_model, scalar_terrain, base, q));
    for (int k = 0; k < 4; ++k) {
        detail::Vector4<Complex> perturbed;
        for (int i = 0; i < 4; ++i) {
            perturbed[i] = Complex(q[i], i == k ? detail::kComplexStep : 0.0);
        }
        const DynamicsTerms<Complex> terms = evaluate_terms<Complex>(
            complex_model, detail::evaluate_chain(complex_model, complex_terrain, base, perturbed));
        for (int row = 0; row < 4; ++row) {
            for (int column = 0; column < 4; ++column) {
                result.mass_derivative[k](row, column) =
                    terms.mass(row, column).imag() / detail::kComplexStep;
            }
            result.coupling_derivative[k][row] = terms.coupling[row].imag() / detail::kComplexStep;
        }
        result.base_inertia_derivative[k] = terms.base_inertia.imag() / detail::kComplexStep;
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
    Vector4d christoffel = Vector4d::Zero();
    for (int j = 0; j < 4; ++j) {
        for (int k = 0; k < 4; ++k) {
            for (int l = 0; l < 4; ++l) {
                christoffel[j] +=
                    (derivatives.mass_derivative[k](j, l) -
                     0.5 * derivatives.mass_derivative[j](k, l)) * qdot[k] * qdot[l];
            }
        }
    }
    Vector4d yaw_coupling = Vector4d::Zero();
    for (int j = 0; j < 4; ++j) {
        for (int k = 0; k < 4; ++k) {
            yaw_coupling[j] +=
                (derivatives.coupling_derivative[k][j] -
                 derivatives.coupling_derivative[j][k]) * qdot[k];
        }
    }
    yaw_coupling *= base.yaw_rate;
    const Vector3d gravity_vector(0.0, 0.0, -model.gravity);
    const Vector4d bias = -terms.momentum_jacobian.transpose() * gravity_vector +
                          terms.momentum_jacobian.transpose() * base.acceleration +
                          terms.coupling * base.yaw_acceleration -
                          0.5 * derivatives.base_inertia_derivative *
                              (base.yaw_rate * base.yaw_rate) +
                          christoffel + yaw_coupling;
    return {terms, bias};
}

}  // namespace

ModelParameters ModelParameters::demo() {
    ModelParameters model;
    model.bodies = {{
        BodyParameters{0.90, Eigen::Vector3d(0.004, 0.002, 0.018),
                       (Eigen::Matrix3d() << 0.00140, 0.00002, 0.00003,
                                             0.00002, 0.00155, 0.00001,
                                             0.00003, 0.00001, 0.00110).finished()},
        BodyParameters{0.42, Eigen::Vector3d(0.012, 0.003, 0.032),
                       (Eigen::Matrix3d() << 0.00028, 0.00001, 0.00002,
                                             0.00001, 0.00033, 0.00001,
                                             0.00002, 0.00001, 0.00021).finished()},
        BodyParameters{0.25, Eigen::Vector3d(-0.004, -0.003, 0.022),
                       (Eigen::Matrix3d() << 0.00011, 0.000005, 0.000004,
                                             0.000005, 0.00014, 0.000003,
                                             0.000004, 0.000003, 0.00009).finished()},
        BodyParameters{0.55, Eigen::Vector3d(0.008, 0.001, 0.026),
                       (Eigen::Matrix3d() << 0.00034, 0.00002, 0.00001,
                                             0.00002, 0.00039, 0.000015,
                                             0.00001, 0.000015, 0.00026).finished()},
    }};
    model.joint_offsets = {{
        Eigen::Vector3d(0.0, 0.0, 0.045),
        Eigen::Vector3d(0.028, 0.0, 0.095),
        Eigen::Vector3d(0.0, 0.0, 0.072),
        Eigen::Vector3d(0.0, 0.0, 0.058),
    }};
    model.damping = Eigen::Vector4d(0.020, 0.015, 0.010, 0.008);
    model.gravity = 9.81;
    return model;
}

Eigen::Vector4d forward_dynamics(const ModelParameters& model, const Terrain& terrain,
                                 const BaseState& base, const Eigen::Vector4d& q,
                                 const Eigen::Vector4d& qdot,
                                 const Eigen::Vector4d& torque) {
    const BiasResult bias = evaluate_bias(model, terrain, base, q, qdot);
    const Vector4d right_hand_side = torque - model.damping.cwiseProduct(qdot) - bias.bias;
    return detail::solve_mass(bias.terms.mass, right_hand_side);
}

}  // namespace gimbal
