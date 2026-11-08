#include "stage_jacobians.hpp"

namespace gimbal {
namespace detail {
namespace {

// 阶段 Jacobian 的块结构：Jx = [[0, I], [Jq, Jv]]。
// 直接按块计算 Jx * M，避免 4×4 零块与单位块的乘加。
template <int Columns>
Eigen::Matrix<double, 8, Columns> multiply_jacobian(
    const Eigen::Matrix<double, 8, 8>& jacobian,
    const Eigen::Matrix<double, 8, Columns>& matrix) {
    Eigen::Matrix<double, 8, Columns> result;
    result.template topRows<4>() = matrix.template bottomRows<4>();
    result.template bottomRows<4>() =
        jacobian.template block<4, 4>(4, 0) * matrix.template topRows<4>() +
        jacobian.template block<4, 4>(4, 4) * matrix.template bottomRows<4>();
    return result;
}

// W = I + h/6 (J1 + 2 J2 A2 + 2 J3 A3 + J4 A4)
// A2 = I + h/2 J1，A3 = I + h/2 J2 A2，A4 = I + h J3 A3
Eigen::Matrix<double, 8, 8> propagation_matrix(const StageJacobians stage[4], double h) {
    const Eigen::Matrix<double, 8, 8> identity = Eigen::Matrix<double, 8, 8>::Identity();
    const Eigen::Matrix<double, 8, 8> a2 = identity + 0.5 * h * stage[0].state;
    const Eigen::Matrix<double, 8, 8> a3 = identity + 0.5 * h * multiply_jacobian(stage[1].state, a2);
    const Eigen::Matrix<double, 8, 8> a4 = identity + h * multiply_jacobian(stage[2].state, a3);
    return identity + (h / 6.0) * (stage[0].state + 2.0 * multiply_jacobian(stage[1].state, a2) +
                                   2.0 * multiply_jacobian(stage[2].state, a3) +
                                   multiply_jacobian(stage[3].state, a4));
}

}  // namespace

void assemble_substep(const StageJacobians stage[4], double h,
                      Eigen::Matrix<double, 8, 8>& propagation,
                      Eigen::Matrix<double, 8, 4>& input_jacobian) {
    propagation = propagation_matrix(stage, h);
    const Eigen::Matrix<double, 8, 4> basis1 = stage[0].input;
    const Eigen::Matrix<double, 8, 4> basis2 =
        stage[1].input + 0.5 * h * multiply_jacobian(stage[1].state, basis1);
    const Eigen::Matrix<double, 8, 4> basis3 =
        stage[2].input + 0.5 * h * multiply_jacobian(stage[2].state, basis2);
    const Eigen::Matrix<double, 8, 4> basis4 =
        stage[3].input + h * multiply_jacobian(stage[3].state, basis3);
    input_jacobian = (h / 6.0) * (basis1 + 2.0 * basis2 + 2.0 * basis3 + basis4);
}

void assemble_parameter_substep(const StageJacobians stage[4],
                                const ParameterJacobian stage_parameter[4], double h,
                                Eigen::Matrix<double, 8, 8>& propagation,
                                ParameterJacobian& parameter_jacobian) {
    propagation = propagation_matrix(stage, h);
    const ParameterJacobian basis1 = stage_parameter[0];
    const ParameterJacobian basis2 =
        stage_parameter[1] + 0.5 * h * multiply_jacobian(stage[1].state, basis1);
    const ParameterJacobian basis3 =
        stage_parameter[2] + 0.5 * h * multiply_jacobian(stage[2].state, basis2);
    const ParameterJacobian basis4 =
        stage_parameter[3] + h * multiply_jacobian(stage[3].state, basis3);
    parameter_jacobian = (h / 6.0) * (basis1 + 2.0 * basis2 + 2.0 * basis3 + basis4);
}

}  // namespace detail
}  // namespace gimbal
