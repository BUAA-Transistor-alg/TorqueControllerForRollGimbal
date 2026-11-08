#include "stage_jacobians.hpp"

namespace gimbal {
namespace detail {
namespace {

// 阶段 Jacobian 的块结构：Jx = [[0, I], [Jq, Jv]]。
// 直接按块计算 Jx * M，避免 4×4 零块与单位块的乘加。
template <int Columns>
math::Matrix<double, 8, Columns> multiply_jacobian(
    const math::Matrix8d& jacobian, const math::Matrix<double, 8, Columns>& matrix) {
    math::Matrix<double, 8, Columns> result;
    for (int column = 0; column < Columns; ++column) {
        for (int row = 0; row < 4; ++row) {
            result.data[row][column] = matrix.data[row + 4][column];
            double upper = 0.0;
            for (int k = 0; k < 4; ++k) {
                upper += jacobian.data[4 + row][k] * matrix.data[k][column];
            }
            double lower = 0.0;
            for (int k = 0; k < 4; ++k) {
                lower += jacobian.data[4 + row][4 + k] * matrix.data[4 + k][column];
            }
            result.data[4 + row][column] = upper + lower;
        }
    }
    return result;
}

// W = I + h/6 (J1 + 2 J2 A2 + 2 J3 A3 + J4 A4)
// A2 = I + h/2 J1，A3 = I + h/2 J2 A2，A4 = I + h J3 A3
math::Matrix8d propagation_matrix(const StageJacobians stage[4], double h) {
    const math::Matrix8d identity = math::Matrix8d::identity();
    const math::Matrix8d a2 = identity + stage[0].state * (0.5 * h);
    const math::Matrix8d a3 = identity + multiply_jacobian(stage[1].state, a2) * (0.5 * h);
    const math::Matrix8d a4 = identity + multiply_jacobian(stage[2].state, a3) * h;
    return identity + (stage[0].state + multiply_jacobian(stage[1].state, a2) * 2.0 +
                       multiply_jacobian(stage[2].state, a3) * 2.0 +
                       multiply_jacobian(stage[3].state, a4)) *
                          (h / 6.0);
}

}  // namespace

void assemble_substep(const StageJacobians stage[4], double h, math::Matrix8d& propagation,
                      math::Matrix84d& input_jacobian) {
    propagation = propagation_matrix(stage, h);
    const math::Matrix84d basis1 = stage[0].input;
    const math::Matrix84d basis2 =
        stage[1].input + multiply_jacobian(stage[1].state, basis1) * (0.5 * h);
    const math::Matrix84d basis3 =
        stage[2].input + multiply_jacobian(stage[2].state, basis2) * (0.5 * h);
    const math::Matrix84d basis4 =
        stage[3].input + multiply_jacobian(stage[3].state, basis3) * h;
    input_jacobian = (basis1 + basis2 * 2.0 + basis3 * 2.0 + basis4) * (h / 6.0);
}

void assemble_parameter_substep(const StageJacobians stage[4],
                                const ParameterJacobian stage_parameter[4], double h,
                                math::Matrix8d& propagation,
                                ParameterJacobian& parameter_jacobian) {
    propagation = propagation_matrix(stage, h);
    const ParameterJacobian basis1 = stage_parameter[0];
    const ParameterJacobian basis2 =
        stage_parameter[1] + multiply_jacobian(stage[1].state, basis1) * (0.5 * h);
    const ParameterJacobian basis3 =
        stage_parameter[2] + multiply_jacobian(stage[2].state, basis2) * (0.5 * h);
    const ParameterJacobian basis4 =
        stage_parameter[3] + multiply_jacobian(stage[3].state, basis3) * h;
    parameter_jacobian = (basis1 + basis2 * 2.0 + basis3 * 2.0 + basis4) * (h / 6.0);
}

}  // namespace detail
}  // namespace gimbal
