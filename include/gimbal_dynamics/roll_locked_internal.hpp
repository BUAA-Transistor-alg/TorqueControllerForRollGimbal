#pragma once

// 内部：锁定 roll 轴三自由度模型的链求值与阶段量。
//
// 与原 model_evaluation.hpp 的差别：
//   * 三个刚体、三个关节，关节轴依次为 z、z、x；
//   * 每个关节在局部旋转之前有一个固定旋转 F_i，R_i = R_{i-1} F_i R_local(q_i)；
//   * 第 0 个关节固定旋转为 I，第 1 个为 Ry(locked_angle)，第 2 个为 I。

#include <array>

#include "gimbal_dynamics/math_types.hpp"
#include "gimbal_dynamics/roll_locked.hpp"

namespace tcrg::gimbal_dynamics {
namespace roll_locked {
namespace detail {

using Vector3 = math::Vector<Real, 3>;
using Vector6 = math::Vector<Real, 6>;
using Matrix3 = math::Matrix<Real, 3, 3>;
using Matrix6 = math::Matrix<Real, 6, 6>;
using Matrix63 = math::Matrix<Real, 6, 3>;

// 三个关节轴在其父坐标系中的方向：z、z、x
Vector3 axis_local(int joint);

// 三个关节的局部旋转：Rz、Rz、Rx
Matrix3 local_rotation(int joint, Real angle);

// 三自由度链的位姿、关节原点、轴与质心，位置均相对底盘参考点。
struct ChainT {
    Vector3 normal = Vector3::zero();
    std::array<Matrix3, 3> rotation{};
    std::array<Vector3, 3> joint_origin{};
    std::array<Vector3, 3> offset{};  // r_i = R_{i-1} o_i
    std::array<Vector3, 3> com{};     // g_i
    std::array<Vector3, 3> axis{};
};

ChainT evaluate_chain(const ModelT& model, const Terrain& terrain, const BaseState& base,
                      const Vector3& angle);

// 质量矩阵、速度/角速度雅可比与惯量世界矩阵。
void evaluate_mass_matrix(const ModelT& model, const ChainT& chain, Matrix3& mass,
                          std::array<Matrix3, 3>* velocity_jacobian,
                          std::array<Matrix3, 3>* angular_jacobian,
                          std::array<Matrix3, 3>* inertia_world);

// 偏置力递推的中间量，供阶段 Jacobian 复用。
struct BiasInternals {
    std::array<Vector3, 3> omega{};
    std::array<Vector3, 3> alpha{};
    std::array<Vector3, 3> com_acceleration{};
    std::array<Vector3, 3> force{};
    std::array<Vector3, 3> couple{};
    std::array<Vector3, 3> joint_force{};
    std::array<Vector3, 3> joint_moment{};
};

// 偏置力 b：按逆动力学递推、令 q̈ = 0 求值。
Vector3 evaluate_bias(const ModelT& model, const ChainT& chain, const BaseState& base,
                      const Vector3& velocity, BiasInternals* internals = nullptr);

// 3×3 对称正定 Cholesky 分解
struct Cholesky3 {
    Real lower[3][3] = {{0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}};
    bool valid = false;

    void factorize(const Matrix3& mass);
    Vector3 solve(const Vector3& rhs) const;
};

Vector3 solve3(const Matrix3& mass, const Vector3& rhs);

// 阶段量：链、质量矩阵、偏置力与加速度
struct StageData {
    BaseState base;
    Vector3 angle = Vector3::zero();
    Vector3 velocity = Vector3::zero();
    ChainT chain;
    std::array<Matrix3, 3> velocity_jacobian{};
    std::array<Matrix3, 3> angular_jacobian{};
    std::array<Matrix3, 3> inertia_world{};
    BiasInternals bias_internals;
    Matrix3 mass = Matrix3::zero();
    Vector3 bias = Vector3::zero();
    Cholesky3 factor;
    Vector3 acceleration = Vector3::zero();
};

void evaluate_stage_data(const ModelT& model, const Terrain& terrain, const BaseState& base,
                         const StateT& state, const Vector3& torque, StageData& out);

// 阶段状态 Jacobian ∂f/∂y（6×6）与输入 Jacobian ∂f/∂τ（6×3）。
struct StageJacobians {
    Matrix6 state = Matrix6::zero();
    Matrix63 input = Matrix63::zero();
};

StageJacobians evaluate_stage_jacobians(const StageData& stage, const ModelT& model);

// 单子步 RK4 的传播子 W = ∂y'/∂y 与输入 Jacobian Ψ = ∂y'/∂τ。
void assemble_substep(const StageJacobians stage[4], Real h, Matrix6& propagation,
                      Matrix63& input_jacobian);

// 单个 RK4 阶段的输入，供反向伴随重算阶段量。
struct StageInput {
    BaseState base;
    Vector3 angle = Vector3::zero();
    Vector3 velocity = Vector3::zero();
    Vector3 torque = Vector3::zero();
};

// 主步损失与状态部分的梯度（力矩直接项不在此函数内）。
Real step_loss(const LossSpecT& loss, std::size_t num_steps, std::size_t step,
               const StateT& state, const Vector3& torque, const Vector3* previous_torque);
Vector6 step_loss_gradient(const LossSpecT& loss, std::size_t num_steps, std::size_t step,
                           const StateT& state);

}  // namespace detail
}  // namespace roll_locked
}  // namespace tcrg::gimbal_dynamics
