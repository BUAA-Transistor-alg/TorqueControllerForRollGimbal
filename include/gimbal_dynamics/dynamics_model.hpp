#pragma once

// 模型参数与底盘状态：公开类型按精度模板化，处理器在构造时选择精度。

#include <array>

#include "gimbal_dynamics/math_types.hpp"

namespace tcrg::gimbal_dynamics {

// 单个刚体的惯性参数
template <typename Real>
struct BodyParametersT {
    Real mass = Real(0);
    math::Vector<Real, 3> com = math::Vector<Real, 3>::zero();
    math::Matrix<Real, 3, 3> inertia = math::Matrix<Real, 3, 3>::zero();
};

// 串链模型参数。B0 为底盘，运动由外部给定。
template <typename Real>
struct ModelParametersT {
    std::array<BodyParametersT<Real>, 4> bodies{};
    std::array<math::Vector<Real, 3>, 4> joint_offsets = {
        math::Vector<Real, 3>::zero(), math::Vector<Real, 3>::zero(),
        math::Vector<Real, 3>::zero(), math::Vector<Real, 3>::zero()};
    math::Vector<Real, 4> damping = math::Vector<Real, 4>::zero();
    Real gravity = Real(9.81);

    // 示例参数，实际使用时替换为实测值
    static ModelParametersT demo() {
        ModelParametersT model;
        model.bodies = {{
            BodyParametersT<Real>{Real(0.90),
                                  math::Vector<Real, 3>::from(Real(0.004), Real(0.002), Real(0.018)),
                                  make_inertia(Real(0.00140), Real(0.00002), Real(0.00003),
                                                     Real(0.00155), Real(0.00001), Real(0.00110))},
            BodyParametersT<Real>{Real(0.42),
                                  math::Vector<Real, 3>::from(Real(0.012), Real(0.003), Real(0.032)),
                                  make_inertia(Real(0.00028), Real(0.00001), Real(0.00002),
                                                     Real(0.00033), Real(0.00001), Real(0.00021))},
            BodyParametersT<Real>{Real(0.25),
                                  math::Vector<Real, 3>::from(Real(-0.004), Real(-0.003), Real(0.022)),
                                  make_inertia(Real(0.00011), Real(0.000005), Real(0.000004),
                                                     Real(0.00014), Real(0.000003), Real(0.00009))},
            BodyParametersT<Real>{Real(0.55),
                                  math::Vector<Real, 3>::from(Real(0.008), Real(0.001), Real(0.026)),
                                  make_inertia(Real(0.00034), Real(0.00002), Real(0.00001),
                                                     Real(0.00039), Real(0.000015), Real(0.00026))},
        }};
        model.joint_offsets = {{
            math::Vector<Real, 3>::from(Real(0.0), Real(0.0), Real(0.045)),
            math::Vector<Real, 3>::from(Real(0.028), Real(0.0), Real(0.095)),
            math::Vector<Real, 3>::from(Real(0.0), Real(0.0), Real(0.072)),
            math::Vector<Real, 3>::from(Real(0.0), Real(0.0), Real(0.058)),
        }};
        model.damping = math::Vector<Real, 4>::from(Real(0.020), Real(0.015), Real(0.010), Real(0.008));
        model.gravity = Real(9.81);
        return model;
    }

private:
    static math::Matrix<Real, 3, 3> make_inertia(Real xx, Real xy, Real xz, Real yy, Real yz,
                                                Real zz) {
        math::Matrix<Real, 3, 3> inertia;
        inertia(0, 0) = xx;
        inertia(0, 1) = xy;
        inertia(0, 2) = xz;
        inertia(1, 0) = xy;
        inertia(1, 1) = yy;
        inertia(1, 2) = yz;
        inertia(2, 0) = xz;
        inertia(2, 1) = yz;
        inertia(2, 2) = zz;
        return inertia;
    }
};

// 地面坡角与坡向
template <typename Real>
struct TerrainT {
    Real slope_angle = Real(0);
    Real slope_direction = Real(0);
};

// 底盘运动输入
template <typename Real>
struct BaseStateT {
    math::Vector<Real, 3> acceleration = math::Vector<Real, 3>::zero();
    Real yaw = Real(0);
    Real yaw_rate = Real(0);
    Real yaw_acceleration = Real(0);
};

// 后端内部按构建精度使用的别名
using BodyParameters = BodyParametersT<Real>;
using ModelParameters = ModelParametersT<Real>;
using Terrain = TerrainT<Real>;
using BaseState = BaseStateT<Real>;

// 后端内部单步接口
math::Vector<Real, 4> forward_dynamics(const ModelParameters& model, const Terrain& terrain,
                                       const BaseState& base, const math::Vector<Real, 4>& q,
                                       const math::Vector<Real, 4>& qdot,
                                       const math::Vector<Real, 4>& torque);

}  // namespace tcrg::gimbal_dynamics
