// 校验：闭式 Christoffel 形式（forward_dynamics）与仿真内核使用的逆动力学递推形式
// 在随机状态、随机底盘运动与随机地形下给出相同的状态导数。
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>
#include <type_traits>

#include "gimbal_dynamics/dynamics_model.hpp"
#include "gimbal_dynamics/simulation.hpp"

using namespace gimbal;

int main() {
    std::mt19937 rng(20261108);
    std::uniform_real_distribution<double> unit(-1.0, 1.0);

    ModelParameters model = ModelParameters::demo();
    for (int i = 0; i < 4; ++i) {
        model.bodies[i].mass *= 1.0 + 0.3 * unit(rng);
        for (int c = 0; c < 3; ++c) {
            model.bodies[i].com[c] += 0.01 * unit(rng);
            model.joint_offsets[i][c] += 0.005 * unit(rng);
        }
    }

    Real max_relative = Real(0);
    for (int trial = 0; trial < 200; ++trial) {
        Terrain terrain;
        terrain.slope_angle = 0.3 * unit(rng);
        terrain.slope_direction = 3.0 * unit(rng);

        BaseState base;
        base.yaw = 3.0 * unit(rng);
        base.yaw_rate = 2.0 * unit(rng);
        base.yaw_acceleration = 5.0 * unit(rng);
        base.acceleration[0] = 3.0 * unit(rng);
        base.acceleration[1] = 3.0 * unit(rng);
        base.acceleration[2] = 3.0 * unit(rng);

        math::Vector4d q;
        math::Vector4d qdot;
        math::Vector4d torque;
        for (int i = 0; i < 4; ++i) {
            q[i] = 2.0 * unit(rng);
            qdot[i] = 4.0 * unit(rng);
            torque[i] = 2.0 * unit(rng);
        }

        const math::Vector4d closed = forward_dynamics(model, terrain, base, q, qdot, torque);

        simulation::State state;
        state.angle = q;
        state.velocity = qdot;
        const math::Vector8d derivative =
            simulation::state_derivative(model, terrain, base, state, torque);

        for (int i = 0; i < 4; ++i) {
            const Real error = std::fabs(closed[i] - derivative[4 + i]);
            max_relative = std::max(max_relative, error / std::max(Real(1), std::fabs(closed[i])));
            max_relative = std::max(
                max_relative,
                std::fabs(qdot[i] - derivative[i]) / std::max(Real(1), std::fabs(qdot[i])));
        }
    }

    printf("闭式形式与逆动力学递推形式的最大相对偏差 = %.3e\n",
           static_cast<double>(max_relative));
    const Real tolerance = std::is_same<Real, float>::value ? Real(1e-4) : Real(1e-10);
    if (!(max_relative < tolerance)) {
        printf("校验失败\n");
        return 1;
    }
    printf("校验通过\n");
    return 0;
}
