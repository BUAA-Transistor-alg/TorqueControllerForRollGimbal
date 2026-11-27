"""MPC 参数失真鲁棒性实验模块校验。

运行：python3 tests/mpc_robustness_check.py
"""

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

import gimbal_dynamics as gd
from mpc import robustness, scenarios
from mpc.controller import MpcController, controller_parameters


class RampController:
    """每次调用递增的指令，用于检查控制时延与执行增益。"""

    horizon = 1

    def __init__(self):
        self.count = 0
        self.last_info = {}

    def command(self, angle, velocity, target_angle, terrain, base):
        self.count += 1
        return np.array([0.1 * self.count, -0.05 * self.count,
                         0.02 * self.count, 0.03 * self.count])


def check_sampling():
    truth = robustness.TRUTH_RAW
    distortion = robustness.COMBINED_DISTORTION
    rng = np.random.default_rng(123)
    for _ in range(24):
        trial = robustness.sample_distortion(rng, distortion)
        raw = trial.raw
        masses = raw[0:4]
        assert np.all(masses >= truth[0:4] * 0.85 - 1e-12)
        assert np.all(masses <= truth[0:4] * 1.15 + 1e-12)
        com = raw[4:16]
        assert np.all(np.abs(com - truth[4:16]) <= 0.010 + 1e-12)
        inertia = raw[16:40]
        assert np.all(np.abs(inertia - truth[16:40])
                      <= 0.15 * np.abs(truth[16:40]) + 1e-12)
        offsets = raw[40:52]
        assert np.all(np.abs(offsets - truth[40:52]) <= 0.008 + 1e-12)
        damping = raw[52:56]
        assert np.all(damping >= truth[52:56] * 0.5 - 1e-12)
        assert np.all(damping <= truth[52:56] * 1.5 + 1e-12)

        gain = trial.torque_gain
        assert np.all(gain > 0.0)
        assert np.all(gain != 1.0)
        deviation = np.abs(gain - 1.0)
        assert np.all(deviation >= robustness.MIN_TORQUE_GAIN_DEVIATION - 1e-12)
        assert np.all(deviation <= 0.15 + 1e-12)

        assert np.all(np.abs(trial.angle_bias) <= 0.02 + 1e-12)
        assert np.all(np.abs(trial.velocity_bias) <= 0.05 + 1e-12)
        assert abs(trial.slope_bias) <= np.radians(3.0) + 1e-12
        assert abs(trial.direction_bias) <= np.radians(10.0) + 1e-12
        assert abs(trial.yaw_bias) <= np.radians(10.0) + 1e-12
        assert abs(trial.yaw_rate_bias) <= 0.10 + 1e-12
        assert np.all(np.abs(trial.base_acceleration_bias) <= 0.05 + 1e-12)

    first = robustness.sample_distortion(np.random.default_rng(9), distortion)
    second = robustness.sample_distortion(np.random.default_rng(9), distortion)
    assert np.array_equal(first.raw, second.raw)
    assert np.array_equal(first.torque_gain, second.torque_gain)
    assert np.array_equal(first.angle_bias, second.angle_bias)

    clean = robustness.sample_distortion(np.random.default_rng(4), robustness.Distortion())
    assert np.array_equal(clean.raw, truth)
    assert np.all(clean.torque_gain == 1.0)
    assert np.all(clean.angle_bias == 0.0)
    print("失真抽样范围、非单位增益与确定性校验通过")


def check_delay_and_gain():
    context = gd.Context(2)
    try:
        distortion = replace(robustness.NOMINAL_DISTORTION,
                             control_delay_steps=2, observation_delay_steps=2,
                             torque_gain_relative=0.15)
        trial = robustness.sample_distortion(np.random.default_rng(21), distortion)
        scenario = scenarios.Scenario(seed=3, steps=80)
        records = robustness.run_closed_loop(
            scenario, RampController(), trial, context, noise_seed=5)
        command = records["command"]
        applied = records["applied"]
        gain = trial.torque_gain
        assert np.all(records["control_delay"] >= 0)
        assert np.all(records["control_delay"] <= 2)
        assert np.all(records["observation_delay"] >= 0)
        assert np.all(records["observation_delay"] <= 2)
        for index in range(len(command)):
            expected = gain[None, :] * command[:index + 1]
            match = np.all(np.abs(applied[index][None, :] - expected) <= 1e-12, axis=1)
            positions = np.flatnonzero(match)
            if positions.size == 0:
                # 还没有任何指令到达时施加零力矩
                assert np.all(np.abs(applied[index]) <= 1e-12)
                continue
            send_index = int(positions[-1])
            assert index - send_index <= 2

        zero = replace(distortion, control_delay_steps=0, observation_delay_steps=0)
        clean_trial = robustness.sample_distortion(np.random.default_rng(21), zero)
        clean_records = robustness.run_closed_loop(
            scenario, RampController(), clean_trial, context, noise_seed=5)
        assert np.allclose(clean_records["applied"],
                           clean_trial.torque_gain[None, :] * clean_records["command"],
                           atol=1e-12)
        assert np.all(clean_records["control_delay"] == 0)
        assert np.all(clean_records["observation_delay"] == 0)

        episode_records = robustness.run_closed_loop(
            scenario, RampController(), trial, context, noise_seed=5,
            delay_mode="episode")
        assert np.unique(episode_records["control_delay"]).size == 1
        assert np.unique(episode_records["observation_delay"]).size == 1
    finally:
        context.close()
    print("控制时延、观测时延与执行增益校验通过")


def check_parameters():
    current = controller_parameters("current")
    robust = controller_parameters("robust")
    for parameters in (current, robust):
        assert parameters["horizon"] >= 1
        assert parameters["max_iter"] >= 1
        assert parameters["tolerance"] > 0.0
        for key in ("angle_weights", "velocity_weights", "torque_weights",
                    "torque_rate_weights", "limit_weights"):
            assert np.asarray(parameters["weights"][key]).shape == (4,)
    assert robust["horizon"] != current["horizon"]
    try:
        controller_parameters("unknown")
    except ValueError:
        pass
    else:
        raise AssertionError("未知参数集合应当抛出 ValueError")
    print("控制器参数集合校验通过")


def check_closed_loop():
    context = gd.Context(2)
    try:
        conditions = robustness.distortion_conditions()
        index = list(conditions).index("combined")
        seed = 7
        trial = robustness.sample_distortion(
            np.random.default_rng(seed * 1000003 + index * 7919 + 17),
            conditions["combined"])
        scenario = robustness.make_scenario(seed, 240)
        results = {}
        for name in ("current", "robust"):
            controller = MpcController(scenarios.design_model(),
                                       torque_limit=robustness.TORQUE_LIMIT,
                                       thread_count=2,
                                       **controller_parameters(name))
            controller.reset()
            records = robustness.run_closed_loop(
                scenario, controller, trial, context,
                noise_seed=seed * 1000003 + index * 7919 + 500017)
            controller.close()
            results[name] = robustness.compute_metrics(records, scenario)
            assert not results[name]["diverged"]
            assert np.all(np.isfinite(records["angle"]))
            assert np.max(np.abs(records["command"])) <= (
                np.max(robustness.TORQUE_LIMIT) + 1e-12)
        robust = results["robust"]
        assert robust["mean_rmse"] < 0.12, robust["mean_rmse"]
        assert robust["tail_rmse"] < 0.08, robust["tail_rmse"]
        assert robust["tail_saturation"] < 0.02, robust["tail_saturation"]
        print("全失真闭环校验通过：current RMSE=%.4f，robust RMSE=%.4f"
              % (results["current"]["mean_rmse"], robust["mean_rmse"]))
    finally:
        context.close()


def main():
    check_sampling()
    check_delay_and_gain()
    check_parameters()
    check_closed_loop()
    print("MPC 鲁棒性实验模块校验通过")


if __name__ == "__main__":
    main()
