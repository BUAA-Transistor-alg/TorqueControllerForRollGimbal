"""噪声尺度敏感性研究：对记录数据的各个字段加不同尺度的噪声，考察辨识性能。

两套采集协议：
  efficient —— 低激励协议（峰值速度 0.62 rad/s、峰值力矩 0.82 N·m）
  strong    —— 标准协议（峰值速度 1.86 rad/s、峰值力矩 1.52 N·m）

噪声配置：
  isotropic —— 位置/速度/力矩同时放大 k 倍
  field     —— 只放大某一个字段
  mixed     —— 各字段取不同倍数

每组配置用两个平滑宽度各拟合一��，取 0.5 s 开环 RMSE 较小者。
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

import eval_openloop as open_loop
import identify_physical as ip

NOMINAL = {"position": 0.001, "velocity": 0.1, "torque": 0.01, "slope": 0.0,
           "direction": 0.0, "yaw": 0.0}
PROTOCOLS = {
    "efficient": {"clean": Path("data/low_excitation_data.npz"), "widths": (31, 51)},
    "strong": {"clean": Path("data/closed_loop_data.npz"), "widths": (51, 71)},
}


def make_noisy(clean_path: Path, noise: dict, output: Path, seed: int) -> None:
    source = np.load(clean_path)
    fields = {key: source[key] for key in source.files}
    rng = np.random.default_rng(seed)
    q = fields["q"]
    qdot = fields["qdot"]
    tau = fields["tau"]
    terrain = fields["terrain"].copy()
    yaw = fields["base_yaw"].copy()
    noisy = dict(fields)
    noisy["q"] = q + rng.normal(0.0, noise["position"], q.shape)
    noisy["qdot"] = qdot + rng.normal(0.0, noise["velocity"], qdot.shape)
    noisy["tau"] = tau + rng.normal(0.0, noise["torque"], tau.shape)
    slope_std = noise.get("slope", 0.0)
    direction_std = noise.get("direction", 0.0)
    yaw_std = noise.get("yaw", 0.0)
    if slope_std > 0.0:
        noisy["terrain"][:, 0] += rng.normal(0.0, slope_std, terrain.shape[0])
    if direction_std > 0.0:
        noisy["terrain"][:, 1] += rng.normal(0.0, direction_std, terrain.shape[0])
    if yaw_std > 0.0:
        noisy["base_yaw"] = yaw + rng.normal(0.0, yaw_std, yaw.shape)
    noisy.update({"q_clean": q, "qdot_clean": qdot, "tau_clean": tau,
                  "initial": noisy["q"][:, 0].copy(),
                  "initial_velocity": noisy["qdot"][:, 0].copy(),
                  "initial_clean": q[:, 0].copy(),
                  "initial_velocity_clean": qdot[:, 0].copy(),
                  "noise_position_std": noise["position"],
                  "noise_velocity_std": noise["velocity"],
                  "noise_torque_std": noise["torque"]})
    np.savez_compressed(output, **noisy)


def fit_model(noisy: Path, width: int, save: Path) -> Path:
    last_error = None
    for mode in ("corrected", "wls"):
        result = subprocess.run([sys.executable, str(Path(__file__).with_name("regressor_full.py")),
                                 "--data", str(noisy),
                                 "--segments", "100", "--save", str(save), "--mode", mode,
                                 "--width", str(width), "--order", "5", "--stride", "3"],
                                capture_output=True, text=True)
        if result.returncode == 0:
            break
        last_error = result.stderr.strip().splitlines()[-1] if result.stderr else "未知错误"
    else:
        raise RuntimeError("拟合失败：%s" % last_error)
    try:
        ip.build_model(np.load(save)["model"])
        return save
    except RuntimeError:
        repaired = save.with_name(save.stem + "_repaired.npz")
        subprocess.run([sys.executable, str(Path(__file__).with_name("finalize_model.py")),
                        "--identified", str(save),
                        "--save", str(repaired), "--total-mass", "0",
                        "--inertia-floor", "3e-5"], check=True, stdout=subprocess.DEVNULL)
        return repaired


def evaluate_config(tag: str, clean: Path, noise: dict, widths, seed: int) -> dict:
    noisy = Path("data/noise_%s_noisy.npz" % tag)
    make_noisy(clean, noise, noisy, seed)
    best = None
    for width in widths:
        save = Path("data/noise_%s_w%d.npz" % (tag, width))
        try:
            usable = fit_model(noisy, width, save)
            metrics = open_loop.evaluate(np.load(usable)["model"])
        except Exception as error:
            print("  %s 宽度 %d 失败：%s" % (tag, width, error), flush=True)
            continue
        record = {"width": width, "rmse50": metrics["rmse"][50].tolist(),
                  "max50": metrics["maximum"][50].tolist(),
                  "direction": metrics["direction_wrong"]}
        if best is None or max(record["rmse50"]) < max(best["rmse50"]):
            best = record
    return {"tag": tag, "noise": dict(noise), "best": best}


def build_configs():
    configs = []
    for protocol_name, protocol in PROTOCOLS.items():
        if protocol_name == "efficient":
            for scale in (1, 2, 5, 10):
                configs.append((protocol_name, "iso%g" % scale,
                                {key: NOMINAL[key] * scale
                                 for key in ("position", "velocity", "torque")}))
            for field, key in (("pos", "position"), ("vel", "velocity"), ("tor", "torque")):
                for scale in (5, 10):
                    noise = NOMINAL.copy()
                    noise[key] = NOMINAL[key] * scale
                    configs.append((protocol_name, "%s%g" % (field, scale), noise))
            for scale in (5, 10):
                noise = NOMINAL.copy()
                noise["slope"] = np.radians(1.0 * scale)
                noise["direction"] = np.radians(2.0 * scale)
                noise["yaw"] = np.radians(3.0 * scale)
                configs.append((protocol_name, "geo%g" % scale, noise))
            configs.append((protocol_name, "mixed", {"position": 0.01, "velocity": 1.0,
                                                     "torque": 0.1,
                                                     "slope": np.radians(3.0),
                                                     "direction": np.radians(6.0),
                                                     "yaw": np.radians(9.0)}))
        else:
            for scale in (1, 2, 5, 10):
                configs.append((protocol_name, "iso%g" % scale,
                                {key: NOMINAL[key] * scale
                                 for key in ("position", "velocity", "torque")}))
    return configs


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="噪声敏感性研究")
    parser.add_argument("--part", type=int, default=0)
    parser.add_argument("--parts", type=int, default=1)
    arguments = parser.parse_args()
    configs = build_configs()
    configs = [config for index, config in enumerate(configs)
               if index % arguments.parts == arguments.part]
    results = []
    for protocol_name, tag, noise in configs:
        protocol = PROTOCOLS[protocol_name]
        clean = protocol["clean"]
        widths = protocol["widths"]
        print("### %s %s" % (protocol_name, tag), flush=True)
        record = evaluate_config("%s_%s" % (protocol_name, tag), clean, noise, widths,
                                 20261220)
        best = record["best"]
        if best is None:
            print("  %-16s 全部失败" % tag, flush=True)
        else:
            print("  %-16s 位置 %.4f 速度 %.3f 力矩 %.4f | 宽度 %d RMSE50 %s 方向 %d" % (
                tag, noise["position"], noise["velocity"], noise["torque"], best["width"],
                " ".join("%.2e" % v for v in best["rmse50"]), best["direction"]), flush=True)
        results.append({"protocol": protocol_name, **record})
    Path("data/noise_study_%d.json" % arguments.part).write_text(
        json.dumps(results, indent=2, ensure_ascii=False))


def _unused() -> None:
    results = []
    for protocol_name, protocol in PROTOCOLS.items():
        clean = protocol["clean"]
        widths = protocol["widths"]
        source = np.load(clean)
        print("### 协议 %s（峰值速度 %.2f，峰值力矩 %.2f）" % (
            protocol_name, np.abs(source["qdot"]).max(), np.abs(source["tau"]).max()),
            flush=True)
        configs = []
        for scale in (1, 2, 5, 10):
            configs.append(("iso%g" % scale, {key: NOMINAL[key] * scale
                                              for key in ("position", "velocity", "torque")}
                             , NOMINAL.copy()))
        for field, key in (("pos", "position"), ("vel", "velocity"), ("tor", "torque")):
            for scale in (2, 5, 10):
                noise = NOMINAL.copy()
                noise[key] = NOMINAL[key] * scale
                configs.append(("%s%g" % (field, scale), noise, NOMINAL.copy()))
        for scale in (2, 5, 10):
            noise = NOMINAL.copy()
            noise["slope"] = np.radians(1.0 * scale)
            noise["direction"] = np.radians(2.0 * scale)
            noise["yaw"] = np.radians(3.0 * scale)
            configs.append(("geo%g" % scale, noise, NOMINAL.copy()))
        configs.append(("mixed", {"position": 0.01, "velocity": 1.0, "torque": 0.1,
                                  "slope": np.radians(3.0), "direction": np.radians(6.0),
                                  "yaw": np.radians(9.0)}, NOMINAL.copy()))
        for tag, noise, _ in configs:
            record = evaluate_config("%s_%s" % (protocol_name, tag), clean, noise, widths,
                                     20261220)
            best = record["best"]
            if best is None:
                print("  %-16s 全部失败" % tag, flush=True)
            else:
                print("  %-16s 位置 %.4f 速度 %.3f 力矩 %.4f | 宽度 %d RMSE50 %s 方向 %d" % (
                    tag, noise["position"], noise["velocity"], noise["torque"],
                    best["width"], " ".join("%.2e" % v for v in best["rmse50"]),
                    best["direction"]), flush=True)
            results.append({"protocol": protocol_name, **record})
    Path("data/noise_study.json").write_text(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()