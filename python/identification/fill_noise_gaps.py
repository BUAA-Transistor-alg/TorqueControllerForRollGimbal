"""补齐噪声研究中失败的配置：加宽平滑窗口并允许退回普通加权最小二乘。"""
from __future__ import annotations

import glob
import json
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import study_noise as sn


def main() -> None:
    configs = sn.build_configs()
    lookup = {"%s_%s" % (protocol, tag): (protocol, tag, noise)
              for protocol, tag, noise in configs}
    for path in sorted(glob.glob("data/noise_study_[0-9].json")):
        records = json.loads(Path(path).read_text())
        changed = False
        for record in records:
            if record["best"] is not None:
                continue
            key = record["tag"]
            if key not in lookup:
                continue
            protocol_name, tag, noise = lookup[key]
            protocol = sn.PROTOCOLS[protocol_name]
            print("补齐 %s" % key, flush=True)
            extended = sn.evaluate_config(key + "_filled", protocol["clean"], noise,
                                          (51, 71, 91), 20261220)
            if extended["best"] is not None:
                record["best"] = extended["best"]
                record["filled"] = True
                changed = True
                print("  %s 宽度 %d RMSE50 %s 方向 %d" % (
                    key, extended["best"]["width"],
                    " ".join("%.2e" % v for v in extended["best"]["rmse50"]),
                    extended["best"]["direction"]), flush=True)
            else:
                print("  %s 仍然失败" % key, flush=True)
        if changed:
            Path(path).write_text(json.dumps(records, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
