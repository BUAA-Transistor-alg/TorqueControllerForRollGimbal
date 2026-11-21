"""按段区间切分数据集，用于分折辨识与方差评估。"""
from __future__ import annotations

import argparse
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description="切分 npz 数据集")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--end", type=int, required=True)
    arguments = parser.parse_args()
    source = np.load(arguments.input)
    payload = {}
    for key in source.files:
        value = source[key]
        if value.ndim > 0 and value.shape[0] == source["q"].shape[0]:
            payload[key] = value[arguments.start:arguments.end]
        else:
            payload[key] = value
    np.savez_compressed(arguments.output, **payload)
    print("已保存", arguments.output, "段数", payload["q"].shape[0])


if __name__ == "__main__":
    main()
