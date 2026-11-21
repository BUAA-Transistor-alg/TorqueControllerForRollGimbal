"""把多个字段一致的 npz 数据集按段拼接。"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description="拼接 npz 数据集")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("inputs", type=Path, nargs="+")
    arguments = parser.parse_args()
    payloads = [np.load(path) for path in arguments.inputs]
    keys = set(payloads[0].files)
    for payload in payloads[1:]:
        keys &= set(payload.files)
    merged = {}
    for key in sorted(keys):
        arrays = [payload[key] for payload in payloads]
        if (arrays[0].ndim > 0 and all(array.shape[1:] == arrays[0].shape[1:]
                                      for array in arrays)):
            merged[key] = np.concatenate(arrays, axis=0)
        else:
            merged[key] = arrays[0]
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(arguments.output, **merged)
    print("已保存", arguments.output, "段数", merged["q"].shape[0], "字段数", len(merged))


if __name__ == "__main__":
    main()
