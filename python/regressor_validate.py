"""用共同含噪初始状态比较已保存模型。"""
import argparse
import json
from pathlib import Path
import numpy as np
import regressor_fit as rf
import regressor_full as full

a=argparse.Namespace(data="data/closed_loop_data_noisy.npz",stride=5,segments=80)
p=rf.Problem(a)
p.prepare("sg",51,5)
records=[]
paths=list(Path("data").glob("regressor_fullscan_*.npz"))+[Path("data/regressor_corrected41.npz"),Path("data/regressor_corrected51.npz"),Path("data/regressor_corrected61.npz")]
for path in paths:
    raw=np.load(path)["model"]
    score=full.validate(p,raw)
    record=dict(path=str(path),noisy_common_validation=score)
    records.append(record)
    print(record,flush=True)
Path("data/regressor_common_validation.json").write_text(json.dumps(records,indent=2))
print("selected",min(records,key=lambda r:r["noisy_common_validation"]),flush=True)
p.ctx.close()
