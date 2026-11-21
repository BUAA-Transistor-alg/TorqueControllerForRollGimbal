"""含噪验证段中消去初始状态误差后评估轨迹残差。"""
import argparse
import json
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import gimbal_dynamics as gd
import identify_physical as ip
import regressor_fit as rf

a=argparse.Namespace(data="data/closed_loop_data_noisy.npz",stride=5,segments=80)
p=rf.Problem(a)
p.prepare("sg",51,5)

def trajectory(model,s,start,x):
    state=gd.make_state(x[:4],x[4:])
    rows=[x[:4]]
    for torque in p.tau[s,start:start+50]:
        state=p.ctx.forward_step(model,p.terrains[s],p.bases[s],state,torque,step=p.dt,refinement=2)
        rows.append(np.array(list(state.angle)))
    return np.array(rows)

def score(raw):
    model=ip.build_model(raw)
    errors=[]
    for s in range(80,100):
        for start in [30,120,210]:
            x=np.r_[p.validation_states[0,s,start],p.validation_states[1,s,start]]
            expected=p.q[s,start:start+51]
            for iteration in range(2):
                value=trajectory(model,s,start,x)
                derivatives=[]
                for j in range(8):
                    e=np.zeros(8)
                    e[j]=1e-5
                    derivatives.append(((trajectory(model,s,start,x+e)-value)/1e-5).reshape(-1))
                jacobian=np.array(derivatives).T
                delta=np.linalg.lstsq(jacobian,(expected-value).reshape(-1),rcond=None)[0]
                x+=delta
            errors.extend((trajectory(model,s,start,x)-expected).reshape(-1)**2)
    return float(np.mean(errors))

paths=["data/regressor_fullscan_02.npz","data/regressor_fullscan_07.npz","data/regressor_fullscan_14.npz","data/regressor_corrected41.npz","data/regressor_corrected51.npz","data/regressor_corrected61.npz"]
records=[]
for path in paths:
    record=dict(path=path,noisy_initial_profile_score=score(np.load(path)["model"]))
    records.append(record)
    print(record,flush=True)
Path("data/regressor_shooting_validation.json").write_text(json.dumps(records,indent=2))
print("selected",min(records,key=lambda r:r["noisy_initial_profile_score"]),flush=True)
p.ctx.close()
