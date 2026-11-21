"""验证完整惯性回归量的重建恒等式与结构秩。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import param_basis as pb
import gimbal_dynamics as gd
import identify_physical as ip

raw=pb.prior_model()
rows=pb.sensitivity(offsets=raw[40:52].reshape(4,3),model=raw,count=120)
values=pb.inertial_from_raw(raw)
ctx=gd.Context(1)
model=ip.build_model(raw)
torques=[]
for q,v,a,slope,direction,yaw in pb.sample_states():
    torques.extend(ctx.inverse_dynamics(model,gd.make_terrain(slope,direction),gd.make_base(yaw=yaw),gd.make_state(q,v),a))
error=np.max(np.abs(rows@values-torques))
assert error<1e-8
singular=np.linalg.svd(rows,compute_uv=False)
rank=np.sum(singular>singular[0]*1e-8)
assert rank==28
print("重建误差",error,"完整惯性阻尼秩",rank,"奇异值",singular)
ctx.close()
