"""验证固定安装偏置时基参数映射的力矩线性性。"""
import numpy as np
import gimbal_dynamics as gd
import identify_physical as ip
import gradient_fit as gf

b = np.zeros(35)
b[[0,7,13,19]] = 0.1
b[[3,9,15]] = 1.0
b[31:35] = 0.05
b[23:31] = [0.05,0.03,0.02,0.01,0.1,0.02,0.03,0.1]
ctx = gd.Context(1)
t = gd.make_terrain(0.15,0.7)
base = gd.make_base(yaw=0.3)
s = gd.make_state([0.2,-0.3,0.4,0.2],[0.4,0.2,-0.3,0.1])
a = [1.0,-2.0,0.5,0.4]
def torque(beta):
    return np.array(ctx.inverse_dynamics(ip.build_model(ip.map_beta(beta,True)),t,base,s,a))
f = torque(b)
for i in list(range(21))+list(range(31,35)):
    e = np.zeros(35)
    e[i] = 0.001
    plus, minus = torque(b+e), torque(b-e)
    print(i, np.max(np.abs(plus+minus-2*f)), flush=True)
ctx.close()
