"""位置求导、噪声传播与完整惯性回归的命令行入口。"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from scipy.interpolate import make_smoothing_spline
import gimbal_dynamics as gd
import identify_physical as ip

def smoother(n,dt,method,width,order):
    t=np.arange(n)*dt
    matrices=[]
    for derivative in range(3):
        if method=="spline":
            assert width>0.0
            eye=np.eye(n)
            matrix=np.column_stack([make_smoothing_spline(t,eye[:,j],lam=width)(t,nu=derivative) for j in range(n)])
        else:
            assert int(width)==width and int(width)%2==1 and 2<=order<int(width)<=n
            matrix=np.zeros((n,n))
            half=int(width)//2
            for k in range(n):
                start=min(max(k-half,0),n-int(width))
                offsets=(np.arange(start,start+int(width))-k)*dt
                design=np.vander(offsets,N=order+1,increasing=True)
                weights=np.ones(int(width)) if method=="sg" else np.exp(-0.5*(offsets/(half*dt/2))**2)
                inverse=np.linalg.solve(design.T@(weights[:,None]*design),design.T*weights)
                matrix[k,start:start+int(width)]=inverse[derivative]*(2 if derivative==2 else 1)
        matrices.append(matrix)
    return np.array(matrices)

class Problem:
    def __init__(self,args):
        data=np.load(args.data)
        self.q=data["q"]
        self.tau=data["tau"]
        self.dt=float(data["dt"])
        assert self.q.shape==self.tau.shape and self.q.shape[2]==4
        assert np.isfinite(self.q).all() and np.isfinite(self.tau).all()
        self.terrains=[gd.make_terrain(*row) for row in data["terrain"]]
        self.bases=[gd.make_base(yaw=float(y)) for y in data["base_yaw"]]
        self.ctx=gd.Context(1)
        self.stride=args.stride
        self.margin=getattr(args,"margin",None)
        self.limit=args.segments or len(self.q)
        assert self.stride>0 and 0<self.limit<=len(self.q)

    def prepare(self,method,width,order):
        self.filters=smoother(self.q.shape[1],self.dt,method,width,order)
        self.states=np.einsum("dkt,stj->dskj",self.filters,self.q)
        # 使用共同验证初始状态，防止平滑器改变验证起点造成评分偏差。
        validation_filters=smoother(self.q.shape[1],self.dt,"sg",51,5)
        self.validation_states=np.einsum("dkt,stj->dskj",validation_filters,self.q)
        n=self.q.shape[1]
        # 匹配零阶保持输入的二阶导数滤波核。
        lag=np.arange(n)[:,None]-np.arange(n)[None,:]
        integration=np.maximum(lag-0.5,0.0)*(self.dt*self.dt)
        self.torque_filter=self.filters[2]@integration
        self.targets=np.einsum("kt,stj->skj",self.torque_filter,self.tau)
        margin=max(20,int(width)//2+1) if method!="spline" else 20
        margin=margin if self.margin is None else self.margin
        assert 0<=margin<n//2
        self.samples=[(s,k) for s in range(self.limit) for k in range(margin,n-margin,self.stride)]
        self.y=np.array([self.targets[s,k] for s,k in self.samples]).reshape(-1)
        self.cov=np.einsum("dkt,ekt->kde",self.filters,self.filters)*1e-6

    def reconstruct(self,raw,sigma):
        assert sigma>0.0
        model=ip.build_model(raw)
        n=self.q.shape[1]
        D=np.diff(np.eye(n),n=2,axis=0)/(self.dt*self.dt)
        precision=np.eye(n)/1e-6+D.T@D/(sigma*sigma)
        reconstructed=self.q.copy()
        for s in range(len(self.q)):
            acceleration=[]
            for k in range(1,n-1):
                state=gd.make_state(self.states[0,s,k],self.states[1,s,k])
                torque=0.5*(self.tau[s,k-1]+self.tau[s,k])
                advanced=self.ctx.forward_step(model,self.terrains[s],self.bases[s],state,torque,step=1e-6,refinement=1)
                acceleration.append((np.array(list(advanced.velocity))-self.states[1,s,k])/1e-6)
            rhs=self.q[s]/1e-6+D.T@np.array(acceleration)/(sigma*sigma)
            reconstructed[s]=np.linalg.solve(precision,rhs)
        self.states=np.einsum("dkt,stj->dskj",self.filters,reconstructed)

if __name__=="__main__":
    from regressor_full import main
    main()
