"""完整可辨识空间的物理约束线性回归。"""
import argparse
import json
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from scipy.optimize import minimize
import param_basis as pb
import identify_physical as ip
import gimbal_dynamics as gd
import regressor_fit as rf

def feasibility(p):
    values = [*p[40:44]]
    for body in range(4):
        v = p[body*10:body*10+10]
        m,h=v[0],v[1:4]
        I=np.array([[v[4],v[7],v[8]],[v[7],v[5],v[9]],[v[8],v[9],v[6]]])
        values.append(m-0.01)
        # 空间惯量正定约束对质量、一阶矩与原点惯量保持仿射。
        skew=np.array([[0.0,-h[2],h[1]],[h[2],0.0,-h[0]],[-h[1],h[0],0.0]])
        spatial=np.block([[I,skew],[-skew,m*np.eye(3)]])
        values.extend(np.linalg.eigvalsh(spatial)-1e-6)
    return np.array(values)

def basis_rows(problem,offsets,shift=None):
    raw=pb.nominal_model()
    raw[40:52]=offsets.reshape(-1)
    model=ip.build_model(raw)
    rows=[]
    for s,k in problem.samples:
        z=problem.states[:,s,k].copy()
        if shift is not None:
            z+=shift[k]
        rows.append(np.array(problem.ctx.inertial_basis(model,problem.terrains[s],problem.bases[s],gd.make_state(z[0],z[1]),z[2])).T)
    return np.array(rows).reshape(-1,44)

def solve(problem,offsets,prior,mode):
    B,_,rank=pb.compute_basis(offsets)
    assert rank==28
    Y=basis_rows(problem,offsets)
    p0=pb.inertial_from_raw(prior)
    p=p0.copy()
    for iteration in range(3):
        raw=pb.raw_from_inertial(p,offsets)
        model=ip.build_model(raw)
        variances=[]
        for s,k in problem.samples:
            z=problem.states[:,s,k]
            state=gd.make_state(z[0],z[1])
            zero=np.array(problem.ctx.inverse_dynamics(model,problem.terrains[s],problem.bases[s],state,np.zeros(4)))
            M=np.column_stack([np.array(problem.ctx.inverse_dynamics(model,problem.terrains[s],problem.bases[s],state,np.eye(4)[j]))-zero for j in range(4)])
            variances.append(1e-4*np.sum(problem.torque_filter[k]**2)+problem.cov[k,2,2]*np.sum(M*M,axis=1))
        w=1/np.sqrt(np.maximum(np.array(variances).reshape(-1),1e-10))
        A=(w[:,None]*Y)@B
        y=w*problem.y
        scales=1/np.sqrt(np.sum(A*A,axis=0))
        Z=A*scales
        if mode=="tls":
            _,_,vt=np.linalg.svd(np.column_stack([Z,y]),full_matrices=False)
            coordinate=-vt[-1,:-1]/vt[-1,-1]*scales
        elif mode=="corrected":
            correction=np.zeros((28,28))
            for axis in range(4):
                derivatives=[]
                for derivative in range(3):
                    shift=np.zeros((problem.q.shape[1],3,4))
                    shift[:,derivative,axis]=1e-4
                    derivatives.append(((basis_rows(problem,offsets,shift)-basis_rows(problem,offsets,-shift))/2e-4)@B)
                for d in range(3):
                    for e in range(3):
                        covariance=np.repeat([problem.cov[k,d,e] for s,k in problem.samples],4)
                        correction+=derivatives[d].T@((w*w*covariance)[:,None]*derivatives[e])
            H=A.T@A-correction
            assert np.linalg.eigvalsh(H)[0]>0, "噪声修正正规矩阵不正定"
            coordinate=np.linalg.solve(H,A.T@y)
        else:
            coordinate=np.linalg.lstsq(Z,y,rcond=1e-10)[0]*scales
        # 在完整参数空间施加物理约束，避免不可实现的无约束线性解。
        physical_scale=np.tile([1.0,0.1,0.1,0.1,0.02,0.02,0.02,0.02,0.02,0.02],4)
        physical_scale=np.r_[physical_scale,[0.1]*4]
        full=(w[:,None]*Y)*physical_scale
        H=full.T@full
        g=full.T@y
        if mode=="corrected":
            transform=B.T*physical_scale
            H=transform.T@(A.T@A-correction)@transform
            g=transform.T@(A.T@y)
        elif mode=="tls":
            transform=B.T*physical_scale
            H=transform.T@(A.T@A)@transform
            g=transform.T@(A.T@A@coordinate)
        normalizer=max(np.max(np.diag(H)),1.0)
        def objective(x):
            return (0.5*x@H@x-g@x)/normalizer,(H@x-g)/normalizer
        result=minimize(objective,p/physical_scale,jac=True,method="SLSQP",constraints=[{"type":"ineq","fun":lambda x:feasibility(x*physical_scale)}],options={"maxiter":5000,"ftol":1e-10})
        p=result.x*physical_scale
        assert result.success and np.min(feasibility(p))>=-1e-7, f"物理约束回归失败: {result.message}"
        coordinate=B.T@p
    residual=w*(Y@p-problem.y)
    return pb.raw_from_inertial(p,offsets),float(np.mean(residual*residual)),B,coordinate

def validate(problem,raw):
    model=ip.build_model(raw)
    errors=[]
    for s in range(80,len(problem.q)):
        for start in [30,90,150,210]:
            state=gd.make_state(problem.validation_states[0,s,start],problem.validation_states[1,s,start])
            end=problem.ctx.forward_simulate(model,problem.terrains[s],problem.bases[s],state,problem.tau[s,start:start+25],step=problem.dt,refinement=2)
            errors.append((np.array(list(end.angle))-problem.q[s,start+25])**2)
    return float(np.mean(errors))

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--data",default="data/closed_loop_data_noisy.npz")
    p.add_argument("--init",default="data/coarse_prior.npz")
    p.add_argument("--save",default="data/identified_regressor.npz")
    p.add_argument("--method",choices=["sg","local","spline"],default="sg")
    p.add_argument("--width",type=float,default=51)
    p.add_argument("--order",type=int,default=5)
    p.add_argument("--mode",choices=["wls","tls","corrected"],default="corrected")
    p.add_argument("--stride",type=int,default=3)
    p.add_argument("--segments",type=int,default=100)
    p.add_argument("--margin",type=int)
    p.add_argument("--outer",type=int,default=0)
    p.add_argument("--refine",type=int,default=0)
    p.add_argument("--prior-sigma",type=float,default=0.5)
    p.add_argument("--evaluate",action="store_true")
    args=p.parse_args()
    problem=rf.Problem(args)
    problem.prepare(args.method,args.width,args.order)
    prior=np.load(args.init)["model"]
    offsets=pb.NOMINAL_OFFSETS.copy()
    keep=[3,4,6,7,8,9,10,11]
    if args.outer:
        def objective(x):
            o=offsets.copy().reshape(-1)
            o[keep]=x
            raw,loss,_,_=solve(problem,o.reshape(4,3),prior,args.mode)
            print("outer",loss,flush=True)
            return loss
        result=minimize(objective,offsets.reshape(-1)[keep],method="L-BFGS-B",bounds=[(value-0.02,value+0.02) for value in offsets.reshape(-1)[keep]],options={"maxiter":args.outer,"maxls":5,"ftol":1e-7,"eps":1e-4})
        offsets.reshape(-1)[keep]=result.x
    raw,loss,B,z=solve(problem,offsets,prior,args.mode)
    score=validate(problem,raw)
    for iteration in range(args.refine):
        previous=problem.states.copy()
        problem.reconstruct(raw,args.prior_sigma)
        candidate,candidate_loss,candidate_basis,candidate_z=solve(problem,offsets,raw,args.mode)
        candidate_score=validate(problem,candidate)
        print("refinement",iteration,candidate_score,flush=True)
        if candidate_score>=score:
            problem.states=previous
            break
        raw,loss,B,z,score=candidate,candidate_loss,candidate_basis,candidate_z,candidate_score
    record=dict(configuration=vars(args),loss=loss,validation=score)
    np.savez(args.save,model=raw,reduced=np.r_[z,offsets.reshape(-1)[keep]],basis=B,inertial=pb.inertial_from_raw(raw),configuration=json.dumps(vars(args)))
    if args.evaluate:
        import eval_openloop as ol
        m=ol.evaluate(raw)
        record.update(rmse50=m["rmse"][50].tolist(),max50=m["maximum"][50].tolist(),direction_wrong=m["direction_wrong"],static_direction=m["static_direction"])
    Path(args.save+".json").write_text(json.dumps(record,indent=2))
    print(json.dumps(record),flush=True)
    problem.ctx.close()

if __name__=="__main__":
    main()
