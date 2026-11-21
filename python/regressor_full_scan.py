"""完整惯性基的平滑器与回归方法扫描。"""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
configs=[("sg",w,o,"wls") for w in [21,31,41,51] for o in [3,5]]
configs += [("local",w,4,"wls") for w in [21,31,41]]
configs += [("spline",w,3,"wls") for w in [1e-4,1e-3]]
configs += [("sg",31,5,m) for m in ["tls","corrected"]]
def run(item):
    index,(method,width,order,mode)=item
    path=f"data/regressor_fullscan_{index:02d}.npz"
    command=["python3","regressor_fit.py","--method",method,"--width",str(width),"--order",str(order),"--mode",mode,"--save",path,"--segments","80","--stride","5","--evaluate"]
    result=subprocess.run(command,capture_output=True,text=True)
    print(index,result.stdout,result.stderr,flush=True)
    if result.returncode:
        return dict(configuration=dict(method=method,width=width,order=order,mode=mode),failure=result.stderr)
    return json.loads(Path(path+".json").read_text())
records=[]
with ThreadPoolExecutor(max_workers=3) as executor:
    for record in executor.map(run,enumerate(configs)):
        records.append(record)
        Path("data/regressor_fullscan_metrics.json").write_text(json.dumps(records,indent=2))
print("noisy_best",min((r for r in records if "validation" in r),key=lambda r:r["validation"]),flush=True)
