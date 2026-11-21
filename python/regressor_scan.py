"""系统扫描平滑器与误差在变量处理，评估独立保存。"""
import subprocess
import json
from pathlib import Path

configs = [("sg",w,o,"wls") for w in [7,11,21,31,41] for o in [3,5] if w>o]
configs += [("local",w,4,"wls") for w in [11,21,31]]
configs += [("spline",w,3,"wls") for w in [1e-7,1e-6,1e-5,1e-4]]
configs += [("sg",11,3,m) for m in ["tls","corrected"]]
records=[]
for index,(method,width,order,mode) in enumerate(configs):
    path = Path(f"data/regressor_scan_{index:02d}.npz")
    command = ["python3","regressor_fit.py","--method",method,"--width",str(width),"--order",str(order),"--mode",mode,"--save",str(path),"--segments","80","--stride","5","--evaluate"]
    subprocess.run(command,check=True)
    record=json.loads(Path(str(path)+".json").read_text())
    records.append(record)
    Path("data/regressor_scan_metrics.json").write_text(json.dumps(records,indent=2))
print("noisy_validation_best",min(records,key=lambda r:r["validation"]),flush=True)
