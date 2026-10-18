from pathlib import Path
import subprocess,os,json,time,sys
out=Path('/home/cns/HiPepTox/Paper/model/baseline/toxteller/results/full_length_5_50_20261013')
exe='/home/cns/.cache/HiPepTox/toxteller_full_length_20261013/bin/python'
for attempt in range(90):
    p=subprocess.run([exe,'-c','import numpy,pandas,sklearn,xgboost'],capture_output=True,text=True,timeout=20)
    if p.returncode==0:break
    (out/'progress.json').write_text(json.dumps({'phase':'waiting_dependency','attempt':attempt,'last_error':p.stderr[-500:]}))
    time.sleep(5)
else:
    (out/'progress.json').write_text(json.dumps({'phase':'failed','error':'Dependency installation did not finish.'}))
    sys.exit(1)
result=subprocess.run([exe,str(out/'run_full_benchmark.py')],cwd=str(out))
if result.returncode:
    (out/'progress.json').write_text(json.dumps({'phase':'failed','returncode':result.returncode,'log':str(out/'launcher.log')}))
sys.exit(result.returncode)
