from __future__ import annotations
from pathlib import Path
import csv,json,math
import numpy as np

def load_rows(path):
    p=Path(path)
    if p.suffix.lower()=='.json':
        d=json.loads(p.read_text(encoding='utf-8')); return d if isinstance(d,list) else d.get('rows',[])
    rows=[]
    with p.open(encoding='utf-8') as f:
        for line in f:
            if line.strip(): rows.append(json.loads(line))
    return rows

def save_report(path,report,rows=None):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps({'metrics':report,'rows':rows or []},indent=2),encoding='utf-8')

def prf(tp,fp,fn):
    p=tp/(tp+fp) if tp+fp else 0.0;r=tp/(tp+fn) if tp+fn else 0.0;f=2*p*r/(p+r) if p+r else 0.0;return p,r,f

def p95(xs):return float(np.percentile(xs,95)) if xs else None

def box_iou(a,b):
    x1=max(a[0],b[0]);y1=max(a[1],b[1]);x2=min(a[2],b[2]);y2=min(a[3],b[3]);inter=max(0,x2-x1)*max(0,y2-y1);aa=max(0,a[2]-a[0])*max(0,a[3]-a[1]);bb=max(0,b[2]-b[0])*max(0,b[3]-b[1]);return inter/(aa+bb-inter+1e-12)
def mask_iou(a,b):
    import numpy as np
    from apex_grasp.common.io import load_mask
    A=load_mask(a);B=load_mask(b);return float(np.logical_and(A,B).sum()/max(1,np.logical_or(A,B).sum()))
def ece(conf,correct,bins=10):
    if not conf:return None
    c=np.asarray(conf);y=np.asarray(correct,dtype=float);v=0.0
    for lo in np.linspace(0,1,bins,endpoint=False):
        hi=lo+1/bins;m=(c>=lo)&(c<(hi if hi<1 else hi+1e-9))
        if m.any():v+=m.mean()*abs(float(c[m].mean()-y[m].mean()))
    return float(v)
