from __future__ import annotations
import argparse
from apex_grasp.common.models import TaskIR
from apex_grasp.common.io import read_json
from .utils import load_rows,save_report,prf,p95

def slots(task):
    s=set();d=task.target.attributes.model_dump(exclude_none=True)
    if task.target.category:s.add(('category',task.target.category))
    for k,v in d.items():s.add((k,str(v)))
    return s

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--manifest',required=True);ap.add_argument('--output',required=True);a=ap.parse_args();rows=load_rows(a.manifest)
    n=len(rows);valid=act=relok=0;tp=fp=fn=0;atp=afp=afn=0;amb_tp=amb_fp=amb_fn=0;hall=0;lat=[];detail=[]
    for r in rows:
        gt=TaskIR.model_validate(r['expected']);pred_raw=read_json(r['prediction_path']) if r.get('prediction_path') else r.get('prediction')
        try: pred=TaskIR.model_validate(pred_raw);valid+=1
        except Exception as e:detail.append({'id':r.get('id'),'valid':False,'error':str(e)});continue
        act+=pred.action==gt.action
        G=slots(gt);P=slots(pred);tp+=len(G&P);fp+=len(P-G);fn+=len(G-P)
        GA={(k,v) for k,v in G if k!='category'};PA={(k,v) for k,v in P if k!='category'};atp+=len(GA&PA);afp+=len(PA-GA);afn+=len(GA-PA)
        gr={(x.type,tuple(x.reference if isinstance(x.reference,list) else [x.reference])) for x in gt.relations};pr={(x.type,tuple(x.reference if isinstance(x.reference,list) else [x.reference])) for x in pred.relations};relok+=gr==pr
        ga=bool(gt.unresolved or gt.ambiguity>=0.5);pa=bool(pred.unresolved or pred.ambiguity>=0.5);amb_tp+=ga and pa;amb_fp+=(not ga) and pa;amb_fn+=ga and (not pa)
        source=r.get('instruction',gt.provenance.original_instruction).lower()
        # Unsupported inserted attributes = predicted non-null slots not present in expected slots.
        hall+=1 if len(P-G)>0 else 0
        if r.get('latency_s') is not None:lat.append(float(r['latency_s']))
        detail.append({'id':r.get('id'),'valid':True,'correct_action':pred.action==gt.action})
    _,_,catf=prf(tp,fp,fn);_,_,attrf=prf(atp,afp,afn);_,_,ambf=prf(amb_tp,amb_fp,amb_fn)
    report={'json_validity':valid/n if n else 0,'action_accuracy':act/n if n else 0,'target_and_slot_f1':catf,'attribute_f1':attrf,'relation_exact_accuracy':relok/n if n else 0,'ambiguity_f1':ambf,'hallucination_example_rate':hall/n if n else 0,'p95_latency_s':p95(lat),'acceptance_targets':{'json_validity':0.99,'action_accuracy':0.98,'target_category_f1':0.95,'attribute_f1':0.92,'relation_accuracy':0.95,'ambiguity_f1':0.90,'hallucination_rate_max':0.01,'p95_latency_s_max':2.0}}
    save_report(a.output,report,detail)
if __name__=='__main__':main()
