from __future__ import annotations
import argparse
from .utils import load_rows,save_report,p95

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--manifest',required=True);ap.add_argument('--output',required=True);a=ap.parse_args();rows=load_rows(a.manifest)
    t=[];g=[];task=[];recover=[];lat=[]
    for r in rows:
        t.append(bool(r.get('correct_target')));g.append(bool(r.get('valid_grasp')));task.append(bool(r.get('task_success')));recover.append(bool(r.get('recoverable_failure')))
        if r.get('latency_s') is not None:lat.append(float(r['latency_s']))
    mean=lambda x:sum(x)/len(x) if x else None
    save_report(a.output,{'instruction_to_correct_target_success':mean(t),'instruction_to_valid_grasp_success':mean(g),'end_to_end_task_success':mean(task),'recoverable_failure_rate':mean(recover),'mean_latency_s':mean(lat),'p95_latency_s':p95(lat),'acceptance_note':'The architecture document requires these metrics but does not specify universal numeric end-to-end thresholds; define them for your target hardware/scenario.'})
if __name__=='__main__':main()
