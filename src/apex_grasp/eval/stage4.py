from __future__ import annotations
import argparse
from apex_grasp.common.io import read_json
from apex_grasp.common.models import Stage4Result
from .utils import load_rows,save_report

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--manifest',required=True);ap.add_argument('--output',required=True);a=ap.parse_args();rows=load_rows(a.manifest)
    top1=[];topk=[];cf=[];cs=[];clear=[];sim=[];pre=post=0;detail=[]
    for r in rows:
        p=Stage4Result.model_validate(read_json(r['prediction_path']) if r.get('prediction_path') else r['prediction']);allg=p.grasp_candidates+p.rejected_candidates;pre+=len(allg);post+=len(p.grasp_candidates)
        valid_ids=set(r.get('valid_grasp_ids',[])); ids=[g.provenance.get('grasp_id') for g in p.grasp_candidates]
        if valid_ids:top1.append(bool(ids and ids[0] in valid_ids));topk.append(any(x in valid_ids for x in ids[:10]))
        if p.grasp_candidates:cf.extend([bool(g.collision_free) for g in p.grasp_candidates]);cs.extend([not r.get('constrained',False) or (g.constraint_score or 0)>0 for g in p.grasp_candidates]);clear.extend([g.clearance_m for g in p.grasp_candidates if g.clearance_m is not None])
        if r.get('simulation_success') is not None:sim.append(bool(r['simulation_success']))
        detail.append({'id':r.get('id'),'accepted':len(p.grasp_candidates),'rejected':len(p.rejected_candidates)})
    mean=lambda x:sum(x)/len(x) if x else None
    report={'top1_grasp_validity':mean(top1),'top10_recall':mean(topk),'collision_free_rate_post_filter':mean(cf),'constraint_satisfaction_rate':mean(cs),'minimum_accepted_clearance_m':min(clear) if clear else None,'simulation_success_rate':mean(sim),'pre_filter_candidate_count':pre,'post_filter_candidate_count':post,'acceptance_targets':{'top1_validity':0.90,'top10_recall':0.95,'collision_free_rate':0.95,'constraint_satisfaction':0.95,'simulation_success':0.80,'approach_clearance':'above configured safety margin'}}
    save_report(a.output,report,detail)
if __name__=='__main__':main()
