from __future__ import annotations
import argparse,math
from apex_grasp.common.io import read_json
from .utils import load_rows,save_report

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--manifest',required=True);ap.add_argument('--output',required=True);a=ap.parse_args();rows=load_rows(a.manifest)
    pos=[];rel=[];access=[];depth=[];graph=[];detail=[]
    for r in rows:
        p=read_json(r['prediction_path']) if r.get('prediction_path') else r['prediction'];obj=p['object_record']
        if r.get('centroid_xyz') and obj.get('centroid_xyz'):pos.append(math.dist(r['centroid_xyz'],obj['centroid_xyz']))
        expected_rel=r.get('relations',[]); pred={(x['relation'],tuple(x['reference_ids'])):x.get('satisfied') for x in p.get('relation_evidence',[])}
        for e in expected_rel:rel.append(pred.get((e['relation'],tuple(e['reference_ids'])))==e['satisfied'])
        if r.get('accessible') is not None:access.append((p.get('access_score',0)>=float(r.get('access_threshold',0.5)))==bool(r['accessible']))
        if r.get('depth_consistent') is not None:depth.append(bool(r['depth_consistent']))
        if r.get('scene_graph_consistent') is not None:graph.append(bool(r['scene_graph_consistent']))
        detail.append({'id':r.get('id'),'status':p.get('status')})
    mean=lambda x:sum(x)/len(x) if x else None
    report={'mean_3d_position_error_m':mean(pos),'relation_classification_accuracy':mean(rel),'accessibility_accuracy':mean(access),'depth_consistency_rate':mean(depth),'scene_graph_consistency_rate':mean(graph),'acceptance_note':'The architecture document specifies these Stage 3 metrics but does not give numeric acceptance thresholds. Set thresholds from your calibrated validation set; this evaluator does not invent them.'}
    save_report(a.output,report,detail)
if __name__=='__main__':main()
