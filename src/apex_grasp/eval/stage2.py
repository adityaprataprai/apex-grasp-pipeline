from __future__ import annotations
import argparse
from apex_grasp.common.io import read_json
from apex_grasp.common.models import Stage2Result
from .utils import load_rows,save_report,box_iou,mask_iou,ece,p95

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--manifest',required=True);ap.add_argument('--output',required=True);a=ap.parse_args();rows=load_rows(a.manifest)
    acc=[];rec5=[];bi=[];mi=[];aa=[];amb=[];conf=[];corr=[];lat=[];details=[]
    for r in rows:
        p=Stage2Result.model_validate(read_json(r['prediction_path']) if r.get('prediction_path') else r['prediction']);gtid=r['target_object_id'];ids=[x.object_id for x in p.candidates];top=(p.target_object.object_id if p.target_object else (ids[0] if ids else None));ok=top==gtid;acc.append(ok);rec5.append(gtid in ids[:5]);conf.append((p.target_object.confidence_2d if p.target_object else (p.candidates[0].confidence_2d if p.candidates else 0)));corr.append(ok)
        pred_obj=next((x for x in p.candidates if x.object_id==top),None)
        if pred_obj and r.get('bbox_xyxy'):bi.append(box_iou(pred_obj.bbox_xyxy,r['bbox_xyxy']))
        if pred_obj and pred_obj.mask_path and r.get('mask_path'):mi.append(mask_iou(pred_obj.mask_path,r['mask_path']))
        if r.get('attribute_correct') is not None:aa.append(bool(r['attribute_correct']))
        if r.get('intentionally_ambiguous') is not None:amb.append((not r['intentionally_ambiguous']) or p.needs_3d_disambiguation)
        if r.get('latency_s') is not None:lat.append(float(r['latency_s']))
        details.append({'id':r.get('id'),'top1_correct':ok,'recall5':gtid in ids[:5]})
    mean=lambda x:sum(x)/len(x) if x else None
    report={'grounding_accuracy_at_1':mean(acc),'recall_at_5':mean(rec5),'mean_box_iou':mean(bi),'mean_mask_iou':mean(mi),'attribute_accuracy':mean(aa),'ambiguity_deferral_rate':mean(amb),'expected_calibration_error':ece(conf,corr),'p95_latency_s':p95(lat),'acceptance_targets':{'accuracy_at_1':0.90,'recall_at_5':0.97,'mean_box_iou':0.70,'mean_mask_iou':0.75,'attribute_accuracy':0.90,'ambiguity_deferral':0.90,'ece_max':0.10,'p95_latency_s_max':5.0}}
    save_report(a.output,report,details)
if __name__=='__main__':main()
