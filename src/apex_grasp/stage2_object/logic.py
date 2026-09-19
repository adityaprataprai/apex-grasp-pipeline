from __future__ import annotations
from pathlib import Path
import numpy as np
from PIL import Image
from apex_grasp.common.models import TaskIR,ObjectRecord,ObjectEvidence,Stage2Result
from apex_grasp.common.config import require
from apex_grasp.common.errors import PipelineFailure
from .adapters import make_detector,make_segmenter,make_attribute_verifier

def iou(a,b):
    x1=max(a[0],b[0]); y1=max(a[1],b[1]); x2=min(a[2],b[2]); y2=min(a[3],b[3]); inter=max(0,x2-x1)*max(0,y2-y1)
    aa=max(0,a[2]-a[0])*max(0,a[3]-a[1]); bb=max(0,b[2]-b[0])*max(0,b[3]-b[1]); return inter/(aa+bb-inter+1e-12)

def deduplicate(ds,thr):
    ds=sorted(ds,key=lambda d:float(d.get("prompt_score",d.get("category_score",0))),reverse=True); keep=[]
    for d in ds:
        if all(iou(d["bbox_xyxy"],k["bbox_xyxy"])<thr for k in keep): keep.append(d)
    return keep

def prompts_for(desc):
    cat=desc.category
    if not cat: return []
    attrs=desc.attributes.model_dump(exclude_none=True)
    ps=[cat]
    for v in attrs.values(): ps.append(f"{v} {cat}")
    if attrs: ps.append(" ".join([str(v) for v in attrs.values()]+[cat]))
    return list(dict.fromkeys(ps))

def clean_mask(mask,bbox,min_ratio):
    import cv2
    m=np.asarray(mask,dtype=np.uint8)
    n,labels,stats,_=cv2.connectedComponentsWithStats(m,8)
    if n>1:
        idx=1+int(np.argmax(stats[1:,cv2.CC_STAT_AREA])); m=(labels==idx).astype(np.uint8)
    box_area=max(1.0,(bbox[2]-bbox[0])*(bbox[3]-bbox[1]))
    if m.sum()/box_area < min_ratio: return None
    return m.astype(bool)

def relation_2d_score(rel,target_box,ref_box):
    tx=(target_box[0]+target_box[2])/2; ty=(target_box[1]+target_box[3])/2; rx=(ref_box[0]+ref_box[2])/2; ry=(ref_box[1]+ref_box[3])/2
    typ=rel.type
    if typ=="LEFT_OF": return 1.0 if tx<rx else 0.0
    if typ=="RIGHT_OF": return 1.0 if tx>rx else 0.0
    if typ=="ABOVE": return 1.0 if ty<ry else 0.0
    if typ=="BELOW": return 1.0 if ty>ry else 0.0
    # Depth-sensitive relations are hints only; neutral evidence prevents pretending 2D proves 3D.
    return 0.5

def weighted(e:dict,weights:dict):
    active=[(k,v) for k,v in e.items() if v is not None and weights.get(k) is not None]
    if not active: return 0.0
    denom=sum(float(weights[k]) for k,_ in active)
    if denom<=0: raise PipelineFailure("CONFIG_INVALID","Stage 2 active scoring weights must sum to > 0")
    return sum(float(weights[k])*float(v) for k,v in active)/denom

def ground_description(image,desc,cfg,artifact_dir,prefix):
    detector=make_detector(cfg); segmenter=make_segmenter(cfg); verifier=make_attribute_verifier(cfg)
    ps=prompts_for(desc)
    if not ps: return []
    ds=deduplicate(detector.detect(image,ps),float(require(cfg,"stage2.nms_iou_threshold")))
    boxes=[d["bbox_xyxy"] for d in ds]; masks=segmenter.segment(image,boxes)
    if len(masks)!=len(boxes): raise PipelineFailure("SEGMENTER_FAILED","Segmenter returned a different number of masks than boxes")
    out=[]; art=Path(artifact_dir); art.mkdir(parents=True,exist_ok=True)
    for i,(d,m) in enumerate(zip(ds,masks),1):
        cm=clean_mask(m,d["bbox_xyxy"],float(require(cfg,"stage2.min_mask_area_ratio")))
        if cm is None: continue
        mask_path=art/f"{prefix}_{i:02d}_mask.png"; Image.fromarray((cm*255).astype(np.uint8)).save(mask_path)
        attr_score=None; attr_meta={}
        if verifier is not None: attr_score,attr_meta=verifier.score(image,cm,desc.attributes.model_dump(exclude_none=True))
        ev=ObjectEvidence(category=float(d.get("category_score",d.get("prompt_score",0))),attribute=attr_score,visual_quality=min(1.0,float(cm.sum())/max(1.0,(d['bbox_xyxy'][2]-d['bbox_xyxy'][0])*(d['bbox_xyxy'][3]-d['bbox_xyxy'][1]))),prompt_similarity=float(d.get("prompt_score",d.get("category_score",0))))
        out.append(ObjectRecord(object_id=f"{prefix}_{i:02d}",category=desc.category,attributes={**desc.attributes.model_dump(exclude_none=True),"attribute_meta":attr_meta},bbox_xyxy=[float(x) for x in d["bbox_xyxy"]],mask_path=str(mask_path),confidence_2d=0.0,evidence=ev,provenance={"raw_detector_id":d.get("raw_id"),"prompt":d.get("prompt")}))
    return out

def run_stage2(image,task:TaskIR,cfg,artifact_dir):
    targets=ground_description(image,task.target,cfg,artifact_dir,"obj")
    if not targets: raise PipelineFailure("TARGET_NOT_FOUND","No target candidates survived Stage 2")
    refs={}
    for r in task.references: refs[r.id]=ground_description(image,r,cfg,artifact_dir,f"{r.id}_obj")
    if any(len(v)==0 for v in refs.values()): raise PipelineFailure("REFERENCE_NOT_FOUND","A required reference has no candidates")
    weights={k:require(cfg,f"stage2.scoring_weights.{k}") for k in ["category","attribute","reference_2d","visual_quality","prompt_similarity"]}
    for t in targets:
        relscores=[]
        for rel in task.relations:
            rid=rel.reference[0] if isinstance(rel.reference,list) else rel.reference
            if refs.get(rid): relscores.append(max(relation_2d_score(rel,t.bbox_xyxy,r.bbox_xyxy) for r in refs[rid]))
        t.evidence.reference_2d=(sum(relscores)/len(relscores)) if relscores else None
        e=t.evidence.model_dump(); t.confidence_2d=weighted(e,weights)
    targets.sort(key=lambda x:x.confidence_2d,reverse=True)
    top1=targets[0]; top2=targets[1].confidence_2d if len(targets)>1 else 0.0; margin=top1.confidence_2d-top2
    accept=top1.confidence_2d>=float(require(cfg,"stage2.min_accept_score")) and margin>=float(require(cfg,"stage2.min_margin"))
    return Stage2Result(target_object=top1 if accept else None,candidates=targets[:int(require(cfg,"stage2.top_k"))],references={k:v[:int(require(cfg,"stage2.top_k"))] for k,v in refs.items()},needs_3d_disambiguation=(not accept or bool(task.relations)),decision_margin=margin)
