from __future__ import annotations
from pathlib import Path
import numpy as np
from apex_grasp.common.models import TaskIR,Stage2Result,ObjectRecord,CameraIntrinsics,RelationEvidence,TargetObject
from apex_grasp.common.io import load_mask
from apex_grasp.common.config import require
from apex_grasp.common.errors import PipelineFailure
from .geometry import backproject,transform_points,robust_filter,voxel_downsample_np,bbox_extent,pointcloud_distance

def _norm(v):
    v=np.asarray(v,dtype=float); n=np.linalg.norm(v)
    if n<=0: raise PipelineFailure("CONFIG_INVALID","Axis vector must be nonzero")
    return v/n

def attach_geometry(obj:ObjectRecord,depth_raw:np.ndarray,K:CameraIntrinsics,cfg,T=None):
    if not obj.mask_path: raise PipelineFailure("DEPTH_INVALID",f"Object {obj.object_id} has no mask_path")
    mask=load_mask(obj.mask_path)
    if mask.shape!=depth_raw.shape[:2]: raise PipelineFailure("DEPTH_INVALID",f"Mask/depth shape mismatch for {obj.object_id}")
    scale=float(require(cfg,"stage3.depth_unit_to_m")); depth=depth_raw.astype(float)*scale
    valid=(depth>=float(require(cfg,"stage3.min_valid_depth_m")))&(depth<=float(require(cfg,"stage3.max_valid_depth_m")))
    pts,z=backproject(depth,mask&valid,K)
    pts,z=robust_filter(pts,z,float(require(cfg,"stage3.outlier_percentile_low")),float(require(cfg,"stage3.outlier_percentile_high")))
    if len(pts)<int(require(cfg,"stage3.min_points_per_object")): raise PipelineFailure("DEPTH_INVALID",f"Too few valid depth points for {obj.object_id}",{"points":len(pts)})
    robust_depth=float(np.median(z)); pts_scene=transform_points(pts,T); pts_scene=voxel_downsample_np(pts_scene,float(require(cfg,"stage3.voxel_size_m")))
    centroid,extent=bbox_extent(pts_scene); obj.centroid_xyz=centroid; obj.extent_xyz=extent; obj.visibility_ratio=min(1.0,len(pts)/max(1,int(mask.sum())))
    return pts_scene,robust_depth

def bbox_overlap_ratio(a,b):
    x1=max(a[0],b[0]);y1=max(a[1],b[1]);x2=min(a[2],b[2]);y2=min(a[3],b[3]); inter=max(0,x2-x1)*max(0,y2-y1)
    denom=min(max(1,(a[2]-a[0])*(a[3]-a[1])),max(1,(b[2]-b[0])*(b[3]-b[1])))
    return inter/denom

def relation_evidence(rel,target,ref_objs,geom,depths,cfg):
    t=np.asarray(target.centroid_xyz); refs=[np.asarray(r.centroid_xyz) for r in ref_objs]
    typ=rel.type; ev={}; score=0.; sat=None
    if typ in ("BEHIND","IN_FRONT_OF"):
        margin=float(require(cfg,"stage3.relation.depth_margin_m")); overlap=max(bbox_overlap_ratio(target.bbox_xyxy,r.bbox_xyxy) for r in ref_objs); dz=float(depths[target.object_id]-np.median([depths[r.object_id] for r in ref_objs])); ev={"depth_delta_m":dz,"image_overlap":overlap}
        direction=dz if typ=="BEHIND" else -dz
        score=float(np.clip(direction/(margin+1e-12),0,1))*float(np.clip(overlap/max(float(require(cfg,"stage3.relation.image_overlap_min")),1e-9),0,1)); sat=direction>=margin and overlap>=float(require(cfg,"stage3.relation.image_overlap_min"))
    elif typ in ("ABOVE","BELOW"):
        up=_norm(require(cfg,"stage3.scene_up_axis")); delta=float(np.dot(t-np.mean(refs,axis=0),up)); margin=float(require(cfg,"stage3.relation.depth_margin_m")); signed=delta if typ=="ABOVE" else -delta; score=float(np.clip(signed/(margin+1e-12),0,1)); sat=signed>=margin; ev={"axis_delta_m":delta}
    elif typ=="NEAR":
        d=min(pointcloud_distance(geom[target.object_id],geom[r.object_id]) for r in ref_objs); thr=float(require(cfg,"stage3.relation.near_distance_m")); score=float(np.clip(1-d/max(thr,1e-9),0,1)); sat=d<=thr; ev={"surface_distance_m":d}
    elif typ in ("BETWEEN","WEDGED_BETWEEN"):
        if len(refs)!=2: return RelationEvidence(relation=typ,target_id=target.object_id,reference_ids=[r.object_id for r in ref_objs],score=0,satisfied=False,numeric_evidence={})
        a,b=refs; ab=b-a; denom=float(np.dot(ab,ab)); u=float(np.dot(t-a,ab)/(denom+1e-12)); proj=a+np.clip(u,0,1)*ab; perp=float(np.linalg.norm(t-proj)); corridor=float(require(cfg,"stage3.relation.between_corridor_m")); between=(0<=u<=1 and perp<=corridor); score=float(np.clip(1-perp/max(corridor,1e-9),0,1)) if 0<=u<=1 else 0.; ev={"segment_parameter":u,"corridor_distance_m":perp}
        if typ=="WEDGED_BETWEEN":
            clear=min(pointcloud_distance(geom[target.object_id],geom[r.object_id]) for r in ref_objs); w=float(require(cfg,"stage3.relation.wedged_clearance_m")); wedged=between and clear<=w; score*=float(np.clip(1-clear/max(w,1e-9),0,1)); sat=wedged; ev["neighbor_clearance_m"]=clear
        else:sat=between
    elif typ in ("LEFT_OF","RIGHT_OF"):
        # Image-plane relation remains a 2D relation. Kept measurable and separate from depth claims.
        tx=(target.bbox_xyxy[0]+target.bbox_xyxy[2])/2; rx=np.mean([(r.bbox_xyxy[0]+r.bbox_xyxy[2])/2 for r in ref_objs]); signed=(rx-tx) if typ=="LEFT_OF" else (tx-rx); score=1.0 if signed>0 else 0.0; sat=signed>0; ev={"image_center_delta_px":float(signed)}
    elif typ in ("INSIDE",):
        # Requires container geometry definition not specified in source docs; do not invent it.
        score=0.; sat=None; ev={}
    else:
        score=0.; sat=None
    return RelationEvidence(relation=typ,target_id=target.object_id,reference_ids=[r.object_id for r in ref_objs],score=score,satisfied=sat,numeric_evidence=ev)

def access_score(target:ObjectRecord,all_objs:list[ObjectRecord],geom:dict,cfg):
    # Six-axis local corridor probe using observed point geometry only.
    tpts=geom[target.object_id]; c=np.asarray(target.centroid_xyz); probe=float(require(cfg,"stage3.access_probe_distance_m")); obs=np.vstack([geom[o.object_id] for o in all_objs if o.object_id!=target.object_id and o.object_id in geom]) if len(all_objs)>1 else np.empty((0,3))
    if len(obs)==0:return 1.0
    axes=np.vstack([np.eye(3),-np.eye(3)]); clear=[]
    for a in axes:
        # points in forward half-space and within a radial tube around the axis
        v=obs-c; longitudinal=v@a; radial=np.linalg.norm(v-longitudinal[:,None]*a,axis=1); hit=longitudinal[(longitudinal>0)&(longitudinal<probe)&(radial<probe*0.25)]
        clear.append(probe if len(hit)==0 else float(hit.min()))
    return float(max(clear)/probe)

def final_candidate_score(obj:ObjectRecord,rels:list[RelationEvidence],access:float,cfg):
    vals={"lang":obj.evidence.category,"attr":obj.evidence.attribute,"rel":np.mean([r.score for r in rels]) if rels else None,"visual":obj.evidence.visual_quality,"access":access}
    ws={k:float(require(cfg,f"stage3.decision.scoring_weights.{k}")) for k in vals}
    active=[k for k,v in vals.items() if v is not None]; denom=sum(ws[k] for k in active)
    if denom<=0:raise PipelineFailure("CONFIG_INVALID","Stage 3 active scoring weights must sum to >0")
    return float(sum(ws[k]*float(vals[k]) for k in active)/denom)

def run_stage3(task:TaskIR,s2:Stage2Result,depth_raw,K:CameraIntrinsics,cfg,T=None):
    candidates=s2.candidates; ref_lists=s2.references; all_objs=list(candidates)+[o for lst in ref_lists.values() for o in lst]
    geom={};depths={}; unique={o.object_id:o for o in all_objs}
    for o in unique.values(): geom[o.object_id],depths[o.object_id]=attach_geometry(o,depth_raw,K,cfg,T)
    results=[]
    for t in candidates:
        rels=[]
        for rel in task.relations:
            from itertools import product
            ids=rel.reference if isinstance(rel.reference,list) else [rel.reference]
            candidate_lists=[ref_lists.get(rid,[]) for rid in ids]
            if any(not x for x in candidate_lists):
                continue
            combos=list(product(*candidate_lists))
            scored=[relation_evidence(rel,t,list(combo),geom,depths,cfg) for combo in combos]
            # Preserve alternatives up to Stage 3, then choose the reference binding with strongest numeric relation evidence.
            rels.append(max(scored,key=lambda x:x.score))
        acc=access_score(t,all_objs,geom,cfg); score=final_candidate_score(t,rels,acc,cfg); results.append(TargetObject(object_record=t,relation_evidence=rels,access_score=acc,final_score=score,status="INSUFFICIENT_EVIDENCE"))
    results.sort(key=lambda x:x.final_score or 0,reverse=True)
    if not results: raise PipelineFailure("TARGET_NOT_FOUND","No Stage 3 candidates")
    top=results[0]; second=(results[1].final_score or 0) if len(results)>1 else 0.; margin=(top.final_score or 0)-second
    required_rel_ok=all(r.satisfied is True and r.score>=float(require(cfg,"stage3.decision.relation_min_score")) for r in top.relation_evidence) if task.relations else True
    if (top.final_score or 0)>=float(require(cfg,"stage3.decision.final_accept_score")) and margin>=float(require(cfg,"stage3.decision.final_margin")) and required_rel_ok: top.status="VERIFIED"
    elif margin<float(require(cfg,"stage3.decision.final_margin")): top.status="AMBIGUOUS"
    else: top.status="INSUFFICIENT_EVIDENCE"
    return top,geom
