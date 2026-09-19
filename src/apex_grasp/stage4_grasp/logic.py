from __future__ import annotations
import numpy as np
from apex_grasp.common.config import require
from apex_grasp.common.models import GraspCandidate,Stage4Result,TaskIR,TargetObject
from apex_grasp.common.errors import PipelineFailure
from .adapters import make_generator

def _unit(v):
    v=np.asarray(v,dtype=float); n=np.linalg.norm(v)
    if n<=1e-12: return None
    return v/n

def quat_to_rot(q):
    x,y,z,w=map(float,q); n=(x*x+y*y+z*z+w*w)**0.5
    if n<=1e-12: raise PipelineFailure("GRASP_INVALID","Zero quaternion")
    x,y,z,w=x/n,y/n,z/n,w/n
    return np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],[2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],[2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])

def _inside_box(points_local,center,half):
    return np.all(np.abs(points_local-np.asarray(center))<=np.asarray(half),axis=1)

def gripper_collision(position,q,jaw_width,obstacles,cfg):
    if len(obstacles)==0:return False,float('inf')
    R=quat_to_rot(q); local=(obstacles-np.asarray(position))@R
    fw=float(require(cfg,"stage4.gripper.finger_width_m")); fh=float(require(cfg,"stage4.gripper.finger_height_m")); fl=float(require(cfg,"stage4.gripper.finger_length_m")); pd=float(require(cfg,"stage4.gripper.palm_depth_m")); margin=float(require(cfg,"stage4.collision_margin_m"))
    # Convention: local x = jaw opening axis, local z = approach/depth axis. Exact robot convention must match generator adapter.
    left=_inside_box(local,[ jaw_width/2+fw/2,0,fl/2],[fw/2+margin,fh/2+margin,fl/2+margin])
    right=_inside_box(local,[-jaw_width/2-fw/2,0,fl/2],[fw/2+margin,fh/2+margin,fl/2+margin])
    palm=_inside_box(local,[0,0,-pd/2],[jaw_width/2+fw+margin,fh/2+margin,pd/2+margin])
    coll=bool(np.any(left|right|palm)); d=float(np.min(np.linalg.norm(obstacles-np.asarray(position),axis=1)))
    return coll,d

def contact_support(g,target_points,cfg):
    R=quat_to_rot(g.quaternion_xyzw); local=(target_points-np.asarray(g.position_xyz))@R
    fl=float(require(cfg,"stage4.gripper.finger_length_m")); fh=float(require(cfg,"stage4.gripper.finger_height_m")); w=g.jaw_width_m
    region=(np.abs(local[:,0])<=w/2)&(np.abs(local[:,1])<=fh)&(local[:,2]>=0)&(local[:,2]<=fl)
    return float(region.mean()) if len(local) else 0.0

def constraint_score(g,task:TaskIR,obstacles,cfg):
    pref=task.constraints.preferred_grasp_region
    if not pref:return 1.0
    if pref=="exposed_side":
        a=_unit(g.approach_vector)
        if a is None:return 0.0
        probe=float(require(cfg,"stage4.max_approach_distance_m")); p=np.asarray(g.position_xyz)-a*probe
        if len(obstacles)==0:return 1.0
        # Corridor clearance proxy. Exact meshes/FCL can be supplied via a custom validator in a later robot layer.
        v=obstacles-p; along=v@a; radial=np.linalg.norm(v-along[:,None]*a,axis=1); hit=along[(along>=0)&(along<=probe)&(radial<float(require(cfg,"stage4.safety_clearance_m")))]
        return 1.0 if len(hit)==0 else float(np.clip(hit.min()/probe,0,1))
    # Source docs name other possible constraints but do not define exact numeric rules; fail closed rather than invent.
    return 0.0

def validate_and_score(g:GraspCandidate,target_points,scene_points,task,cfg):
    pos=np.asarray(g.position_xyz); mn=np.asarray(require(cfg,"stage4.workspace.min_xyz_m"),float); mx=np.asarray(require(cfg,"stage4.workspace.max_xyz_m"),float)
    if np.any(pos<mn)|np.any(pos>mx): g.workspace_valid=False;g.rejection_reason="ROBOT_CONSTRAINT_VIOLATION";return g
    g.workspace_valid=True
    wmin=float(require(cfg,"stage4.gripper.width_min_m"))+float(require(cfg,"stage4.gripper.width_margin_m"));wmax=float(require(cfg,"stage4.gripper.width_max_m"))-float(require(cfg,"stage4.gripper.width_margin_m"))
    if not (wmin<=g.jaw_width_m<=wmax):g.rejection_reason="GRIPPER_WIDTH_INVALID";return g
    a=_unit(g.approach_vector)
    if a is None:g.rejection_reason="APPROACH_VECTOR_INVALID";return g
    # Remove points very near target cloud from obstacle proxy to avoid treating target contact as scene collision.
    obs=np.asarray(scene_points,float)
    coll,clear=gripper_collision(g.position_xyz,g.quaternion_xyzw,g.jaw_width_m,obs,cfg);g.clearance_m=clear
    if coll:g.collision_free=False;g.rejection_reason="FINAL_POSE_COLLISION";return g
    g.collision_free=True
    samples=int(require(cfg,"stage4.approach_samples"));dist=float(require(cfg,"stage4.max_approach_distance_m")); approach_ok=True;min_clear=clear
    for d in np.linspace(dist,0,samples):
        p=pos-a*d; c,cl=gripper_collision(p,g.quaternion_xyzw,g.jaw_width_m,obs,cfg);min_clear=min(min_clear,cl)
        if c:approach_ok=False;break
    g.approach_valid=approach_ok;g.clearance_m=min_clear
    if not approach_ok:g.rejection_reason="APPROACH_COLLISION";return g
    support=contact_support(g,target_points,cfg);g.target_contact_support=support
    if support<float(require(cfg,"stage4.min_contact_support")):g.rejection_reason="INSUFFICIENT_TARGET_CONTACT";return g
    g.visibility_score=float(np.clip(support/max(float(require(cfg,"stage4.min_contact_support")),1e-9),0,1));g.stability_score=float(np.clip((g.jaw_width_m-wmin)/max(wmax-wmin,1e-9),0,1));g.constraint_score=constraint_score(g,task,obs,cfg)
    if task.constraints.preferred_grasp_region and g.constraint_score<=0:g.rejection_reason="INSTRUCTION_CONSTRAINT_VIOLATION";return g
    safety=float(require(cfg,"stage4.safety_clearance_m")); clearance_score=float(np.clip((g.clearance_m or 0)/max(safety,1e-9),0,1)); approach_score=1.0 if g.approach_valid else 0.0
    vals={"model_quality":g.model_quality,"clearance":clearance_score,"approach":approach_score,"visibility":g.visibility_score,"stability":g.stability_score,"constraint":g.constraint_score}; ws={k:float(require(cfg,f"stage4.scoring_weights.{k}")) for k in vals};den=sum(ws.values())
    if den<=0:raise PipelineFailure("CONFIG_INVALID","Stage 4 scoring weights must sum to >0")
    g.final_score=float(sum(ws[k]*float(vals[k] or 0) for k in vals)/den);return g

def run_stage4(task:TaskIR,target:TargetObject,target_points,scene_points,cfg):
    if target.status!="VERIFIED":raise PipelineFailure("RELATION_UNCERTAIN",f"Stage 4 requires VERIFIED target, got {target.status}")
    gen=make_generator(cfg); raw=gen.generate(target_points,scene_points,target.object_record.object_id,{"task_ir":task.model_dump(),"target":target.model_dump()})
    raw=sorted(raw,key=lambda g:g.model_quality,reverse=True)[:int(require(cfg,"stage4.raw_top_k"))]
    good=[];bad=[]
    for g in raw:
        if g.target_object_id!=target.object_record.object_id:g.rejection_reason="WRONG_TARGET_ID";bad.append(g);continue
        g=validate_and_score(g,target_points,scene_points,task,cfg)
        (bad if g.rejection_reason else good).append(g)
    good.sort(key=lambda x:x.final_score or 0,reverse=True)
    if not good:raise PipelineFailure("NO_SAFE_GRASP","All grasp candidates failed validation",{"rejections":[g.rejection_reason for g in bad]})
    return Stage4Result(target_object_id=target.object_record.object_id,grasp_candidates=good[:int(require(cfg,"stage4.output_top_n"))],rejected_candidates=bad)
