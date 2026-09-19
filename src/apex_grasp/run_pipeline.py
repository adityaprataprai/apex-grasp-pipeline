from __future__ import annotations
import argparse, time, traceback
from pathlib import Path
import numpy as np
from apex_grasp.common.config import load_yaml, require
from apex_grasp.common.io import load_rgb, load_depth, read_json, write_json, load_mask
from apex_grasp.common.models import CameraIntrinsics
from apex_grasp.common.errors import PipelineFailure
from apex_grasp.stage1_language.run import run_stage1
from apex_grasp.stage2_object.logic import run_stage2
from apex_grasp.stage3_3d.logic import run_stage3
from apex_grasp.stage3_3d.geometry import backproject, transform_points, robust_filter, voxel_downsample_np
from apex_grasp.stage4_grasp.logic import run_stage4

def build_obstacle_cloud(depth_raw, target_mask, K, cfg, T=None):
    scale=float(require(cfg,"stage3.depth_unit_to_m")); depth=depth_raw.astype(float)*scale
    valid=(depth>=float(require(cfg,"stage3.min_valid_depth_m")))&(depth<=float(require(cfg,"stage3.max_valid_depth_m")))
    mask=valid & (~target_mask)
    pts,z=backproject(depth,mask,K)
    pts,z=robust_filter(pts,z,float(require(cfg,"stage3.outlier_percentile_low")),float(require(cfg,"stage3.outlier_percentile_high")))
    pts=transform_points(pts,T)
    return voxel_downsample_np(pts,float(require(cfg,"stage3.voxel_size_m")))

def run(args):
    cfg=load_yaml(args.config); outdir=Path(require(cfg,"project.artifact_dir")); outdir.mkdir(parents=True,exist_ok=True)
    rgb=load_rgb(args.image); depth=load_depth(args.depth); K=CameraIntrinsics.model_validate(read_json(args.intrinsics)); T=np.asarray(read_json(args.camera_to_scene),dtype=float) if args.camera_to_scene else None
    trace={"request_id":args.request_id,"scene_id":args.scene_id,"timings_s":{},"status":"RUNNING"}
    t=time.perf_counter(); task=run_stage1(args.instruction,cfg); trace["timings_s"]["stage1"]=time.perf_counter()-t; write_json(outdir/f"{args.request_id}.stage1.json",task.model_dump())
    t=time.perf_counter(); s2=run_stage2(rgb,task,cfg,outdir/f"{args.request_id}_stage2"); trace["timings_s"]["stage2"]=time.perf_counter()-t; write_json(outdir/f"{args.request_id}.stage2.json",s2.model_dump())
    t=time.perf_counter(); target,geom=run_stage3(task,s2,depth,K,cfg,T); trace["timings_s"]["stage3"]=time.perf_counter()-t; target_points=geom[target.object_record.object_id]; np.save(outdir/f"{args.request_id}.target_points.npy",target_points); write_json(outdir/f"{args.request_id}.stage3.json",target.model_dump())
    if target.status!="VERIFIED": raise PipelineFailure("RELATION_UNCERTAIN",f"Stage 3 target status is {target.status}")
    mask=load_mask(target.object_record.mask_path); scene_points=build_obstacle_cloud(depth,mask,K,cfg,T); np.save(outdir/f"{args.request_id}.scene_obstacles.npy",scene_points)
    t=time.perf_counter(); s4=run_stage4(task,target,target_points,scene_points,cfg); trace["timings_s"]["stage4"]=time.perf_counter()-t; write_json(outdir/f"{args.request_id}.stage4.json",s4.model_dump())
    trace["status"]="SUCCESS"; trace["total_s"]=sum(trace["timings_s"].values()); trace["final"]={"target_object_id":target.object_record.object_id,"selected_grasp":s4.grasp_candidates[0].model_dump()}; write_json(outdir/f"{args.request_id}.trace.json",trace); return trace

def main():
    ap=argparse.ArgumentParser(description="End-to-end Apex grasp pipeline; performs no downloads and requires explicit configuration.")
    ap.add_argument('--config',required=True);ap.add_argument('--request-id',required=True);ap.add_argument('--scene-id',required=True);ap.add_argument('--image',required=True);ap.add_argument('--depth',required=True);ap.add_argument('--intrinsics',required=True);ap.add_argument('--camera-to-scene');ap.add_argument('--instruction',required=True);a=ap.parse_args()
    try:
        trace=run(a); print(f"SUCCESS: {trace['request_id']}")
    except PipelineFailure as e:
        cfg=load_yaml(a.config); outdir=Path(cfg.get('project',{}).get('artifact_dir') or '.');outdir.mkdir(parents=True,exist_ok=True);write_json(outdir/f"{a.request_id}.failure.json",e.to_dict());print(f"FAILURE {e.code}: {e.message}");raise SystemExit(2)
if __name__=='__main__':main()
