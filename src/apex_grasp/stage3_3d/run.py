from __future__ import annotations
import argparse, numpy as np
from apex_grasp.common.config import load_yaml
from apex_grasp.common.io import read_json,write_json,load_depth
from apex_grasp.common.models import TaskIR,Stage2Result,CameraIntrinsics
from .logic import run_stage3

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--config",required=True);ap.add_argument("--task-ir",required=True);ap.add_argument("--stage2",required=True);ap.add_argument("--depth",required=True);ap.add_argument("--intrinsics",required=True);ap.add_argument("--camera-to-scene");ap.add_argument("--output",required=True);a=ap.parse_args()
    cfg=load_yaml(a.config); task=TaskIR.model_validate(read_json(a.task_ir)); s2=Stage2Result.model_validate(read_json(a.stage2)); K=CameraIntrinsics.model_validate(read_json(a.intrinsics));T=np.asarray(read_json(a.camera_to_scene),dtype=float) if a.camera_to_scene else None
    result,geom=run_stage3(task,s2,load_depth(a.depth),K,cfg,T)
    # Save target point cloud explicitly for Stage 4.
    pc_path=a.output+".target_points.npy"; np.save(pc_path,geom[result.object_record.object_id]); out=result.model_dump();out["target_point_cloud_path"]=pc_path;write_json(a.output,out)
if __name__=="__main__":main()
