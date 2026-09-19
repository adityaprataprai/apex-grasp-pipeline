from __future__ import annotations
import argparse,numpy as np
from apex_grasp.common.config import load_yaml
from apex_grasp.common.io import read_json,write_json
from apex_grasp.common.models import TaskIR,TargetObject
from .logic import run_stage4

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--config",required=True);ap.add_argument("--task-ir",required=True);ap.add_argument("--target",required=True);ap.add_argument("--target-points",required=True);ap.add_argument("--scene-points",required=True);ap.add_argument("--output",required=True);a=ap.parse_args()
    result=run_stage4(TaskIR.model_validate(read_json(a.task_ir)),TargetObject.model_validate({k:v for k,v in read_json(a.target).items() if k!="target_point_cloud_path"}),np.load(a.target_points),np.load(a.scene_points),load_yaml(a.config));write_json(a.output,result.model_dump())
if __name__=="__main__":main()
