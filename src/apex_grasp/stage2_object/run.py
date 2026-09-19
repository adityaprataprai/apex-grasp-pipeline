from __future__ import annotations
import argparse
from apex_grasp.common.config import load_yaml,require
from apex_grasp.common.io import read_json,write_json,load_rgb
from apex_grasp.common.models import TaskIR
from .logic import run_stage2

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--config",required=True); ap.add_argument("--image",required=True); ap.add_argument("--task-ir",required=True); ap.add_argument("--output",required=True); a=ap.parse_args()
    cfg=load_yaml(a.config); task=TaskIR.model_validate(read_json(a.task_ir)); result=run_stage2(load_rgb(a.image),task,cfg,require(cfg,"project.artifact_dir")); write_json(a.output,result.model_dump())
if __name__=="__main__": main()
