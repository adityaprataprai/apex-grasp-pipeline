from __future__ import annotations
import argparse
from apex_grasp.common.config import load_yaml, require
from apex_grasp.common.io import write_json
from apex_grasp.common.models import TaskIR
from .parser import make_parser

def run_stage1(instruction:str,cfg:dict):
    ontology=load_yaml(require(cfg,"stage1.ontology_path"))
    raw=make_parser(cfg,ontology).parse(instruction)
    return TaskIR.model_validate(raw)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--config",required=True); ap.add_argument("--instruction",required=True); ap.add_argument("--output",required=True); a=ap.parse_args()
    result=run_stage1(a.instruction,load_yaml(a.config)); write_json(a.output,result.model_dump())
if __name__=="__main__": main()
