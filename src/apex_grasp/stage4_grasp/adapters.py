from __future__ import annotations
import importlib,json
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from apex_grasp.common.config import require
from apex_grasp.common.errors import PipelineFailure
from apex_grasp.common.models import GraspCandidate

@dataclass
class PythonCallableGenerator:
    module:str; function:str
    def generate(self,target_points,scene_points,target_object_id,context):
        fn=getattr(importlib.import_module(self.module),self.function)
        raw=fn(target_points=target_points,scene_points=scene_points,target_object_id=target_object_id,context=context)
        return [GraspCandidate.model_validate(x) for x in raw]
@dataclass
class ReplayGenerator:
    path:str
    def generate(self,target_points,scene_points,target_object_id,context):
        data=json.loads(Path(self.path).read_text(encoding="utf-8"))
        return [GraspCandidate.model_validate(x) for x in data.get("grasps",[])]

def make_generator(cfg):
    b=require(cfg,"stage4.generator_backend")
    if b=="python_callable": return PythonCallableGenerator(require(cfg,"stage4.python_callable.module"),require(cfg,"stage4.python_callable.function"))
    if b=="replay": return ReplayGenerator(require(cfg,"stage4.replay_predictions"))
    raise PipelineFailure("CONFIG_INVALID",f"Unsupported stage4.generator_backend={b}. No GraspNet fork/API is silently selected; connect your chosen implementation through python_callable.")
