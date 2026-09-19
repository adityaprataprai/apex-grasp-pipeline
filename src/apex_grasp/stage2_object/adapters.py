from __future__ import annotations
import importlib, json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol
import numpy as np
from apex_grasp.common.errors import PipelineFailure
from apex_grasp.common.config import require

class Detector(Protocol):
    def detect(self, image: np.ndarray, prompts: list[str]) -> list[dict[str,Any]]: ...
class Segmenter(Protocol):
    def segment(self, image: np.ndarray, boxes_xyxy: list[list[float]]) -> list[np.ndarray]: ...
class AttributeVerifier(Protocol):
    def score(self, image: np.ndarray, mask: np.ndarray, requested: dict[str,Any]) -> tuple[float,dict[str,Any]]: ...

@dataclass
class PythonCallableDetector:
    module:str; function:str
    def detect(self,image,prompts): return getattr(importlib.import_module(self.module),self.function)(image,prompts)
@dataclass
class PythonCallableSegmenter:
    module:str; function:str
    def segment(self,image,boxes_xyxy): return getattr(importlib.import_module(self.module),self.function)(image,boxes_xyxy)
@dataclass
class PythonCallableAttributeVerifier:
    module:str; function:str
    def score(self,image,mask,requested): return getattr(importlib.import_module(self.module),self.function)(image,mask,requested)

@dataclass
class ReplayDetector:
    path:str
    def __post_init__(self): self.data=json.loads(Path(self.path).read_text(encoding="utf-8"))
    def detect(self,image,prompts): return self.data.get("detections",[])
@dataclass
class ReplaySegmenter:
    path:str
    def __post_init__(self): self.data=json.loads(Path(self.path).read_text(encoding="utf-8"))
    def segment(self,image,boxes_xyxy):
        masks=[]
        from apex_grasp.common.io import load_mask
        for item in self.data.get("detections",[])[:len(boxes_xyxy)]: masks.append(load_mask(item["mask_path"]))
        return masks

@dataclass
class GroundingDINODetector:
    config_path:str; checkpoint_path:str; box_threshold:float; text_threshold:float
    def __post_init__(self):
        try:
            from groundingdino.util.inference import Model
            self.model=Model(model_config_path=self.config_path,model_checkpoint_path=self.checkpoint_path)
        except Exception as e:
            raise PipelineFailure("MODEL_IMPORT_FAILED",f"Grounding DINO adapter expects groundingdino.util.inference.Model API: {e}")
    def detect(self,image,prompts):
        # One call per prompt preserves prompt-specific evidence and avoids hidden prompt aggregation.
        out=[]
        for prompt in prompts:
            try:
                det=self.model.predict_with_caption(image=image,caption=prompt,box_threshold=self.box_threshold,text_threshold=self.text_threshold)
                boxes=getattr(det,"xyxy",None); conf=getattr(det,"confidence",None)
                if boxes is None:
                    boxes, logits, phrases = det
                    conf=logits
                for i,b in enumerate(np.asarray(boxes)):
                    out.append({"bbox_xyxy":[float(x) for x in b],"category_score":float(np.asarray(conf)[i]),"prompt_score":float(np.asarray(conf)[i]),"prompt":prompt,"raw_id":f"{prompt}:{i}"})
            except Exception as e: raise PipelineFailure("DETECTOR_FAILED",f"Grounding DINO inference failed: {e}")
        return out

@dataclass
class SAM2Segmenter:
    model_cfg:str; checkpoint_path:str; device:str
    def __post_init__(self):
        try:
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor
            model=build_sam2(self.model_cfg,self.checkpoint_path,device=self.device)
            self.predictor=SAM2ImagePredictor(model)
        except Exception as e:
            raise PipelineFailure("MODEL_IMPORT_FAILED",f"SAM 2 adapter expects sam2.build_sam/build_sam2 and SAM2ImagePredictor APIs: {e}")
    def segment(self,image,boxes_xyxy):
        self.predictor.set_image(image)
        masks=[]
        for box in boxes_xyxy:
            try:
                m,s,_=self.predictor.predict(box=np.array(box,dtype=np.float32),multimask_output=False)
                masks.append(np.asarray(m[0],dtype=bool))
            except Exception as e: raise PipelineFailure("SEGMENTER_FAILED",f"SAM 2 inference failed: {e}")
        return masks

@dataclass
class BasicColorVerifier:
    prototypes:dict[str,Any]
    def score(self,image,mask,requested):
        color=requested.get("color")
        if not color: return 1.0,{"note":"no requested color"}
        if color not in self.prototypes: raise PipelineFailure("CONFIG_REQUIRED",f"No calibrated color prototype configured for '{color}'")
        import cv2
        pix=image[mask]
        if len(pix)==0: return 0.0,{"reason":"empty_mask"}
        hsv=cv2.cvtColor(pix.reshape(-1,1,3).astype(np.uint8),cv2.COLOR_RGB2HSV).reshape(-1,3)
        median=np.median(hsv,axis=0)
        proto=np.asarray(self.prototypes[color],dtype=float)
        # User supplies normalization scales with prototype if desired; simple bounded Euclidean proxy.
        dist=float(np.linalg.norm((median-proto)/np.array([180.0,255.0,255.0])))
        return max(0.0,1.0-dist),{"median_hsv":median.tolist(),"prototype":proto.tolist()}

def make_detector(cfg):
    b=require(cfg,"stage2.detector_backend")
    if b=="replay": return ReplayDetector(require(cfg,"stage2.replay_predictions"))
    if b=="python_callable": return PythonCallableDetector(require(cfg,"stage2.python_callable.detector_module"),require(cfg,"stage2.python_callable.detector_function"))
    if b=="grounding_dino": return GroundingDINODetector(require(cfg,"stage2.detector_config"),require(cfg,"stage2.detector_checkpoint"),float(require(cfg,"stage2.box_threshold")),float(require(cfg,"stage2.text_threshold")))
    raise PipelineFailure("CONFIG_INVALID",f"Unsupported stage2.detector_backend={b}")
def make_segmenter(cfg):
    b=require(cfg,"stage2.segmenter_backend")
    if b=="replay": return ReplaySegmenter(require(cfg,"stage2.replay_predictions"))
    if b=="python_callable": return PythonCallableSegmenter(require(cfg,"stage2.python_callable.segmenter_module"),require(cfg,"stage2.python_callable.segmenter_function"))
    if b=="sam2": return SAM2Segmenter(require(cfg,"stage2.sam2_model_cfg"),require(cfg,"stage2.sam2_checkpoint"),require(cfg,"project.device"))
    raise PipelineFailure("CONFIG_INVALID",f"Unsupported stage2.segmenter_backend={b}")
def make_attribute_verifier(cfg):
    b=require(cfg,"stage2.attribute_backend")
    if b=="none": return None
    if b=="python_callable": return PythonCallableAttributeVerifier(require(cfg,"stage2.python_callable.attribute_module"),require(cfg,"stage2.python_callable.attribute_function"))
    if b=="basic_color":
        import yaml
        d=yaml.safe_load(Path(require(cfg,"stage2.color_prototypes_path")).read_text(encoding="utf-8")) or {}
        return BasicColorVerifier(d.get("prototypes",{}))
    raise PipelineFailure("CONFIG_INVALID",f"Unsupported stage2.attribute_backend={b}")
