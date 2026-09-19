from __future__ import annotations
from typing import Any, Literal
from pydantic import BaseModel, Field, model_validator

class Attributes(BaseModel):
    color: str | None = None
    condition: str | None = None
    size: str | None = None
    material: str | None = None
    transparency: str | None = None

class TargetDescription(BaseModel):
    category: str | None = None
    attributes: Attributes = Field(default_factory=Attributes)

class ReferenceDescription(BaseModel):
    id: str
    category: str | None = None
    attributes: Attributes = Field(default_factory=Attributes)

class Relation(BaseModel):
    type: str
    target: str = "TARGET"
    reference: str | list[str]

class Constraints(BaseModel):
    preferred_grasp_region: str | None = None
    max_approach_collision: str | None = None
    avoid_object_ids: list[str] = Field(default_factory=list)
    preferred_direction: list[float] | None = None

class Provenance(BaseModel):
    parser_version: str
    original_instruction: str
    model: str | None = None

class TaskIR(BaseModel):
    action: str
    target: TargetDescription
    references: list[ReferenceDescription] = Field(default_factory=list)
    relations: list[Relation] = Field(default_factory=list)
    constraints: Constraints = Field(default_factory=Constraints)
    sequence: int = 1
    ambiguity: float = Field(ge=0.0, le=1.0)
    unresolved: list[str] = Field(default_factory=list)
    provenance: Provenance

    @model_validator(mode="after")
    def relation_refs_exist(self):
        ids = {r.id for r in self.references}
        for rel in self.relations:
            refs = rel.reference if isinstance(rel.reference, list) else [rel.reference]
            for rid in refs:
                if rid not in ids:
                    raise ValueError(f"Relation references unknown id: {rid}")
            if rel.type == "BETWEEN" and len(refs) != 2:
                raise ValueError("BETWEEN requires exactly two reference ids")
        if self.unresolved and self.ambiguity <= 0:
            raise ValueError("unresolved entries require ambiguity > 0")
        return self

class ObjectEvidence(BaseModel):
    category: float | None = None
    attribute: float | None = None
    reference_2d: float | None = None
    visual_quality: float | None = None
    prompt_similarity: float | None = None

class ObjectRecord(BaseModel):
    object_id: str
    category: str | None = None
    attributes: dict[str, Any] = Field(default_factory=dict)
    bbox_xyxy: list[float]
    mask_path: str | None = None
    mask_rle: str | None = None
    confidence_2d: float = 0.0
    evidence: ObjectEvidence = Field(default_factory=ObjectEvidence)
    point_cloud_path: str | None = None
    centroid_xyz: list[float] | None = None
    extent_xyz: list[float] | None = None
    visibility_ratio: float | None = None
    occlusion_score: float | None = None
    provenance: dict[str, Any] = Field(default_factory=dict)

class Stage2Result(BaseModel):
    target_object: ObjectRecord | None
    candidates: list[ObjectRecord]
    references: dict[str, list[ObjectRecord]] = Field(default_factory=dict)
    needs_3d_disambiguation: bool
    decision_margin: float | None = None

class CameraIntrinsics(BaseModel):
    fx: float
    fy: float
    cx: float
    cy: float

class RelationEvidence(BaseModel):
    relation: str
    target_id: str
    reference_ids: list[str]
    score: float = Field(ge=0.0, le=1.0)
    satisfied: bool | None = None
    numeric_evidence: dict[str, float] = Field(default_factory=dict)

class TargetObject(BaseModel):
    object_record: ObjectRecord
    relation_evidence: list[RelationEvidence] = Field(default_factory=list)
    access_score: float | None = None
    final_score: float | None = None
    status: Literal["VERIFIED", "AMBIGUOUS", "INSUFFICIENT_EVIDENCE"]

class GraspCandidate(BaseModel):
    target_object_id: str
    position_xyz: list[float]
    quaternion_xyzw: list[float]
    approach_vector: list[float]
    jaw_width_m: float
    model_quality: float = 0.0
    clearance_m: float | None = None
    collision_free: bool | None = None
    workspace_valid: bool | None = None
    approach_valid: bool | None = None
    target_contact_support: float | None = None
    visibility_score: float | None = None
    stability_score: float | None = None
    constraint_score: float | None = None
    final_score: float | None = None
    rejection_reason: str | None = None
    provenance: dict[str, Any] = Field(default_factory=dict)

class Stage4Result(BaseModel):
    target_object_id: str
    grasp_candidates: list[GraspCandidate]
    rejected_candidates: list[GraspCandidate] = Field(default_factory=list)
