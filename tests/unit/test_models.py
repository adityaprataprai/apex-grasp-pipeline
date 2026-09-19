import pytest
from apex_grasp.common.models import TaskIR

def base():
    return {"action":"GRASP","target":{"category":"carton","attributes":{}},"references":[{"id":"ref_1","category":"bottle","attributes":{}}],"relations":[{"type":"BEHIND","target":"TARGET","reference":"ref_1"}],"constraints":{},"sequence":1,"ambiguity":0.1,"unresolved":[],"provenance":{"parser_version":"test","original_instruction":"x"}}

def test_task_ir_valid():
    assert TaskIR.model_validate(base()).target.category=="carton"

def test_unknown_reference_rejected():
    d=base();d['relations'][0]['reference']='ref_missing'
    with pytest.raises(Exception):TaskIR.model_validate(d)

def test_between_requires_two():
    d=base();d['relations'][0]={"type":"BETWEEN","target":"TARGET","reference":"ref_1"}
    with pytest.raises(Exception):TaskIR.model_validate(d)
