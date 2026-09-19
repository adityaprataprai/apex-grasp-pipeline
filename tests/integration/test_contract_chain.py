from apex_grasp.common.models import TaskIR,ObjectRecord,Stage2Result,TargetObject,Stage4Result,GraspCandidate

def test_contract_chain_serializes():
    task=TaskIR.model_validate({"action":"GRASP","target":{"category":"carton","attributes":{}},"references":[],"relations":[],"constraints":{},"ambiguity":0.05,"unresolved":[],"provenance":{"parser_version":"t","original_instruction":"grab carton"}})
    o=ObjectRecord(object_id='obj_01',category='carton',bbox_xyxy=[0,0,10,10])
    s2=Stage2Result(target_object=o,candidates=[o],needs_3d_disambiguation=False)
    t=TargetObject(object_record=o,status='VERIFIED',final_score=1.0)
    g=GraspCandidate(target_object_id='obj_01',position_xyz=[0,0,0],quaternion_xyzw=[0,0,0,1],approach_vector=[0,0,1],jaw_width_m=.05)
    s4=Stage4Result(target_object_id='obj_01',grasp_candidates=[g])
    assert s4.model_dump()['target_object_id']==s2.target_object.object_id==t.object_record.object_id==task.target.category.replace('carton','obj_01')
