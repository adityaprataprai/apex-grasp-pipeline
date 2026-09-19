from apex_grasp.stage2_object.logic import iou,weighted

def test_iou_identity(): assert abs(iou([0,0,10,10],[0,0,10,10])-1)<1e-9

def test_weight_redistribution():
    e={'category':1.0,'attribute':None,'reference_2d':0.5}
    w={'category':0.3,'attribute':0.2,'reference_2d':0.2}
    assert abs(weighted(e,w)-(0.3*1+0.2*0.5)/0.5)<1e-9
