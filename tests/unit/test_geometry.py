import numpy as np
from apex_grasp.common.models import CameraIntrinsics
from apex_grasp.stage3_3d.geometry import backproject,transform_points

def test_backproject_center():
    d=np.zeros((3,3));d[1,1]=2.0;m=d>0;K=CameraIntrinsics(fx=100,fy=100,cx=1,cy=1)
    p,z=backproject(d,m,K)
    assert np.allclose(p,[[0,0,2]])

def test_transform():
    T=np.eye(4);T[:3,3]=[1,2,3]
    assert np.allclose(transform_points(np.array([[0.,0.,1.]]),T),[[1,2,4]])
