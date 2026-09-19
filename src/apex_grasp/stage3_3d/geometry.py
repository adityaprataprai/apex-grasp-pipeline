from __future__ import annotations
import numpy as np
from apex_grasp.common.models import CameraIntrinsics
from apex_grasp.common.errors import PipelineFailure

def backproject(depth_m:np.ndarray,mask:np.ndarray,K:CameraIntrinsics)->tuple[np.ndarray,np.ndarray]:
    ys,xs=np.nonzero(mask & np.isfinite(depth_m) & (depth_m>0))
    z=depth_m[ys,xs]
    x=(xs-K.cx)*z/K.fx; y=(ys-K.cy)*z/K.fy
    return np.column_stack([x,y,z]).astype(np.float64), z

def transform_points(points:np.ndarray,T:np.ndarray|None)->np.ndarray:
    if T is None: return points
    if T.shape!=(4,4): raise PipelineFailure("CALIBRATION_INVALID","camera_to_scene transform must be 4x4")
    h=np.column_stack([points,np.ones(len(points))]); return (h@T.T)[:,:3]

def robust_filter(points:np.ndarray,depths:np.ndarray,low:float,high:float)->tuple[np.ndarray,np.ndarray]:
    if len(points)==0:return points,depths
    lo,hi=np.percentile(depths,[low,high]); keep=(depths>=lo)&(depths<=hi); return points[keep],depths[keep]

def voxel_downsample_np(points:np.ndarray,voxel:float)->np.ndarray:
    if voxel<=0 or len(points)==0:return points
    keys=np.floor(points/voxel).astype(np.int64)
    _,idx=np.unique(keys,axis=0,return_index=True)
    return points[np.sort(idx)]

def bbox_extent(points):
    if len(points)==0:return None,None
    mn=points.min(axis=0); mx=points.max(axis=0); return ((mn+mx)/2).tolist(),(mx-mn).tolist()

def pointcloud_distance(a:np.ndarray,b:np.ndarray)->float:
    if len(a)==0 or len(b)==0:return float('inf')
    # Bounded-memory exact nearest distance for evaluation-sized object point clouds.
    best=float('inf')
    chunk=2048
    for i in range(0,len(a),chunk):
        d=np.linalg.norm(a[i:i+chunk,None,:]-b[None,:,:],axis=2)
        best=min(best,float(d.min()))
    return best
