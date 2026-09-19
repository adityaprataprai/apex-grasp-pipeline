import numpy as np
from apex_grasp.stage4_grasp.logic import quat_to_rot

def test_identity_quaternion():
    assert np.allclose(quat_to_rot([0,0,0,1]),np.eye(3))
