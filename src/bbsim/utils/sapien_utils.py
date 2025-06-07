# Sapien
import sapien

import numpy as np
import transforms3d as t3d

def look_at(eye: np.ndarray, target: np.ndarray, up: np.ndarray) -> sapien.Pose:
    """
    Create a sapien.Pose that looks at a target point from an eye position with a specified up vector.
    
    Args:
        eye (np.ndarray): The position of the camera (eye).
        target (np.ndarray): The point to look at.
        up (np.ndarray): The up vector for the camera orientation.
        
    Returns:
        sapien.Pose: The pose of the camera looking at the target.
    """
    forward = target - eye
    forward = forward / (np.linalg.norm(forward) + 1e-6)

    up = up / (np.linalg.norm(up) + 1e-6)
    
    left = np.cross(up, forward)
    left = left / (np.linalg.norm(left) + 1e-6)
    
    rotation_matrix = np.stack([forward, -left, -up], axis=1)

    return sapien.Pose(
        p=eye,
        q=t3d.quaternions.mat2quat(rotation_matrix)
    )