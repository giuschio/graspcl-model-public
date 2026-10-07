import random
import numpy as np
import open3d as o3d

import copy
import math
from typing import List, Dict, Any

def _rz(angle: float) -> np.ndarray:
    ca, sa = math.cos(angle), math.sin(angle)
    return np.array([[ca, -sa, 0.0],
                     [sa,  ca, 0.0],
                     [0.0, 0.0, 1.0]], dtype=np.float32)

def _rand_in_range(lo, hi) -> float:
    return random.uniform(lo, hi)


def augment(
    cloud_o3d: o3d.geometry.PointCloud,
    grasps: List[Dict[str, Any]],
    angle_range_rad=(-math.pi, math.pi),
    max_shift=(0.02, 0.02, 0.02),
    pivot=(0.15, 0.15, 0.0)
):
    """
    Rotate about world-Z around `pivot` and translate; apply to Open3D cloud + grasp dicts.

    Transforms:
      - pose   := A @ pose
      - point1 := (A @ [x,y,z,1])[:3]
      - point2 := (A @ [x,y,z,1])[:3]
    where A encodes a rotation about Z around `pivot` and a random translation within `max_shift`.
    """

    # --- sample rotation + translation (reuse your helpers) ---
    a  = _rand_in_range(angle_range_rad[0], angle_range_rad[1])
    Rz = _rz(a)
    tx = _rand_in_range(-max_shift[0], max_shift[0])
    ty = _rand_in_range(-max_shift[1], max_shift[1])
    tz = _rand_in_range(-max_shift[2], max_shift[2])

    p = np.asarray(pivot, dtype=np.float32)
    trans = p - (Rz @ p) + np.array([tx, ty, tz], dtype=np.float32)

    A = np.eye(4, dtype=np.float32)
    A[:3, :3] = Rz
    A[:3, 3]  = trans

    # --- transform cloud (clone first) ---
    cloud_aug = copy.deepcopy(cloud_o3d)
    cloud_aug.transform(A)

    # --- helper to transform a 3D point with A ---
    def _tf_point(pt):
        v = np.asarray(pt, dtype=np.float32).reshape(3)
        vh = np.concatenate([v, np.array([1.0], dtype=np.float32)])
        return (A @ vh)[:3]

    # --- transform grasps (deep-copy, keep numpy types for geometry) ---
    aug_grasps: List[Dict[str, Any]] = []
    for g in grasps:
        g_new = dict(g)  # shallow copy of dict
        # pose
        if "pose" in g and g["pose"] is not None:
            pose = np.asarray(g["pose"], dtype=np.float32).reshape(4, 4)
            g_new["pose"] = (A @ pose).astype(np.float32)

        # points
        if "point1" in g and g["point1"] is not None:
            g_new["point1"] = _tf_point(g["point1"])
        if "point2" in g and g["point2"] is not None:
            g_new["point2"] = _tf_point(g["point2"])

        # width/type/score are rigid-invariant; keep as-is
        aug_grasps.append(g_new)

    return cloud_aug, aug_grasps