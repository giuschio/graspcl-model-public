import hashlib

import matplotlib.colors as mcolors
import numpy as np
import open3d as o3d


def create_cylinder_between(p1, p2, radius=0.005, color=(1, 0, 0)):
    cylinder = o3d.geometry.TriangleMesh.create_cylinder(radius=radius, height=np.linalg.norm(p2 - p1))
    cylinder.compute_vertex_normals()

    direction = p2 - p1
    mid_point = (p1 + p2) / 2
    direction_norm = direction / np.linalg.norm(direction)

    z_axis = np.array([0, 0, 1])
    axis = np.cross(z_axis, direction_norm)
    if np.linalg.norm(axis) < 1e-6:
        rotation = np.eye(3)
    else:
        axis = axis / np.linalg.norm(axis)
        angle = np.arccos(np.clip(np.dot(z_axis, direction_norm), -1.0, 1.0))
        rotation = o3d.geometry.get_rotation_matrix_from_axis_angle(axis * angle)

    cylinder.rotate(rotation, center=np.zeros(3))
    cylinder.translate(mid_point)
    cylinder.paint_uniform_color(color)
    return cylinder


def get_colormap():
    colors = [(1, 0, 0), (1, 1, 0), (0, 1, 0)]
    colormap = mcolors.LinearSegmentedColormap.from_list("red_yellow_green", colors, N=100)

    def map_color(value):
        color = colormap(value)
        return list(color[:3])

    return map_color


def string_to_color(value):
    if value == "unclassified":
        return (0, 0, 0)
    digest = hashlib.md5(value.encode()).digest()
    return digest[0] / 255.0, digest[1] / 255.0, digest[2] / 255.0


def visualize_scene_pcd_util(
    pcd,
    grasps,
    visualize_ref_system=True,
    color_key="success",
    vis_radius=0.001,
    visualize_grasp_rotation=False,
):
    """Return Open3D geometries for visualizing a point cloud and grasp dicts."""
    direct_color = None
    if color_key.startswith("color-"):
        direct_color = list(mcolors.to_rgb(color_key[len("color-"):]))
    elif color_key == "type":
        color_map = string_to_color
    else:
        color_map = get_colormap()

    geometries = [pcd]

    for grasp in grasps:
        pose = grasp["pose"]
        p1 = np.array(grasp["point1"])
        p2 = np.array(grasp["point2"])
        z_dir = pose[:3, 2]

        p1_end = p1 - z_dir * 0.035
        p2_end = p2 - z_dir * 0.035
        color = direct_color if direct_color is not None else color_map(float(grasp.get(color_key, 1.0)))
        try:
            side1 = create_cylinder_between(p1, p1_end, vis_radius, color)
            side2 = create_cylinder_between(p2, p2_end, vis_radius, color)
            side3 = create_cylinder_between(
                0.5 * (p2_end + p1_end),
                0.5 * (p2_end + p1_end) - z_dir * 0.035,
                vis_radius,
                color,
            )
            bar = create_cylinder_between(p1_end, p2_end, vis_radius, color)
            geometries.extend([side1, side2, side3, bar])
        except Exception:
            pass

        if visualize_grasp_rotation:
            grasp_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.03)
            grasp_frame.transform(pose)
            geometries.append(grasp_frame)

    if visualize_ref_system:
        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05)
        geometries.append(frame)

    return geometries


def visualize_scene_pcd(
    pcd,
    grasps,
    visualize_ref_system=True,
    color_key="success",
    window_name="Grasps",
    visualize_grasp_rotation=False,
):
    geometries = visualize_scene_pcd_util(
        pcd,
        grasps,
        visualize_ref_system=visualize_ref_system,
        color_key=color_key,
        visualize_grasp_rotation=visualize_grasp_rotation,
        vis_radius=0.001,
    )
    o3d.visualization.draw_geometries(geometries, window_name=window_name)
