import os
import numpy as np
import open3d as o3d

from grasp_learning_cl.grasp_datagen.datasets import SceneGraspDataHandler
from scipy.spatial.transform import Rotation as R
from grasp_learning_cl.visualization import visualize_scene_pcd_util

def create_grasp_visualization(p1, p2, pose, color=None, thickness=0.017):
    """Return Open3D geometries for a two-finger grasp using box primitives."""
    p1 = np.asarray(p1)
    p2 = np.asarray(p2)
    R = pose[:3, :3]
    color = [1.0, 0.0, 0.0] if color is None else color

    body_dims = np.array([thickness, 0.005, 0.045])
    tip_dims = np.array([thickness, 0.01, 0.017])

    tip_half = tip_dims * 0.5
    body_half = body_dims * 0.5
    z_offset = tip_half[2] + body_half[2]
    y_offset = tip_half[1] - body_half[1]

    tip_center_right = p1 + R @ np.array([0.0, tip_half[1], 0.0])
    tip_center_left = p2 + R @ np.array([0.0, -tip_half[1], 0.0])
    body_center_right = tip_center_right + R @ np.array([0.0, y_offset, -z_offset])
    body_center_left = tip_center_left + R @ np.array([0.0, -y_offset, -z_offset])
    grasp_inner = abs(np.dot(p1 - p2, R[:, 1]))
    grasp_width = grasp_inner + 2.0 * tip_dims[1]
    top_dims = np.array([body_dims[0], grasp_width, body_dims[1]])
    top_center = 0.5 * (p1 + p2) + R @ np.array([0.0, 0.0, -z_offset - body_half[2] - 0.5 * top_dims[2]])

    def make_box(dims, center):
        box = o3d.geometry.TriangleMesh.create_box(width=dims[0], height=dims[1], depth=dims[2])
        box.translate(-0.5 * dims)
        box.rotate(R, center=np.zeros(3))
        box.translate(center)
        box.paint_uniform_color(color)
        return box

    return [
        make_box(tip_dims, tip_center_right),
        make_box(body_dims, body_center_right),
        make_box(tip_dims, tip_center_left),
        make_box(body_dims, body_center_left),
        make_box(top_dims, top_center),
    ]


class GraspAnnotatorBase:
    def __init__(self, cloud, grasp_proposals=None, table_height=0.054):
        self.cloud = cloud
        self.frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05)
        self.table_height = table_height
        self.hand_depth = 0.035

        self.vis = o3d.visualization.VisualizerWithKeyCallback()
        self.vis.create_window()
        self.vis.add_geometry(cloud)
        self.vis.add_geometry(self.frame)
        self.proposal_geometries = []
        if grasp_proposals:
            proposal_geometries = visualize_scene_pcd_util(
                cloud,
                grasp_proposals,
                visualize_ref_system=False,
                color_key="score",
            )
            for geometry in proposal_geometries:
                if geometry is cloud:
                    continue
                self.vis.add_geometry(geometry, reset_bounding_box=False)
                self.proposal_geometries.append(geometry)

        self.marker_geometries = []

    def point1(self, grasp_pose=None, grasp_width=None):
        if grasp_pose is None: grasp_pose = self.grasp_pose
        if grasp_width is None: grasp_width = self.grasp_width
        center = grasp_pose[:3, 3]
        y_dir = grasp_pose[:3, 1]
        return center + 0.5*grasp_width*y_dir

    def point2(self, grasp_pose=None, grasp_width=None):
        if grasp_pose is None: grasp_pose = self.grasp_pose
        if grasp_width is None: grasp_width = self.grasp_width
        center = grasp_pose[:3, 3]
        y_dir = grasp_pose[:3, 1]
        return center - 0.5*grasp_width*y_dir

    def remove_marker(self):
        for g in self.marker_geometries:
            self.vis.remove_geometry(g, reset_bounding_box=False)
        self.marker_geometries = []

    def add_marker(self, reset_view=False):
        p1 = self.point1()
        p2 = self.point2()
        pose = self.grasp_pose
        geometries = create_grasp_visualization(p1, p2, pose)
        for g in geometries:
            self.vis.add_geometry(g, reset_bounding_box=reset_view)
            self.marker_geometries.append(g)

    def close(self):
        self.vis.destroy_window()


class GraspAnnotatorBinaryFeedback(GraspAnnotatorBase):
    def __init__(self, cloud, grasp, grasp_proposals=None, table_height=0.054):
        super().__init__(cloud, grasp_proposals=grasp_proposals, table_height=table_height)
        self.grasp = grasp
        self.grasp_pose = np.asarray(grasp["pose"])
        self.grasp_width = grasp["width"]
        self.label = None

        self.vis.register_key_callback(ord("Y"), self.mark_success)
        self.vis.register_key_callback(ord("y"), self.mark_success)
        self.vis.register_key_callback(ord("N"), self.mark_failure)
        self.vis.register_key_callback(ord("n"), self.mark_failure)
        self.add_marker(reset_view=True)

    def mark_success(self, vis):
        self.label = 1.0
        vis.close()
        return False

    def mark_failure(self, vis):
        self.label = 0.0
        vis.close()
        return False

    def run(self):
        print("Label grasp: press 'y' for success or 'n' for failure.")
        self.vis.run()
        self.close()
        if self.label is None:
            raise RuntimeError("Binary grasp feedback was closed without a y/n label.")
        return self.label


class GraspAnnotatorDemonstration(GraspAnnotatorBase):
    def __init__(
        self,
        cloud,
        grasp_proposals=None,
        log_path=None,
        grasp_type="annotated",
        table_height=0.054,
        max_demonstrations=None,
    ):
        super().__init__(cloud, grasp_proposals=grasp_proposals, table_height=table_height)
        self.folder = log_path
        if log_path is not None:
            self.grasps_path = os.path.join(self.folder, "annotated_grasps.csv")
            self._scene_grasp_data_handler = SceneGraspDataHandler(self.grasps_path)
        self.grasp_type = grasp_type
        self.max_demonstrations = max_demonstrations

        # Grasp pose as 4x4 transformation
        self.grasp_pose = np.array([[-1., 0., 0., 0.15],
                                    [0., 1., 0., 0.15],
                                    [0., 0., -1., 0.15],
                                    [0., 0., 0., 1.0]])
        self.grasp_width = 0.08  # meters

        self.grasps = []
        self.declined = False
        self.direction_sign = 1  # 1 or -1 depending on Shift state

        self.register_callbacks()
        self.add_marker(reset_view=True)

    def register_callbacks(self):
        # Movement
        self.vis.register_key_callback(ord("W"), lambda vis: self.translate_grasp([0, 0, 0.002]))
        self.vis.register_key_callback(ord("A"), lambda vis: self.translate_grasp([0.002, 0, 0]))
        self.vis.register_key_callback(ord("D"), lambda vis: self.translate_grasp([0, 0.002, 0]))

        # Rotation (around X, Y, Z)
        self.vis.register_key_callback(ord("J"), lambda vis: self.rotate_grasp("x", 2))
        self.vis.register_key_callback(ord("I"), lambda vis: self.rotate_grasp("y", 2))
        self.vis.register_key_callback(ord("L"), lambda vis: self.rotate_grasp("z", 2))

        # Width adjustment
        self.vis.register_key_callback(ord("0"), lambda vis: self.change_width(0.002))
        self.vis.register_key_callback(ord("9"), lambda vis: self.change_width(-0.002))

        # Save grasp
        self.vis.register_key_callback(257, self.save_grasp)

        # Decline providing a demonstration and close the annotator.
        self.vis.register_key_callback(ord("N"), self.decline_demonstration)
        self.vis.register_key_callback(ord("n"), self.decline_demonstration)

        # Modifier key (Shift) to invert direction
        self.vis.register_key_callback(340, lambda vis: self.toggle_direction())  # Shift (Left)

        # Exit and export
        # self.vis.register_key_callback(256, self.export_and_exit)  # ESC

    def toggle_direction(self):
        self.direction_sign *= -1

    def check_valid(self, grasp_pose_proposal=None, grasp_width_proposal=None):
        if grasp_pose_proposal is None: grasp_pose_proposal = self.grasp_pose
        if grasp_width_proposal is None: grasp_width_proposal = self.grasp_width
        p1 = self.point1(grasp_pose_proposal, grasp_width_proposal)
        p2 = self.point2(grasp_pose_proposal, grasp_width_proposal)
        if p1[2] <= self.table_height: return False
        if p2[2] <= self.table_height: return False
        return True

    def translate_grasp(self, delta_local):
        delta = np.array(delta_local) * self.direction_sign
        translation = self.grasp_pose[:3, :3] @ delta
        grasp_pose_proposal = np.copy(self.grasp_pose)
        grasp_pose_proposal[:3, 3] += translation
        if self.check_valid(grasp_pose_proposal=grasp_pose_proposal):
            self.grasp_pose = grasp_pose_proposal
        self.remove_marker()
        self.add_marker()
        # self.transform_markers(t=translation)

    def rotate_grasp(self, axis, angle_deg):
        angle = angle_deg * self.direction_sign
        R_axis = R.from_euler(axis, angle, degrees=True).as_matrix()
        rot = np.eye(4)
        rot[:3, :3] = R_axis
        grasp_pose_proposal = self.grasp_pose @ rot
        if self.check_valid(grasp_pose_proposal=grasp_pose_proposal):
            self.grasp_pose = grasp_pose_proposal
        self.remove_marker()
        self.add_marker()

    def change_width(self, delta):
        grasp_width_proposal = min(0.08, max(0.01, self.grasp_width + delta))
        if self.check_valid(grasp_width_proposal=grasp_width_proposal):
            self.grasp_width = grasp_width_proposal
        self.remove_marker()
        self.add_marker()

    def save_grasp(self, vis):
        grasp = {
            "pose": self.grasp_pose,
            "point1": self.point1(),
            "point2": self.point2(),
            "width": self.grasp_width,
            "type": self.grasp_type,
            "score": 1.0
        }
        if self.folder is not None:
            self._scene_grasp_data_handler.append(grasp, res={"success": 1.0})
        self.grasps.append(grasp)
        if self.max_demonstrations is not None and len(self.grasps) >= self.max_demonstrations:
            vis.close()

    def decline_demonstration(self, vis):
        self.declined = True
        vis.close()
        return False

    def run(self):
        print("Demonstration annotator: press Enter to save a grasp or 'n' to skip.")
        self.vis.run()
        self.close()
        return self.grasps


GraspAnnotator = GraspAnnotatorDemonstration
