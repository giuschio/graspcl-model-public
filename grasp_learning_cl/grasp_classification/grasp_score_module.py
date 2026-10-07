import time

from copy import deepcopy
from typing import Optional

import numpy as np
import open3d as o3d
import torch

from grasp_learning_cl.grasp_classification.encoders import BpsSnnAutoencoderLightning
from grasp_learning_cl.grasp_classification.collision_checker import CollisionChecker
from grasp_learning_cl.grasp_classification.table_filter import TableFilter
from grasp_learning_cl.transform import Transform

from grasp_learning_cl.grasp_heuristics import get_sampler


class GraspScoringBaselineModule:
    """Manage grasp demonstrations, continual learning data, and prediction."""

    CONFIG_FILENAME = "config.json"

    def __init__(
        self,
        encoder_path: str,
        device: str = "cpu",
    ) -> None:
        self.pretrained_grasp_proposal_model = get_sampler("ensemblehq")
        self.encoder_path = str(encoder_path)
        self.encoder_device = torch.device(device)

        encoder = BpsSnnAutoencoderLightning.load_from_checkpoint(self.encoder_path)
        # print("Loaded snnautoencoder")

        self.encoder = encoder
        self.encoder.feature_key = "feature_encoder"
        self.encoder.to(self.encoder_device)
        self.encoder.eval()
        self.encoder.freeze()
 
        # the panda hand is (4 x 20 x 8) cm. The fingers are 4cm offset
        self.collision_checker = CollisionChecker()
        self.table_filter = TableFilter()

    # ------------------------------------------------------------------
    # Prediction pipeline
    # ------------------------------------------------------------------
    def predict(
        self,
        pointcloud_w_table: o3d.geometry.PointCloud,
        object_type: Optional[str] = None,
        grasp_type: Optional[str] = None,
        grasp_score_threshold: float = 0.85,
        grasp_style_threshold: float = 0.85,
        estimation_mode: str = "high_confidence",
        score_key: str = "score",
        debug: bool = False
    ) -> list:
        """
        Generate grasp proposals for a scene, score them, and return filtered results.

        Args:
            pointcloud: Scene point cloud from which to generate grasps. Includes table.
            object_type: Optional object label used by recall/style modules.
            grasp_type: Optional grasp style to condition predictions and filtering.
            grasp_score_threshold: Minimum grasp score to keep after scoring.
            grasp_style_threshold: Minimum style score to keep when ``grasp_type`` is set.
            estimation_mode: Mode forwarded to the grasp scoring module.
            score_key: Dictionary key containing the grasp quality score to sort/filter by.

        Returns:
            List of grasp proposal dictionaries enriched with features and scores, sorted
            by ``score_key`` in descending order and filtered by the provided thresholds.

        Notes:
            The grasp embedding is trained on pointclouds where the table has been cut out
            See scripts_experiments/experiment_11_collect_training_data_for_distillation.py
            The filtering of the table is exact (i.e. the table is 50mm tall and we cut 50mm)

            This module assumes we provide unfiltered pointclouds (i.e. with the table). Most components
            will filter the pcd, but the table is needed for collision checking.
        """
        total_t0 = time.perf_counter()
        object_recall_ms = 0.0
        patch_recall_ms = 0.0
        pretrained_sampling_ms = 0.0
        collision_checking_ms = 0.0
        encoding_ms = 0.0
        scoring_ms = 0.0

        grasp_proposals: list = []

        pointcloud_w_table = pointcloud_w_table
        pointcloud_no_table = self.table_filter(pointcloud_w_table)

        t0 = time.perf_counter()
        if self.pretrained_grasp_proposal_model is not None:
            external_proposals = self.pretrained_grasp_proposal_model(
                cloud_objects=pointcloud_no_table,
                cloud_collisions=pointcloud_w_table,
                n_grasps=512,
                debug=debug)
            grasp_proposals.extend(external_proposals)
        pretrained_sampling_ms = (time.perf_counter() - t0) * 1000

        """
        At this point, we expect each grasp to have the following keys:
        - pose
        - point1
        - point2
        - width
        - score_style (if they come from the grasp_learning_cl.recall_module)
        """
        t0 = time.perf_counter()
        grasp_proposals = self.collision_checker.filter_grasps(
            pointcloud_w_table,
            grasp_proposals,
            debug=debug,
        )
        collision_checking_ms = (time.perf_counter() - t0) * 1000

        if not grasp_proposals:
            return []

        t0 = time.perf_counter()
        # already does the scoring
        grasp_proposals = self.encoder.predict_proba(pointcloud_no_table, grasp_proposals)
        encoding_ms = (time.perf_counter() - t0) * 1000

        filtered = grasp_proposals

        # print("Score of the recalled grasp: ", grasp_proposals[0]["score"])
        filtered = [g for g in filtered if g.get(score_key, -np.inf) >= grasp_score_threshold]
        # duplicate grasps by flipping them symmetrically
        t = Transform.from_6dofs([0., 0., 0., 0., 0., 180.], degrees=True)
        filtered_flipped = list()
        for g in filtered:
            gf = deepcopy(g)
            gf["pose"] = gf["pose"] @ t.as_matrix()
            gf["point1"], gf["point2"] = gf["point2"], gf["point1"]
            filtered_flipped.append(gf)
        filtered = filtered + filtered_flipped

        filtered.sort(key=lambda g: g.get(score_key, -np.inf), reverse=True)
        return filtered
