import json
import filecmp
import shutil
import time

from copy import deepcopy
from pathlib import Path
from typing import Optional

import numpy as np
import open3d as o3d
import torch

from grasp_learning_cl.grasp_classification.encoders import BpsSnnAutoencoderLightning

from grasp_learning_cl.grasp_classification.grasp_score_module_probabilistic import GraspScoreProbabilisticModule
from grasp_learning_cl.grasp_classification.collision_checker import CollisionChecker
from grasp_learning_cl.grasp_classification.table_filter import TableFilter
from grasp_learning_cl.grasp_recall.grasp_recall_module_registration_object import GraspRecallModuleObject
from grasp_learning_cl.grasp_recall.grasp_recall_module_registration_patch import GraspRecallModulePatch
from grasp_learning_cl.transform import Transform
from grasp_learning_cl.grasp_datagen.annotation import GraspAnnotatorBinaryFeedback, GraspAnnotatorDemonstration

from grasp_learning_cl.grasp_recall_filter_module import GraspRecallFilterModule
from grasp_learning_cl.grasp_heuristics import get_sampler


def flip_if_necessary(grasp_proposals):
    # todo: given a list of grasp_proposals (which include the pose key a 4x4 matrix) if the z axis points up, flip it (i.e. rotate by 180 degs around y)
    pass


class ContinualLearningModule:
    """Manage grasp demonstrations, continual learning data, and prediction."""

    CONFIG_FILENAME = "config.json"
    FEEDBACK_COUNTS_FILENAME = "feedback_counts.json"
    ENCODER_LOCATIONS = {"internal", "external"}
    INTERNAL_ENCODER_DIRECTORY = "encoder"

    @classmethod
    def _load_json_if_exists(cls, path: Path, default):
        if not path.exists():
            return default
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)

    @classmethod
    def load_training_metadata(cls, modules_root: str | Path) -> dict:
        modules_root_path = Path(modules_root)
        config = cls._load_json_if_exists(modules_root_path / cls.CONFIG_FILENAME, {})
        feedback_counts = cls._load_json_if_exists(modules_root_path / cls.FEEDBACK_COUNTS_FILENAME, {})

        base_train_object_set = config.get("base_train_object_set")
        adaptation_object_sets = config.get("adaptation_object_sets", [])
        if isinstance(adaptation_object_sets, str):
            adaptation_object_sets = [adaptation_object_sets]

        for key in ("train_object_set", "object_set"):
            if config.get(key):
                train_object_set = config[key]
                break
        else:
            trained_categories = [
                key
                for key, counts in feedback_counts.items()
                if key != "unknown"
                and (
                    int(counts.get("labels", 0)) > 0
                    or int(counts.get("demonstrations", 0)) > 0
                )
            ]
            if base_train_object_set is not None:
                train_object_set_parts = [base_train_object_set] + sorted(set(adaptation_object_sets + trained_categories))
                train_object_set = "+".join(train_object_set_parts)
            elif len(trained_categories) == 1:
                train_object_set = trained_categories[0]
            else:
                train_object_set = None

        return {
            "train_object_set": train_object_set,
            "base_train_object_set": base_train_object_set,
            "adaptation_object_sets": sorted(set(adaptation_object_sets)),
            "feedback_counts": feedback_counts,
        }

    @classmethod
    def infer_train_object_set(cls, modules_root: str | Path) -> str | None:
        return cls.load_training_metadata(modules_root)["train_object_set"]

    @classmethod
    def update_training_metadata(
        cls,
        modules_root: str | Path,
        train_object_set: str | None = None,
        base_train_object_set: str | None = None,
    ) -> None:
        modules_root_path = Path(modules_root)
        config_path = modules_root_path / cls.CONFIG_FILENAME
        if not config_path.exists():
            raise FileNotFoundError(f"Missing CL module config: {config_path}")

        with config_path.open("r", encoding="utf-8") as config_file:
            config = json.load(config_file)

        if base_train_object_set is not None:
            config["base_train_object_set"] = base_train_object_set
            config["train_object_set"] = base_train_object_set

        if train_object_set is not None:
            existing = config.get("adaptation_object_sets", [])
            if isinstance(existing, str):
                existing = [existing]
            config["adaptation_object_sets"] = sorted(set(existing + [train_object_set]))

            base = config.get("base_train_object_set", config.get("train_object_set"))
            parts = []
            if base is not None:
                parts.append(base)
            parts.extend(config["adaptation_object_sets"])
            config["train_object_set"] = "+".join(parts)

        feedback_counts_path = modules_root_path / cls.FEEDBACK_COUNTS_FILENAME
        if feedback_counts_path.exists():
            with feedback_counts_path.open("r", encoding="utf-8") as feedback_counts_file:
                config["feedback_counts"] = json.load(feedback_counts_file)

        with config_path.open("w", encoding="utf-8") as config_file:
            json.dump(config, config_file)

    @staticmethod
    def format_training_feedback_counts(training_metadata: dict) -> str:
        return json.dumps(training_metadata.get("feedback_counts", {}), sort_keys=True)

    def __init__(
        self,
        encoder_path: str,
        device: str = "cpu",
        encoder_pointcloud: str = "no_table",
        encoder_location: str = "external",
        encoder_root: str | Path | None = None,
    ) -> None:
        self.pretrained_grasp_proposal_model = get_sampler("ensemblehq")
        self.encoder_path = str(encoder_path)
        self.encoder_location = self._normalize_encoder_location(encoder_location)
        self.encoder_pointcloud = self._normalize_encoder_pointcloud(encoder_pointcloud)
        self.encoder_device = torch.device(device)
        self.modules_root: Path | None = None
        resolved_encoder_path = self.resolve_encoder_path(encoder_root)
        self.encoder = BpsSnnAutoencoderLightning.load_from_checkpoint(str(resolved_encoder_path))
        self.encoder.feature_key = "feature_encoder"
        self.encoder.to(self.encoder_device)
        self.encoder.eval()
        self.encoder.freeze()
        self.grasp_recall_module_object: GraspRecallModuleObject | None = None
        self.grasp_recall_module_patch: GraspRecallModulePatch | None = None
        self.grasp_recall_filter_module: GraspRecallFilterModule | None = None
        self.grasp_scoring_module_cl: GraspScoreProbabilisticModule | None = None
        # the panda hand is (4 x 20 x 8) cm. The fingers are 4cm offset
        self.collision_checker = CollisionChecker()
        self.table_filter = TableFilter()
        self.feedback_counts = {}

    @classmethod
    def _normalize_encoder_location(cls, encoder_location: str) -> str:
        value = str(encoder_location).strip().lower()
        if value not in cls.ENCODER_LOCATIONS:
            raise ValueError(
                f"Unsupported encoder_location={encoder_location!r}. "
                "Use 'internal' or 'external'."
            )
        return value

    def resolve_encoder_path(self, modules_root: str | Path | None = None) -> Path:
        """Resolve the configured encoder path without changing its stored form."""
        encoder_path = Path(self.encoder_path).expanduser()
        if self.encoder_location == "external":
            return encoder_path

        if encoder_path.is_absolute():
            raise ValueError("An internal encoder_path must be relative to the CL-module root")
        root = Path(modules_root) if modules_root is not None else self.modules_root
        if root is None:
            raise ValueError("The CL-module root is required to resolve an internal encoder")

        root = root.expanduser().resolve()
        resolved_path = (root / encoder_path).resolve()
        try:
            resolved_path.relative_to(root)
        except ValueError as error:
            raise ValueError("An internal encoder_path cannot leave the CL-module root") from error
        return resolved_path

    @staticmethod
    def _normalize_encoder_pointcloud(encoder_pointcloud: str) -> str:
        aliases = {
            "no_table": "no_table",
            "without_table": "no_table",
            "pointcloud_no_table": "no_table",
            "with_table": "with_table",
            "w_table": "with_table",
            "pointcloud_w_table": "with_table",
        }
        value = str(encoder_pointcloud).strip().lower()
        if value not in aliases:
            raise ValueError(
                f"Unsupported encoder_pointcloud={encoder_pointcloud!r}. "
                "Use 'no_table' or 'with_table'."
            )
        return aliases[value]

    def _get_encoder_pointcloud(
        self,
        pointcloud_w_table: o3d.geometry.PointCloud,
        pointcloud_no_table: o3d.geometry.PointCloud | None = None,
    ) -> o3d.geometry.PointCloud:
        if self.encoder_pointcloud == "with_table":
            return pointcloud_w_table
        if pointcloud_no_table is None:
            pointcloud_no_table = self.table_filter(pointcloud_w_table)
        return pointcloud_no_table

    def set_module_root(self, modules_root: str | Path | None) -> None:
        """Configure module roots and lazily instantiate the sub-modules."""
        if modules_root is None:
            self.modules_root = None
            self.grasp_recall_module_object = None
            self.grasp_style_module_cl = None
            self.grasp_recall_module_patch = None
            self.grasp_recall_filter_module = None
            return

        root_path = Path(modules_root)
        root_path.mkdir(parents=True, exist_ok=True)
        self.modules_root = root_path

        if self.grasp_recall_module_object is None:
            self.grasp_recall_module_object = GraspRecallModuleObject(data_folder=root_path / "recall")

        if self.grasp_recall_module_patch is None:
            self.grasp_recall_module_patch = GraspRecallModulePatch(data_folder=root_path / "recall_patches")

        if self.grasp_recall_filter_module is None:
            self.grasp_recall_filter_module = GraspRecallFilterModule(feature_key=self.encoder.feature_key)

        if self.grasp_scoring_module_cl is None:
            score_dir = root_path / "score_cl"
            self.grasp_scoring_module_cl = GraspScoreProbabilisticModule.load(
                score_dir, feature_key=self.encoder.feature_key
            )

    def save(self, modules_root: str | Path) -> None:
        modules_root_path = Path(modules_root)
        modules_root_path.mkdir(parents=True, exist_ok=True)

        if self.modules_root is None:
            self.set_module_root(modules_root_path)
        elif self.modules_root.resolve() != modules_root_path.resolve():
            raise ValueError("modules_root mismatch. Call set_module_root before saving to a different location.")

        score_dir = modules_root_path / "score_cl"
        if self.grasp_scoring_module_cl is not None:
            score_dir.mkdir(parents=True, exist_ok=True)
            self.grasp_scoring_module_cl.save(score_dir)

        self._write_module_config(modules_root_path)

        feedback_counts_path = modules_root_path / self.FEEDBACK_COUNTS_FILENAME
        with feedback_counts_path.open("w", encoding="utf-8") as feedback_counts_file:
            json.dump(self.feedback_counts, feedback_counts_file, indent=2)

    def _write_module_config(self, modules_root: str | Path) -> None:
        modules_root_path = Path(modules_root)
        config_path = modules_root_path / self.CONFIG_FILENAME
        config = {}
        if config_path.exists():
            with config_path.open("r", encoding="utf-8") as config_file:
                config = json.load(config_file)
        config.update({
            "encoder_path": self.encoder_path,
            "encoder_location": self.encoder_location,
            "encoder_pointcloud": self.encoder_pointcloud,
        })
        with config_path.open("w", encoding="utf-8") as config_file:
            json.dump(config, config_file, indent=2)
            config_file.write("\n")

    def internalize_encoder(self) -> Path:
        """Copy the encoder into this CL-module folder and store a relative path."""
        if self.modules_root is None:
            raise ValueError("Call set_module_root before internalizing the encoder")

        source_path = self.resolve_encoder_path(self.modules_root)
        if not source_path.is_file():
            raise FileNotFoundError(f"Encoder checkpoint not found: {source_path}")

        relative_target = Path(self.INTERNAL_ENCODER_DIRECTORY) / source_path.name
        target_path = (self.modules_root / relative_target).resolve()
        target_path.parent.mkdir(parents=True, exist_ok=True)

        if source_path.resolve() != target_path:
            if target_path.exists() and not filecmp.cmp(source_path, target_path, shallow=False):
                raise FileExistsError(
                    f"A different internal encoder already exists at {target_path}"
                )
            if not target_path.exists():
                shutil.copy2(source_path, target_path)

        self.encoder_path = relative_target.as_posix()
        self.encoder_location = "internal"
        self._write_module_config(self.modules_root)
        return target_path

    def drop_online_data(self) -> bool:
        """Delete online scoring and recall data after an explicit confirmation."""
        if self.modules_root is None:
            raise ValueError("Call set_module_root before dropping online data")

        confirmation = input(
            f"Delete all online scoring and recall data from {self.modules_root}? "
            "Type 'yes' to confirm: "
        )
        if confirmation.strip().lower() != "yes":
            print("Online data was not changed.")
            return False

        if self.grasp_scoring_module_cl is not None:
            self.grasp_scoring_module_cl.clear_online_data()
        else:
            online_path = self.modules_root / "score_cl" / "online_data.npz"
            if online_path.exists():
                online_path.unlink()

        for recall_module in (
            self.grasp_recall_module_object,
            self.grasp_recall_module_patch,
        ):
            if recall_module is not None:
                recall_module.clear()

        self.feedback_counts = {}
        feedback_counts_path = self.modules_root / self.FEEDBACK_COUNTS_FILENAME
        with feedback_counts_path.open("w", encoding="utf-8") as feedback_counts_file:
            json.dump(self.feedback_counts, feedback_counts_file, indent=2)
            feedback_counts_file.write("\n")

        config_path = self.modules_root / self.CONFIG_FILENAME
        config = self._load_json_if_exists(config_path, {})
        config.pop("feedback_counts", None)
        config["adaptation_object_sets"] = []
        if config.get("base_train_object_set") is not None:
            config["train_object_set"] = config["base_train_object_set"]
        config.update({
            "encoder_path": self.encoder_path,
            "encoder_location": self.encoder_location,
            "encoder_pointcloud": self.encoder_pointcloud,
        })
        with config_path.open("w", encoding="utf-8") as config_file:
            json.dump(config, config_file, indent=2)
            config_file.write("\n")

        print("Dropped online scoring and recall data.")
        return True

    @classmethod
    def load_inplace(cls, modules_root: str | Path, device: str = "cpu") -> "ContinualLearningModule":
        modules_root_path = Path(modules_root)
        config_path = modules_root_path / cls.CONFIG_FILENAME
        if not config_path.exists():
            raise FileNotFoundError(f"Missing config file at {config_path}")

        with config_path.open("r", encoding="utf-8") as config_file:
            config = json.load(config_file)

        if "encoder_path" not in config:
            raise ValueError(f"encoder_path missing from config: {config_path}")

        instance = cls(
            config["encoder_path"],
            device=device,
            encoder_pointcloud=config.get("encoder_pointcloud", "no_table"),
            encoder_location=config.get("encoder_location", "external"),
            encoder_root=modules_root_path,
        )
        instance.set_module_root(modules_root_path)

        feedback_counts_path = modules_root_path / cls.FEEDBACK_COUNTS_FILENAME
        if feedback_counts_path.exists():
            with feedback_counts_path.open("r", encoding="utf-8") as feedback_counts_file:
                instance.feedback_counts = json.load(feedback_counts_file)

        return instance

    @classmethod
    def load_copy(
        cls,
        modules_root: str | Path,
        new_root: str | Path,
        device: str = "cpu",
    ) -> "ContinualLearningModule":
        src_root = Path(modules_root)
        dst_root = Path(new_root)
        shutil.copytree(src_root, dst_root, dirs_exist_ok=True)
        return cls.load_inplace(dst_root, device=device)


    # ------------------------------------------------------------------
    # Prediction pipeline
    # ------------------------------------------------------------------
    def predict(
        self,
        pointcloud_w_table: o3d.geometry.PointCloud,
        object_type: Optional[str] = None,
        grasp_type: Optional[str] = None,
        grasp_score_threshold: float = 0.0,
        grasp_style_threshold: float = 0.0,
        estimation_mode: str = "posterior",
        score_key: str = "score",
        use_recall: bool = True,
        use_sampler: bool = True,
        use_scoring: bool = True,
        debug: bool = False
    ) -> list:
        """
        Generate grasp proposals for a scene, optionally score them, and return filtered results.

        Args:
            pointcloud_w_table: Scene point cloud from which to generate grasps. Includes table.
            object_type: Optional object label used by recall/style modules.
            grasp_type: Optional grasp style to condition predictions and filtering.
            grasp_score_threshold: Minimum grasp score to keep after scoring.
            grasp_style_threshold: Minimum style score to keep when ``grasp_type`` is set.
            estimation_mode: Mode forwarded to the grasp scoring module.
            score_key: Dictionary key containing the grasp quality score to sort/filter by.
            use_recall: Whether to use the object and patch recall modules.
            use_sampler: Whether to use the pretrained grasp sampler.
            use_scoring: Whether to run the grasp scoring module. If ``False``, all
                grasps receive a default score of ``1.0``.

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

        recalled_grasp_proposals: list = []
        pretrained_grasp_proposals: list = []

        pointcloud_w_table = pointcloud_w_table
        pointcloud_no_table = self.table_filter(pointcloud_w_table)

        t0 = time.perf_counter()
        if use_recall and self.grasp_recall_module_object is not None:
            bf = len(recalled_grasp_proposals)
            object_recall_proposals = self.grasp_recall_module_object.predict(
                pointcloud_no_table,
                pointcloud_w_table,
                object_type,
                grasp_type,
                max_matches_per_template=10,
                max_total_time_ms=1000,
                debug=False)
            for grasp in object_recall_proposals:
                grasp["proposal_source"] = "recall_object"
            recalled_grasp_proposals.extend(object_recall_proposals)
            af = len(recalled_grasp_proposals)
            # print(f"recalled {af-bf} from object")
        object_recall_ms = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        if use_recall and self.grasp_recall_module_patch is not None:
            bf = len(recalled_grasp_proposals)
            patch_recall_proposals = self.grasp_recall_module_patch.predict(
                pointcloud_no_table,
                pointcloud_w_table,
                object_type,
                grasp_type,
                max_matches_per_template=30,
                max_total_time_ms=1000,
                debug=False)
            for grasp in patch_recall_proposals:
                grasp["proposal_source"] = "recall_patch"
            recalled_grasp_proposals.extend(patch_recall_proposals)
            af = len(recalled_grasp_proposals)
            # print(f"recalled {af-bf} from patch")
        patch_recall_ms = (time.perf_counter() - t0) * 1000
        recalled_grasp_count = len(recalled_grasp_proposals)

        t0 = time.perf_counter()
        if use_sampler and self.pretrained_grasp_proposal_model is not None:
            external_proposals = self.pretrained_grasp_proposal_model(
                cloud_objects=pointcloud_no_table,
                cloud_collisions=pointcloud_w_table,
                n_grasps=512,
                debug=debug)
            for grasp in external_proposals:
                grasp["proposal_source"] = "sampler"
            pretrained_grasp_proposals.extend(external_proposals)
            af = len(pretrained_grasp_proposals)
            # print(f"synthesized {len(external_proposals)} from pretrained")
        pretrained_sampling_ms = (time.perf_counter() - t0) * 1000

        """
        At this point, we expect each grasp to have the following keys:
        - pose
        - point1
        - point2
        - width
        - score_style (if they come from the grasp_recall_module)
        """
        t0 = time.perf_counter()
        recalled_grasp_proposals = self.collision_checker.filter_grasps(pointcloud_w_table, recalled_grasp_proposals)
        pretrained_grasp_proposals = self.collision_checker.filter_grasps(pointcloud_w_table, pretrained_grasp_proposals)
        collision_checking_ms = (time.perf_counter() - t0) * 1000
        # print(len(recalled_grasp_proposals))

        if not recalled_grasp_proposals and not pretrained_grasp_proposals:
            return []

        t0 = time.perf_counter()
        pointcloud_encoder = self._get_encoder_pointcloud(pointcloud_w_table, pointcloud_no_table)
        if recalled_grasp_proposals:
            recalled_grasp_proposals = self.encoder.predict(pointcloud_encoder, recalled_grasp_proposals)
        if pretrained_grasp_proposals:
            pretrained_grasp_proposals = self.encoder.predict(pointcloud_encoder, pretrained_grasp_proposals)
        encoding_ms = (time.perf_counter() - t0) * 1000

        # filtering of recalled proposals
        if use_recall and self.grasp_recall_filter_module is not None:
            recalled_grasp_proposals = self.grasp_recall_filter_module.filter_grasps(recalled_grasp_proposals)

        # print(len(recalled_grasp_proposals))
        grasp_proposals = recalled_grasp_proposals + pretrained_grasp_proposals

        t0 = time.perf_counter()
        """
        Add the following keys:
        - score_style (if applicable)
        """

        if use_scoring and self.grasp_scoring_module_cl is not None:
            self.grasp_scoring_module_cl.score_grasps(
                grasp_proposals,
                estimation_mode=estimation_mode,
            )
        elif not use_scoring:
            for grasp in grasp_proposals:
                grasp["score"] = 1.0
                grasp[score_key] = 1.0
        scoring_ms = (time.perf_counter() - t0) * 1000

        if grasp_type is not None:
            filtered = [g for g in grasp_proposals if g.get("score_style", -np.inf) >= grasp_style_threshold]
        else:
            filtered = grasp_proposals

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
        score_tie_eps = 1e-3
        start = 0
        while start < len(filtered):
            base_score = filtered[start].get(score_key, -np.inf)
            end = start + 1
            while end < len(filtered) and base_score - filtered[end].get(score_key, -np.inf) <= score_tie_eps:
                end += 1
            if end - start > 1:
                filtered[start:end] = sorted(filtered[start:end], key=lambda g: g["pose"][2, 2])
            start = end
        if debug:
            total_ms = (time.perf_counter() - total_t0) * 1000
            avg_recall_ms = (object_recall_ms + patch_recall_ms) / recalled_grasp_count if recalled_grasp_count > 0 else 0.0
            print(
                "Predict timing [ms]: object recall={:.2f}, patch recall={:.2f}, avg recall/proposal={:.2f} ({} recalled), pretrained sampling={:.2f}, collision checking={:.2f}, encoding={:.2f}, scoring={:.2f}, total={:.2f}".format(
                    object_recall_ms,
                    patch_recall_ms,
                    avg_recall_ms,
                    recalled_grasp_count,
                    pretrained_sampling_ms,
                    collision_checking_ms,
                    encoding_ms,
                    scoring_ms,
                    total_ms,
                )
            )
        return filtered

    # ------------------------------------------------------------------
    # Data ingestion
    # ------------------------------------------------------------------
    def get_demonstrations(
        self,
        pointcloud_w_table: o3d.geometry.PointCloud,
        grasp_proposals: list,
        max_demonstrations: int = 1,
    ) -> list:
        pointcloud_no_table = self.table_filter(pointcloud_w_table)
        no_table_points = np.asarray(pointcloud_no_table.points)
        if len(no_table_points) > 0:
            table_height = float(np.min(no_table_points[:, 2]))
        else:
            points = np.asarray(pointcloud_w_table.points)
            table_height = float(np.min(points[:, 2])) if len(points) > 0 else 0.0

        display_grasp_proposals = []
        if grasp_proposals:
            scores = np.asarray([grasp.get("score", -np.inf) for grasp in grasp_proposals], dtype=float)
            best_index = int(np.argmax(scores))
            other_indices = np.asarray(
                [index for index in range(len(grasp_proposals)) if index != best_index],
                dtype=int,
            )
            if len(other_indices) > 20:
                other_indices = np.random.choice(other_indices, size=20, replace=False)
            display_grasp_proposals = [grasp_proposals[best_index]]
            display_grasp_proposals.extend([grasp_proposals[int(index)] for index in other_indices])

        annotator = GraspAnnotatorDemonstration(
            cloud=pointcloud_w_table,
            grasp_proposals=display_grasp_proposals,
            log_path=None,
            grasp_type=None,
            table_height=table_height,
            max_demonstrations=max_demonstrations,
        )
        grasps = annotator.run()
        return grasps

    def get_binary_feedback(
        self,
        pointcloud_w_table: o3d.geometry.PointCloud,
        grasp_proposals: list,
        num_feedbacks: int = 20,
        object_type: Optional[str] = None,
    ) -> list:
        """
        Collect binary human feedback for selected candidate grasps and add it
        to the online scoring module.

        The acquisition loop mirrors ``update_cl_module_heuristic_6``: score
        currently available candidates, query the best one, add its binary
        label, then temporarily mask nearby feature neighbors before the next
        query.
        """
        if self.grasp_scoring_module_cl is None:
            raise RuntimeError("grasp_scoring_module_cl is not initialized.")
        if num_feedbacks <= 0:
            return []
        if not grasp_proposals:
            return []

        pointcloud_no_table = self.table_filter(pointcloud_w_table)
        pointcloud_encoder = self._get_encoder_pointcloud(pointcloud_w_table, pointcloud_no_table)
        grasp_proposals = self.encoder.predict(pointcloud_encoder, deepcopy(grasp_proposals))
        self.grasp_scoring_module_cl.score_grasps(
            grasp_proposals,
            estimation_mode="posterior",
        )

        features = np.asarray(
            [np.asarray(grasp[self.encoder.feature_key], dtype=np.float32).reshape(-1) for grasp in grasp_proposals],
            dtype=np.float32,
        )
        num_grasps = len(features)
        num_masked_neighbors = max(1, num_grasps // num_feedbacks)

        unsampled_indices = np.arange(num_grasps, dtype=int)
        available_mask = np.ones(num_grasps, dtype=bool)
        annotated_grasps = []

        for _ in range(min(num_feedbacks, num_grasps)):
            available_indices = unsampled_indices[available_mask[unsampled_indices]]
            if len(available_indices) == 0:
                available_mask[unsampled_indices] = True
                available_indices = unsampled_indices
            if len(available_indices) == 0:
                break

            candidate_features = features[available_indices]
            scores = self.grasp_scoring_module_cl.predict(
                candidate_features,
                estimation_mode="posterior",
            )
            local_index = int(np.argmax(scores))
            best_index = int(available_indices[local_index])
            grasp = grasp_proposals[best_index]

            annotator = GraspAnnotatorBinaryFeedback(pointcloud_w_table, grasp)
            label = float(annotator.run())
            grasp["success"] = label
            annotated_grasps.append(grasp)
            object_type_key = "unknown" if object_type is None else str(object_type)
            if object_type_key not in self.feedback_counts:
                self.feedback_counts[object_type_key] = {"labels": 0, "demonstrations": 0}
            self.feedback_counts[object_type_key]["labels"] += 1

            self.grasp_scoring_module_cl.add_datapoint(
                features[best_index : best_index + 1],
                np.asarray([label], dtype=np.float32),
            )

            unsampled_indices = unsampled_indices[unsampled_indices != best_index]
            if len(unsampled_indices) == 0:
                break

            available_mask[best_index] = False
            sampled_feature = features[best_index : best_index + 1]
            distances = np.sqrt(np.sum((features - sampled_feature) ** 2, axis=1))

            num_to_mask = min(num_masked_neighbors + 1, num_grasps)
            nearest_order = np.argsort(distances)[:num_to_mask]
            masked_indices = nearest_order[nearest_order != best_index]
            available_mask[masked_indices] = False

        return annotated_grasps

    def add_demonstration(
        self,
        pointcloud_w_table: o3d.geometry.PointCloud,
        demo_grasps: list,
        object_type: str,
        grasp_type: str,
        grasp_proposals: list,
        global_annotation: bool = True,
        local_annotation: bool = True,
    ) -> None:
        grasp_type = "default" if grasp_type is None else grasp_type
        object_type_key = "unknown" if object_type is None else str(object_type)
        if object_type_key not in self.feedback_counts:
            self.feedback_counts[object_type_key] = {"labels": 0, "demonstrations": 0}
        self.feedback_counts[object_type_key]["demonstrations"] += len(demo_grasps)

        pointcloud_no_table = self.table_filter(pointcloud_w_table)
        pointcloud_encoder = self._get_encoder_pointcloud(pointcloud_w_table, pointcloud_no_table)
        demo_grasps_with_embeddings = deepcopy(demo_grasps)
        encoded_demo_grasps = []
        if demo_grasps:
            encoded_demo_grasps = self.encoder.predict(pointcloud_encoder, deepcopy(demo_grasps))
            demo_grasps_with_embeddings = deepcopy(encoded_demo_grasps)

        demo_weights = []
        if self.grasp_scoring_module_cl is not None and encoded_demo_grasps:
            encoded_demo_grasps = self.grasp_scoring_module_cl.score_grasps(
                encoded_demo_grasps,
                estimation_mode="posterior",
            )
            demo_weights = self.grasp_scoring_module_cl.add_demonstration(
                encoded_demo_grasps,
                grasp_proposals,
            )

        print("Demonstration weights: " + ", ".join(f"{weight:.3f}" for weight in demo_weights))

        if global_annotation:
            self.grasp_recall_module_object.add_demonstration(
                deepcopy(pointcloud_no_table),
                deepcopy(pointcloud_w_table),
                deepcopy(demo_grasps_with_embeddings),
                object_type,
                grasp_type,
            )
        
        if local_annotation:
            self.grasp_recall_module_patch.add_demonstration(
                deepcopy(pointcloud_no_table),
                deepcopy(pointcloud_w_table),
                deepcopy(demo_grasps_with_embeddings),
                object_type,
                grasp_type,
                debug=True
            )

    def add_datapoint(
        self,
        pointcloud_w_table: o3d.geometry.PointCloud,
        grasps: list,
        object_type: Optional[str] = None,
        record_label: bool = True,
    ) -> None:
        if record_label:
            object_type_key = "unknown" if object_type is None else str(object_type)
            if object_type_key not in self.feedback_counts:
                self.feedback_counts[object_type_key] = {"labels": 0, "demonstrations": 0}
            self.feedback_counts[object_type_key]["labels"] += len(grasps)

        if self.grasp_scoring_module_cl is not None:
            pointcloud_no_table = self.table_filter(pointcloud_w_table)
            pointcloud_encoder = self._get_encoder_pointcloud(pointcloud_w_table, pointcloud_no_table)
            # recalculate feature to be safe
            grasps = self.encoder.predict(pointcloud_encoder, grasps)
            self.grasp_scoring_module_cl.add_datapoint(grasps)
