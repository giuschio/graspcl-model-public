import json
from pathlib import Path

import numpy as np
from sklearn.neighbors import KDTree
from sklearn.metrics import precision_recall_fscore_support

from scipy.stats import beta as stats_beta


def neighbors_within_radius(tree, points, radius):
    points = np.asarray(points)
    indices, distances = tree.query_radius(points, r=radius, return_distance=True, sort_results=True)
    ind_list = [np.asarray(idx, dtype=int) for idx in indices]
    dist_list = [np.asarray(dist, dtype=float) for dist in distances]
    return ind_list, dist_list


def nearest_neighbors_within_radius(tree, points, radius, max_neighbors):
    points = np.asarray(points)
    k = int(max_neighbors)
    k = max(1, k)
    k = min(k, tree.data.shape[0])
    distances, indices = tree.query(points, k=k, return_distance=True, sort_results=True)
    ind_list = []
    dist_list = []
    for dist_row, ind_row in zip(distances, indices):
        mask = dist_row <= radius
        ind_list.append(np.asarray(ind_row[mask], dtype=int))
        dist_list.append(np.asarray(dist_row[mask], dtype=float))
    return ind_list, dist_list


class GraspScoreProbabilisticModule:
    def __init__(
        self,
        temperature: float = 0.2,
        truncation_distance: float = 1.0,
        offline_max_strength: float = 10.0,
        online_max_strength: float = 1000.0,
        online_scale_factor: float = 3.0,
        prior_strength: float = 1.0,
        prior_mean: float = 0.1,
        max_neighbor_candidates: int = 100,
        demo_odds_margin: float = 1.0,
        max_single_demo_weight: float = 50.0,
        feature_key: str = "feature_encoder",
        data_folder: str | Path | None = None,
    ):
        """
        Local Beta classifier with offline fitting + online updates.

        Parameters
        ----------
        temperature : float
            Lengthscale used for the Gaussian neighbor weighting and to derive the search radius.
        offline_max_strength: float
            Max strength of offline data (i.e. the offline data counts for at most offline_max_strength neighbors)
        online_max_strength: float
            Max strength of online data after applying online_scale_factor.
        online_scale_factor : float
            Scaling factor applied to online evidence.
        prior_strength : float
            alpha0 + beta0; typically in [1,5].
        prior_mean : float or None
            Prior mean for p(y=1)
        max_neighbor_candidates : int
            Maximum number of nearest neighbors to examine per query before applying the radius filter.
            Use 0 to consider all neighbors within the radius.
        demo_odds_margin : float
            Minimum odds margin that a demonstration should add over the current greedy grasp.
            Also used as the minimum weight for a single demonstration.
        max_single_demo_weight : float or None
            Maximum weight assigned to one demonstration. Defaults to online_max_strength.
        """
        self.temperature = float(temperature)
        self.online_scale_factor = float(online_scale_factor)
        self.offline_max_strength = float(offline_max_strength)
        self.online_max_strength = float(online_max_strength)
        self.prior_strength = float(prior_strength)
        self.prior_mean = float(prior_mean)
        self.max_neighbor_candidates = int(max_neighbor_candidates)
        self.demo_odds_margin = float(demo_odds_margin)
        self.max_single_demo_weight = (
            float(self.online_max_strength)
            if max_single_demo_weight is None
            else float(max_single_demo_weight)
        )
        self.feature_key = feature_key
        self.truncation_distance = 3.0*temperature if truncation_distance is None else truncation_distance
        
        # Will be filled by fit_offline
        self.X_off = None
        self.y_off = None
        self.tree_off = None

        # Online data
        self.X_on = None
        self.y_on = None
        self.w_on = None
        self.tree_on = None
        self.data_folder = data_folder
        if self.data_folder is not None:
            self.data_folder = Path(data_folder)
            self.data_folder.mkdir(parents=True, exist_ok=True)
            self._save_metadata()


    # ------------------------------------------------------------------
    # Offline fit
    # ------------------------------------------------------------------

    @property
    def radius(self):
        return self.truncation_distance

    @property
    def alpha0(self):
        return float(self.prior_mean) * self.prior_strength
    
    @property
    def beta0(self):
        return (1 - float(self.prior_mean)) * self.prior_strength
    
    def fit_offline(self, X_off, y_off, persist=True):
        """
        Fit the model on offline data:
        - store offline data,
        - define prior (alpha0, beta0),
        - build KD-tree on ALL offline data.

        No calibration of lengthscale happens here (you pick it yourself).
        """
        X_off = np.asarray(X_off)
        y_off = np.asarray(y_off).astype(float)

        self.X_off = X_off
        self.y_off = y_off

        # KDTree on full offline data
        self.tree_off = KDTree(self.X_off)

        # Init empty online store
        self.X_on = np.zeros((0, X_off.shape[1]))
        self.y_on = np.zeros((0,))
        self.w_on = np.zeros((0,))
        self.tree_on = None

        if persist and self.data_folder is not None:
            self._save_metadata()
            self._save_offline_data()
            self._save_online_data()

    # ------------------------------------------------------------------
    # Online updates
    # ------------------------------------------------------------------

    def add_datapoint(self, grasps_or_features, labels=None):
        """
        Add one or more online datapoints. Accepts either a list of grasp dicts
        (each containing the encoded features and a ``success`` label) or
        explicit feature/label arrays.
        """
        if self.X_off is None:
            raise RuntimeError("Call fit_offline before adding online datapoints.")

        features, targets, point_weights = self._prepare_training_data(
            grasps_or_features,
            labels,
        )
        if features.size == 0:
            return

        if features.ndim == 1:
            features = features.reshape(1, -1)
        targets = targets.reshape(-1)
        point_weights = point_weights.reshape(-1)

        if self.X_on is None or self.X_on.shape[0] == 0:
            self.X_on = features
            self.y_on = targets
            self.w_on = point_weights
        else:
            if self.w_on is None:
                self.w_on = np.full(self.y_on.shape, self.online_scale_factor, dtype=float)
            self.X_on = np.vstack([self.X_on, features])
            self.y_on = np.hstack([self.y_on, targets])
            self.w_on = np.hstack([self.w_on, point_weights])

        self.tree_on = KDTree(self.X_on) if self.X_on.shape[0] > 0 else None
        if self.data_folder is not None:
            self._save_online_data()

    def add_demonstration(self, demo_grasps, grasp_proposals):
        """
        Add demonstrated successful grasps with adaptive per-demo weights.

        The weight is chosen so each demonstration reaches at least
        ``demo_odds_margin`` odds above the current greedy grasp estimate, then
        clipped to the configured per-demonstration bounds.
        """
        if not demo_grasps:
            return []
        if self.X_off is None:
            raise RuntimeError("Call fit_offline before adding demonstrations.")

        demo_features = np.asarray(
            [
                np.asarray(grasp[self.feature_key], dtype=float).reshape(-1)
                for grasp in demo_grasps
            ],
            dtype=float,
        )

        if not grasp_proposals:
            demo_weights = np.full(len(demo_grasps), self.online_scale_factor, dtype=float)
        else:
            scores = np.asarray([grasp.get("score", -np.inf) for grasp in grasp_proposals], dtype=float)
            best_index = int(np.argmax(scores))
            best_grasp = grasp_proposals[best_index]
            if "a" not in best_grasp or "b" not in best_grasp:
                raise KeyError("Greedy grasp proposal must include cached evidence keys 'a' and 'b'.")
            greedy_alpha = np.asarray([best_grasp["a"]], dtype=float)
            greedy_beta = np.asarray([best_grasp["b"]], dtype=float)

            if not all("a" in grasp and "b" in grasp for grasp in demo_grasps):
                raise KeyError("Each demonstrated grasp must include cached evidence keys 'a' and 'b'.")
            demo_alpha = np.asarray([grasp["a"] for grasp in demo_grasps], dtype=float)
            demo_beta = np.asarray([grasp["b"] for grasp in demo_grasps], dtype=float)
            greedy_odds = float(greedy_alpha[0]) / max(float(greedy_beta[0]), 1e-6)
            target_odds = greedy_odds + float(self.demo_odds_margin)
            raw_weights = target_odds * np.maximum(demo_beta.astype(float), 1e-6) - demo_alpha.astype(float)
            demo_weights = np.clip(
                raw_weights,
                float(self.demo_odds_margin),
                max(
                    float(self.demo_odds_margin),
                    float(self.max_single_demo_weight),
                ),
            )

        grasps_with_success = []
        for grasp, weight in zip(demo_grasps, demo_weights):
            grasp_with_success = dict(grasp)
            grasp_with_success["success"] = 1.0
            grasp_with_success["weight"] = float(weight)
            grasps_with_success.append(grasp_with_success)

        self.add_datapoint(grasps_with_success)
        return [float(weight) for weight in demo_weights]

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict_proba(self, x, estimation_mode):
        """
        Return Beta summary statistics for p(y=1 | x).

        Parameters
        ----------
        x : array-like or sequence of array-like
            Query point(s).
        estimation_mode : {"offline", "online", "posterior", "high_confidence"}
            "offline" -> use the Beta prior plus offline evidence.
            "online" -> use the Beta prior plus online evidence.
            "posterior" -> use the Beta prior plus offline and online evidence.
            "high_confidence" -> choose between the offline and online estimates
            based on which has higher local evidence mass.
        """
        if self.tree_off is None or self.X_off is None:
            raise RuntimeError("Call fit_offline before requesting predictions.")

        x = np.asarray(x)
        if x.ndim == 1:
            x = x.reshape(1, -1)

        s_off, n_off, s_on, n_on = self._collect_neighbor_stats(x)
        over_mask = n_off > self.offline_max_strength
        if np.any(over_mask):
            scale = np.ones_like(n_off)
            scale[over_mask] = self.offline_max_strength / n_off[over_mask]
            s_off = s_off * scale
            n_off = np.minimum(n_off, self.offline_max_strength)

        over_mask = n_on > self.online_max_strength
        if np.any(over_mask):
            scale = np.ones_like(n_on)
            scale[over_mask] = self.online_max_strength / n_on[over_mask]
            s_on = s_on * scale
            n_on = np.minimum(n_on, self.online_max_strength)

        a_offline = self.alpha0 + s_off
        b_offline = self.beta0 + (n_off - s_off)

        a_online = self.alpha0 + s_on
        b_online = self.beta0 + (n_on - s_on)

        a_post = self.alpha0 + s_off + s_on
        b_post = self.beta0 + (n_off - s_off) + (n_on - s_on)

        if estimation_mode == "offline":
            a = a_offline
            b = b_offline
        elif estimation_mode == "online":
            a = a_online
            b = b_online
        elif estimation_mode == "posterior":
            a = a_post
            b = b_post
        elif estimation_mode == "high_confidence":
            use_prior = n_off > n_on
            a = np.where(use_prior, a_offline, a_online)
            b = np.where(use_prior, b_offline, b_online)
        else:
            raise ValueError(
                "estimation_mode must be one of "
                "{'offline', 'online', 'posterior', 'high_confidence'}"
            )

        mean_arr = a / (a + b)
        return mean_arr, a, b

    def predict(self, x, estimation_mode):
        """
        Return predicted probability p(y=1 | x).
        """
        mean, _, _ = self.predict_proba(x, estimation_mode=estimation_mode)
        return mean

    def score_grasps(self, grasps, estimation_mode):
        """
        Assign score and variance to a list of grasp dictionaries in-place.
        """
        if not grasps:
            return grasps

        features = []
        for grasp in grasps:
            if self.feature_key not in grasp:
                raise KeyError(
                    f"Missing feature key '{self.feature_key}' in grasp: {grasp.keys()}"
                )
            features.append(np.asarray(grasp[self.feature_key]))
        mean, a, b = self.predict_proba(
            features,
            estimation_mode=estimation_mode,
        )
        mean = np.atleast_1d(np.asarray(mean, dtype=float)).reshape(-1)
        a = np.atleast_1d(np.asarray(a, dtype=float)).reshape(-1)
        b = np.atleast_1d(np.asarray(b, dtype=float)).reshape(-1)
        p05 = np.atleast_1d(np.asarray(stats_beta.ppf(0.05, a, b), dtype=float)).reshape(-1)
        p50 = np.atleast_1d(np.asarray(stats_beta.ppf(0.50, a, b), dtype=float)).reshape(-1)
        p95 = np.atleast_1d(np.asarray(stats_beta.ppf(0.95, a, b), dtype=float)).reshape(-1)
        score_thompson = np.atleast_1d(np.asarray(stats_beta.rvs(a, b), dtype=float)).reshape(-1)

        for g, m, alpha, beta, p0, p1, p2, thompson in zip(grasps, mean, a, b, p05, p50, p95, score_thompson):
            g["score"] = float(m)
            g["a"] = float(alpha)
            g["b"] = float(beta)
            g["score_05"] = float(p0)
            g["score_50"] = float(p1)
            g["score_95"] = float(p2)
            g["score_thompson"] = float(thompson)
            g["score_error_bar"] = float(abs(p2 - p0))
            
        return grasps

    # ------------------------------------------------------------------
    # Helper: neighbors
    # ------------------------------------------------------------------

    def _query_neighbor_candidates(self, tree, queries):
        if tree is None or queries.size == 0:
            empty_idx = [np.zeros((0,), dtype=int) for _ in range(len(queries))]
            empty_dist = [np.zeros((0,), dtype=float) for _ in range(len(queries))]
            return empty_idx, empty_dist

        if self.max_neighbor_candidates < 1:
            return neighbors_within_radius(tree, queries, self.radius)
        else:
            return nearest_neighbors_within_radius(tree, queries, self.radius, self.max_neighbor_candidates)

    def _collect_neighbor_stats(self, queries):
        if queries.size == 0:
            zeros = np.zeros(len(queries))
            return zeros, zeros.copy(), zeros.copy(), zeros.copy()

        off_idx_lists, off_dist_lists = self._query_neighbor_candidates(
            self.tree_off,
            queries,
        )
        on_idx_lists, on_dist_lists = self._query_neighbor_candidates(
            self.tree_on,
            queries,
        )

        s_off = np.zeros(len(queries))
        n_off = np.zeros(len(queries))
        s_on = np.zeros(len(queries))
        n_on = np.zeros(len(queries))

        k = self.max_neighbor_candidates
        for i, (off_idx, off_dist, on_idx, on_dist) in enumerate(
            zip(off_idx_lists, off_dist_lists, on_idx_lists, on_dist_lists)
        ):
            total_candidates = off_idx.size + on_idx.size
            if total_candidates == 0:
                continue

            all_distances = np.concatenate([off_dist, on_dist])
            all_labels = np.concatenate([self.y_off[off_idx], self.y_on[on_idx]])
            online_point_weights = (
                self.w_on[on_idx]
                if self.w_on is not None
                else np.full(on_idx.shape, self.online_scale_factor, dtype=float)
            )
            all_point_weights = np.concatenate([
                np.ones(off_idx.size, dtype=float),
                online_point_weights,
            ])
            source_is_online = np.concatenate([
                np.zeros(off_idx.size, dtype=bool),
                np.ones(on_idx.size, dtype=bool),
            ])

            if k > 0 and total_candidates > k:
                keep = np.argsort(all_distances)[:k]
                all_distances = all_distances[keep]
                all_labels = all_labels[keep]
                all_point_weights = all_point_weights[keep]
                source_is_online = source_is_online[keep]

            weights = np.exp(-0.5 * (all_distances / self.temperature) ** 2) * all_point_weights
            offline_mask = ~source_is_online
            if np.any(offline_mask):
                s_off[i] = np.sum(weights[offline_mask] * all_labels[offline_mask])
                n_off[i] = np.sum(weights[offline_mask])
            if np.any(source_is_online):
                s_on[i] = np.sum(weights[source_is_online] * all_labels[source_is_online])
                n_on[i] = np.sum(weights[source_is_online])

        return s_off, n_off, s_on, n_on

    def _prepare_training_data(self, grasps_or_features, labels):
        """
        Convert either grasp dicts or raw feature arrays into numpy arrays.
        """
        if labels is None:
            grasps = grasps_or_features
            if grasps is None:
                return np.zeros((0,)), np.zeros((0,)), np.zeros((0,))
            if not isinstance(grasps, (list, tuple)):
                raise TypeError(
                    "When labels is None, provide an iterable of grasp dictionaries."
                )
            if len(grasps) == 0:
                return np.zeros((0,)), np.zeros((0,)), np.zeros((0,))

            features = []
            targets = []
            point_weights = []
            for grasp in grasps:
                if self.feature_key not in grasp:
                    raise KeyError(
                        f"Missing feature key '{self.feature_key}' in grasp: {grasp.keys()}"
                    )
                if "success" not in grasp:
                    raise KeyError("Each grasp must include a 'success' label.")
                features.append(np.asarray(grasp[self.feature_key], dtype=float))
                targets.append(float(grasp["success"]))
                point_weights.append(float(grasp.get("weight", self.online_scale_factor)))
            features = np.asarray(features)
            targets = np.asarray(targets, dtype=float)
            point_weights = np.asarray(point_weights, dtype=float)
            return features, targets, point_weights

        features = np.asarray(grasps_or_features)
        targets = np.asarray(labels, dtype=float)
        point_weights = np.full(targets.shape, self.online_scale_factor, dtype=float)
        return features, targets, point_weights

    # ------------------------------------------------------------------
    # Lengthscale evaluation
    # ------------------------------------------------------------------

    def evaluate_on_val(self, X_val, y_val, threshold):
        """
        Evaluate a single lengthscale l using the stored offline data
        (training) and a provided validation set.

        Parameters
        ----------
        X_val : array-like, shape (N_val, d)
            Validation feature matrix.
        y_val : array-like, shape (N_val,)
            Validation labels (0/1).
        threshold: float
            Threshold for classification

        Returns
        -------
        tuple
            (precision, recall, f1) for the provided lengthscale.
        """
        if self.X_off is None or self.y_off is None:
            raise RuntimeError("Call fit_offline before evaluating lengthscales.")
        if self.tree_off is None:
            raise RuntimeError("Offline KD-tree not built. Call fit_offline first.")
        if self.alpha0 is None or self.beta0 is None:
            raise RuntimeError("Prior parameters unavailable. Call fit_offline first.")

        X_val = np.asarray(X_val)
        y_val = np.asarray(y_val).astype(float)

        preds = []
        for x in X_val:
            p = self.predict(x, estimation_mode="offline")
            preds.append(1 if p >= threshold else 0)

        precision, recall, f1, _ = precision_recall_fscore_support(
            y_val, preds, average="binary", zero_division=0
        )

        return precision, recall, f1

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------

    def save(self, directory=None):
        """
        Save hyperparameters and offline/online data to the given directory.
        """
        if self.X_off is None or self.y_off is None:
            raise RuntimeError("No data to save. Call fit_offline first.")

        if directory is not None:
            self.data_folder = Path(directory)

        target_dir = self._resolve_directory(directory)
        self._write_metadata(target_dir)
        self._write_offline_data(target_dir)
        self._write_online_data(target_dir)

    @classmethod
    def load(cls, directory, feature_key: str | None = None):
        """
        Load a model that was previously saved with `save`.
        """
        directory = Path(directory)
        config_path = directory / "config.json"
        offline_path = directory / "offline_data.npz"

        if not config_path.exists():
            raise FileNotFoundError(f"Missing config file at {config_path}")
        if not offline_path.exists():
            raise FileNotFoundError(f"Missing offline data at {offline_path}")

        with open(config_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        feature_key = metadata.get("feature_key", feature_key or "feature_encoder")
        init_kwargs = dict(
            temperature=metadata["temperature"],
            truncation_distance=metadata.get("truncation_distance", None),
            online_scale_factor=metadata["online_scale_factor"],
            offline_max_strength=metadata["offline_max_strength"],
            online_max_strength=metadata.get("online_max_strength", 10.0),
            prior_strength=metadata["prior_strength"],
            prior_mean=metadata["prior_mean"],
            max_neighbor_candidates=metadata.get("max_neighbor_candidates", 0),
            demo_odds_margin=metadata["demo_odds_margin"],
            max_single_demo_weight=metadata["max_single_demo_weight"],
            feature_key=feature_key,
            data_folder=directory,
        )

        model = cls(**init_kwargs)

        offline_data = np.load(offline_path)
        X_off = offline_data["X"]
        y_off = offline_data["y"]
        model.fit_offline(X_off, y_off, persist=False)

        online_path = directory / "online_data.npz"
        if online_path.exists():
            online_data = np.load(online_path)
            model.X_on = online_data["X"]
            model.y_on = online_data["y"]
            if "w" in online_data:
                model.w_on = online_data["w"]
            else:
                model.w_on = np.full(model.y_on.shape, model.online_scale_factor, dtype=float)
            if model.X_on is not None and model.X_on.shape[0] > 0:
                model.tree_on = KDTree(model.X_on)
                print("loaded online data")
            else:
                model.tree_on = None
        else:
            model.X_on = np.zeros((0, model.X_off.shape[1]))
            model.y_on = np.zeros((0,))
            model.w_on = np.zeros((0,))
            model.tree_on = None

        return model

    def _resolve_directory(self, directory):
        if directory is None:
            if self.data_folder is None:
                raise ValueError("No directory specified for saving.")
            directory = self.data_folder
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        return directory

    def _write_metadata(self, directory):
        metadata = {
            "temperature": self.temperature,
            "truncation_distance": self.truncation_distance,
            "online_scale_factor": self.online_scale_factor,
            "offline_max_strength": self.offline_max_strength,
            "online_max_strength": self.online_max_strength,
            "prior_strength": self.prior_strength,
            "prior_mean": self.prior_mean,
            "max_neighbor_candidates": self.max_neighbor_candidates,
            "demo_odds_margin": self.demo_odds_margin,
            "max_single_demo_weight": self.max_single_demo_weight,
            "feature_key": self.feature_key,
        }
        with open(directory / "config.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f)

    def _write_offline_data(self, directory):
        if self.X_off is None or self.y_off is None:
            return
        np.savez_compressed(directory / "offline_data.npz", X=self.X_off, y=self.y_off)

    def _write_online_data(self, directory):
        path = directory / "online_data.npz"
        if self.X_on is None or self.X_on.shape[0] == 0:
            if path.exists():
                path.unlink()
            return
        point_weights = self.w_on
        if point_weights is None:
            point_weights = np.full(self.y_on.shape, self.online_scale_factor, dtype=float)
        np.savez_compressed(path, X=self.X_on, y=self.y_on, w=point_weights)

    def clear_online_data(self):
        """Clear online evidence in memory and remove its persisted data file."""
        feature_dim = 0 if self.X_off is None else self.X_off.shape[1]
        dtype = float if self.X_off is None else self.X_off.dtype
        self.X_on = np.zeros((0, feature_dim), dtype=dtype)
        self.y_on = np.zeros((0,), dtype=float)
        self.w_on = np.zeros((0,), dtype=float)
        self.tree_on = None
        if self.data_folder is not None:
            self._save_online_data()

    def _save_metadata(self):
        if self.data_folder is None:
            return
        directory = self._resolve_directory(None)
        self._write_metadata(directory)

    def _save_offline_data(self):
        if self.data_folder is None:
            return
        directory = self._resolve_directory(None)
        self._write_offline_data(directory)

    def _save_online_data(self):
        if self.data_folder is None:
            return
        directory = self._resolve_directory(None)
        self._write_online_data(directory)
