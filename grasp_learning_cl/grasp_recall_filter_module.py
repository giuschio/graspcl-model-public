import numpy as np


def filter_recalled_0(recalled_grasp_proposals, feature_key):
    recall_best = {}
    for grasp in recalled_grasp_proposals:
        original = np.asarray(grasp["metadata"][feature_key], dtype=float).reshape(-1)
        distance = _feature_distance(grasp, feature_key)
        key = tuple(original)
        if key not in recall_best or distance < recall_best[key][0]:
            recall_best[key] = (distance, grasp)
    return [grasp for _, grasp in recall_best.values()]


def filter_recalled_1(recalled_grasp_proposals, feature_key, temperature=0.35):
    return [
        grasp
        for grasp in recalled_grasp_proposals
        if _feature_distance(grasp, feature_key) < temperature
    ]


def filter_recalled_2(recalled_grasp_proposals, feature_key, abs_margin=0.1):
    grouped = {}
    for grasp in recalled_grasp_proposals:
        original = np.asarray(grasp["metadata"][feature_key], dtype=float).reshape(-1)
        distance = _feature_distance(grasp, feature_key)
        grouped.setdefault(tuple(original), []).append((distance, grasp))

    kept = []
    for matches in grouped.values():
        best_distance = min(distance for distance, _ in matches)
        threshold = best_distance + abs_margin
        kept.extend(grasp for distance, grasp in matches if distance <= threshold)
    return kept


def _feature_distance(grasp, feature_key):
    original = np.asarray(grasp["metadata"][feature_key], dtype=float).reshape(-1)
    current = np.asarray(grasp[feature_key], dtype=float).reshape(-1)
    distance = float(np.linalg.norm(current - original))
    grasp["recall_distance"] = distance
    return distance


FILTERING_FNS = {
    "filter_recalled_0": filter_recalled_0,
    "filter_recalled_1": filter_recalled_1,
    "filter_recalled_2": filter_recalled_2,
}


class GraspRecallFilterModule:
    def __init__(
        self,
        filtering_fn_key: str = "filter_recalled_2",
        filtering_fn_params: dict | None = None,
        feature_key: str = "feature_encoder",
    ) -> None:
        self.filtering_fn_key = filtering_fn_key
        if filtering_fn_params is None:
            filtering_fn_params = {"abs_margin": 0.1}
        self.filtering_fn_params = dict(filtering_fn_params)
        self.feature_key = feature_key

    def filter_grasps(self, recalled_grasp_proposals):
        if self.filtering_fn_key not in FILTERING_FNS:
            raise ValueError(
                f"Unknown filtering_fn_key '{self.filtering_fn_key}'. "
                f"Expected one of {sorted(FILTERING_FNS)}."
            )
        return FILTERING_FNS[self.filtering_fn_key](
            recalled_grasp_proposals,
            self.feature_key,
            **self.filtering_fn_params,
        )
