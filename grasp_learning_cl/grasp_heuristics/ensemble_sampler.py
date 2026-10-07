import time


class EnsembleSampler:
    default_sampler_keys = [
        "antipodal",
        "edge",
        "lowquality",
        "side",
        "topdown",
        "sideedge",
        "topdownedge",
        "vgn",
    ]

    def __init__(self, sampler_keys: list[str] | None = None) -> None:
        if sampler_keys is None:
            sampler_keys = list(self.default_sampler_keys)
        if not sampler_keys:
            raise ValueError("EnsembleSampler requires at least one sampler key.")

        self.sampler_keys = list(sampler_keys)
        self.samplers = [self._make_sampler(key) for key in self.sampler_keys]

    def __call__(self, cloud_objects, cloud_collisions: None, n_grasps: int, debug: bool = False) -> list[dict]:
        start_time = time.perf_counter()
        if n_grasps <= 0:
            return []

        base, remainder = divmod(int(n_grasps), len(self.samplers))
        grasps: list[dict] = []

        for idx, sampler in enumerate(self.samplers):
            sampler_grasps = base + (1 if idx < remainder else 0)
            if sampler_grasps <= 0:
                continue

            grasps.extend(
                sampler(
                    cloud_objects=cloud_objects,
                    cloud_collisions=cloud_collisions,
                    n_grasps=sampler_grasps,
                    debug=debug,
                )
            )

        if debug:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            print(f"{self.__class__.__name__}: time {elapsed_ms:.2f} ms")

        return grasps

    @staticmethod
    def _make_sampler(key: str):
        from grasp_learning_cl.grasp_heuristics.antipodal_grasp_sampler import AntipodalGraspSampler, AntipodalGraspSamplerFast
        from grasp_learning_cl.grasp_heuristics.edge_grasp_sampler import EdgeGraspSampler, EdgeGraspSamplerFast
        from grasp_learning_cl.grasp_heuristics.low_quality_sampler import LowQualitySampler
        from grasp_learning_cl.grasp_heuristics.side_grasp_sampler import SideGraspSampler, SideGraspSamplerFast
        from grasp_learning_cl.grasp_heuristics.top_down_grasp_sampler import TopDownGraspSampler, TopDownGraspSamplerFast
        from grasp_learning_cl.grasp_heuristics.side_edge_grasp_sampler import SideEdgeGraspSampler
        from grasp_learning_cl.grasp_heuristics.top_down_edge_grasp_sampler import TopDownEdgeGraspSampler
        from grasp_learning_cl.grasp_heuristics.vgn_grasp_sampler import VGNGraspSampler

        sampling_classes = {
            "antipodal": AntipodalGraspSampler,
            "antipodal_fast": AntipodalGraspSamplerFast,
            "edge": EdgeGraspSampler,
            "edge_fast": EdgeGraspSamplerFast,
            "lowquality": LowQualitySampler,
            "side": SideGraspSampler,
            "side_fast": SideGraspSamplerFast,
            "topdown": TopDownGraspSampler,
            "topdown_fast": TopDownGraspSamplerFast,
            "sideedge": SideEdgeGraspSampler,
            "topdownedge": TopDownEdgeGraspSampler,
            "vgn": VGNGraspSampler,
        }
        return sampling_classes[key]()


class EnsembleSamplerHQ(EnsembleSampler):
    default_sampler_keys = [
        "antipodal",
        "edge",
        # "side",
        "topdown",
    ]


class EnsembleSamplerHQFast(EnsembleSampler):
    default_sampler_keys = [
        "edge_fast",
        "side_fast",
        "topdown_fast",
    ]