from grasp_learning_cl.grasp_heuristics.antipodal_grasp_sampler import AntipodalGraspSampler, AntipodalGraspSamplerFast
from grasp_learning_cl.grasp_heuristics.ensemble_sampler import EnsembleSampler, EnsembleSamplerHQ, EnsembleSamplerHQFast
from grasp_learning_cl.grasp_heuristics.edge_grasp_sampler import EdgeGraspSampler, EdgeGraspSamplerFast
from grasp_learning_cl.grasp_heuristics.low_quality_sampler import LowQualitySampler
from grasp_learning_cl.grasp_heuristics.side_grasp_sampler import SideGraspSampler, SideGraspSamplerFast
from grasp_learning_cl.grasp_heuristics.top_down_grasp_sampler import TopDownGraspSampler, TopDownGraspSamplerFast
from grasp_learning_cl.grasp_heuristics.side_edge_grasp_sampler import SideEdgeGraspSampler
from grasp_learning_cl.grasp_heuristics.top_down_edge_grasp_sampler import TopDownEdgeGraspSampler
from grasp_learning_cl.grasp_heuristics.vgn_grasp_sampler import VGNGraspSampler


def get_sampler(key: str | list[str] | tuple[str, ...]):
    if isinstance(key, (list, tuple)):
        return EnsembleSampler(list(key))

    SamplingClasses = {
        "antipodal": AntipodalGraspSampler,
        "antipodal_fast": AntipodalGraspSamplerFast,
        "edge": EdgeGraspSampler,
        "edge_fast": EdgeGraspSamplerFast,
        "lowquality": LowQualitySampler,
        "side": SideGraspSampler,
        "side_fast": SideGraspSamplerFast,
        "topdown": TopDownGraspSampler,
        "topdown_fast": TopDownGraspSamplerFast,
        "topdownedge": TopDownEdgeGraspSampler,
        "sideedge": SideEdgeGraspSampler,
        "vgn": VGNGraspSampler,
        "ensemble": EnsembleSampler,
        "ensemblehq": EnsembleSamplerHQ,
        "ensemblehqfast": EnsembleSamplerHQFast,
    }
    return SamplingClasses[key]()
