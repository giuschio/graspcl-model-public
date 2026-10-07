def __getattr__(name):
    if name == "PointNetGraspLightningModule":
        from grasp_learning_cl.grasp_classification.encoders.pointnet_grasp import PointNetGraspLightningModule

        return PointNetGraspLightningModule
    raise AttributeError(name)

__all__ = [
    "PointNetGraspLightningModule",
]
