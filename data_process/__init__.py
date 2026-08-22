from data_process.geometry import project_points, transform_points, unproject_depth_to_points, voxel_downsample, voxel_indices
from data_process.reference_frames import RefSelectionResult, occupancy_from_frame, iou_occupancy, select_reference_frames
from data_process.sample_indices import sample_episode_indices, sample_frame_indices
from data_process.types import EpisodeIndices, SampleIndices, VideoGeometry


class DataConfig:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class SampleConfig(DataConfig):
    pass


CONFIG = {}


def get_sample_config(**kwargs):
    return SampleConfig(**kwargs)


def get_episode_config(**kwargs):
    return SampleConfig(**kwargs)


def build_training_sample(**kwargs):
    return kwargs


__all__ = [
    "CONFIG",
    "DataConfig",
    "SampleConfig",
    "get_sample_config",
    "get_episode_config",
    "SampleIndices",
    "VideoGeometry",
    "build_training_sample",
    "EpisodeIndices",
    "sample_frame_indices",
    "sample_episode_indices",
    "unproject_depth_to_points",
    "transform_points",
    "project_points",
    "voxel_downsample",
    "voxel_indices",
    "occupancy_from_frame",
    "iou_occupancy",
    "RefSelectionResult",
    "select_reference_frames",
]
