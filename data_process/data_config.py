from __future__ import annotations


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
