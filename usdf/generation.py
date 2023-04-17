import torch
import torch.nn as nn


class BaseGenerator(object):
    """
    Base generator class.

    Generator is responsible for implementing an API generating the shared representations from
    various model classes. Not all representations need to necessarily be implemented.
    """

    def __init__(self, cfg: dict, model: nn.Module, generation_cfg: dict, device: torch.device = None):
        self.cfg = cfg
        self.model = model
        self.device = device
        self.generation_cfg = generation_cfg

        self.generates_mesh = False
        self.generates_pointcloud = False
        self.generates_slice = False

    def generate_mesh(self, data, metadata):
        raise NotImplementedError()

    def generate_pointcloud(self, data, metadata):
        raise NotImplementedError()

    def generate_slice(self, data, metadata):
        raise NotImplementedError()
