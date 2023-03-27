import torch.utils.data
import torch
import os
import numpy as np


class UncertaintyDataset(torch.utils.data.Dataset):

    def __init__(self, dataset_cfg: dict, split: str, transform=None):
        super().__init__()

        meshes_dir = dataset_cfg["meshes_dir"]
        dataset_dir = dataset_cfg["dataset_dir"]
        partials_dir = os.path.join(dataset_dir, "partials")
        plausibles_dir = os.path.join(dataset_dir, "plausibles")

        # Load split info.
        split_fn = os.path.join(meshes_dir, "splits", dataset_cfg["splits"][split])
        meshes = np.atleast_1d(np.loadtxt(split_fn, dtype=str))
        meshes = [m.replace(".obj", "") for m in meshes]

        # Data arrays.
        self.example_idcs = []  # Index of the partial view.
        self.query_points = []  # Query points for SDF.
        self.sdf = []  # SDF values at query points.

        # Load data.

    def __getitem__(self, index: int):
        pass
