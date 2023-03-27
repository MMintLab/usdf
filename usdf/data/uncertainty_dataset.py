import mmint_utils
import torch.utils.data
import torch
import os
import numpy as np


class UncertaintyDataset(torch.utils.data.Dataset):

    def __init__(self, dataset_cfg: dict, split: str, transform=None):
        super().__init__()
        N_angles = 8  # TODO: config
        N_sdf = 10000 + 10000  # TODO: config
        meshes_dir = dataset_cfg["meshes_dir"]
        dataset_dir = dataset_cfg["dataset_dir"]
        partials_dir = os.path.join(dataset_dir, "partials")
        sdfs_dir = os.path.join(dataset_dir, "sdfs")

        # Load split info.
        split_fn = os.path.join(meshes_dir, "splits", dataset_cfg["splits"][split])
        meshes = np.atleast_1d(np.loadtxt(split_fn, dtype=str))
        meshes = [m.replace(".obj", "") for m in meshes]

        # Data arrays.
        self.example_idcs = []  # Index of the partial view.
        self.query_points = []  # Query points for SDF.
        self.sdf = []  # SDF values at query points.

        # Load data.
        for mesh_idx, partial_mesh_name in enumerate(meshes):
            example_sdf_dir = os.path.join(sdfs_dir, partial_mesh_name)
            for angle_idx in range(N_angles):
                example_angle_sdf_dir = os.path.join(example_sdf_dir, "angle_%d" % angle_idx)

                example_idx = mesh_idx * N_angles + angle_idx

                query_points = None
                sdf_labels = np.empty((N_sdf, 0), dtype=np.float32)

                for match_mesh_name in meshes:
                    # Load the SDF values.
                    sdf_fn = os.path.join(example_angle_sdf_dir, match_mesh_name + ".pkl.gzip")
                    sdf_data = mmint_utils.load_gzip_pickle(sdf_fn)

                    # Append to data arrays.
                    query_points = sdf_data["query_points"]
                    sdf_labels = np.concatenate(sdf_labels, sdf_data["sdf_labels"], axis=1)

                # Append to data arrays.
                self.example_idcs.append(example_idx)
                self.query_points.append(query_points)
                self.sdf.append(sdf_labels)

    def __len__(self):
        return len(self.example_idcs)

    def __getitem__(self, index: int):
        data_dict = {
            "example_idx": self.example_idcs[index],
            "query_points": self.query_points[index],
            "sdf": self.sdf[index],
        }

        return data_dict
