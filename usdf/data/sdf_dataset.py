import os

import numpy as np
import torch.utils.data
import trimesh

import mmint_utils
from vedo import Plotter, Points, Mesh
import transforms3d as tf3d

from usdf.utils import vedo_utils, utils


class SDFDataset(torch.utils.data.Dataset):

    def __init__(self, dataset_cfg: dict, split: str, transform=None):
        super().__init__()
        self.N_angles = dataset_cfg["N_angles"]
        N_sdf = 10000 + 10000  # TODO: config
        self.meshes_dir = dataset_cfg["meshes_dir"]
        self.dataset_dir = dataset_cfg["dataset_dir"]
        partials_dir = os.path.join(self.dataset_dir, "partials")
        sdfs_dir = os.path.join(self.dataset_dir, "sdfs")

        # Load split info.
        split_fn = os.path.join(self.meshes_dir, "splits", dataset_cfg["splits"][split])
        meshes = np.atleast_1d(np.loadtxt(split_fn, dtype=str))
        self.meshes = [m.replace(".obj", "") for m in meshes]

        # Data arrays.
        self.example_idcs = []  # Index of the partial view.
        self.mesh_idcs = []  # Index of the mesh for this example.
        self.query_points = []  # Query points for SDF.
        self.sdf = []  # SDF values at query points.
        self.object_pose = []  # Object pose.
        self.partial_pointcloud = []  # Partial pointcloud.

        # Load data.
        for mesh_idx, partial_mesh_name in enumerate(self.meshes):
            example_sdf_dir = os.path.join(sdfs_dir, partial_mesh_name)
            example_partial_dir = os.path.join(partials_dir, partial_mesh_name)
            for angle_idx in range(self.N_angles):
                example_angle_sdf_dir = os.path.join(example_sdf_dir, "angle_%d" % angle_idx)
                example_angle_partial_dir = os.path.join(example_partial_dir, "angle_%d" % angle_idx)

                sdf_fn = os.path.join(example_angle_sdf_dir, "sdf_data.pkl.gzip")
                sdf_data = mmint_utils.load_gzip_pickle(sdf_fn)

                partial_fn = os.path.join(example_angle_partial_dir, "pointcloud.ply")
                partial_pointcloud = utils.load_pointcloud(partial_fn)

                # Append to data arrays.
                self.example_idcs.append(mesh_idx * self.N_angles + angle_idx)
                self.mesh_idcs.append(mesh_idx)
                self.query_points.append(sdf_data["query_points"])
                self.sdf.append(sdf_data["sdf_values"])
                self.object_pose.append(sdf_data["object_pose"])
                self.partial_pointcloud.append(partial_pointcloud)

    def __len__(self):
        return len(self.example_idcs)

    def __getitem__(self, index: int):
        data_dict = {
            "partial_pointcloud": self.partial_pointcloud[index],
            "example_idx": np.array([self.example_idcs[index]]),
            "mesh_idx": np.array([self.mesh_idcs[index]]),
            "query_points": self.query_points[index],
            "object_pose": self.object_pose[index],
            "sdf": self.sdf[index],
        }
        return data_dict

    def visualize_item(self, data_dict: dict):
        # Load mesh for this example.
        mesh_name = self.meshes[data_dict["example_idx"][0] // self.N_angles]
        mesh_fn = os.path.join(self.meshes_dir, mesh_name + ".obj")
        rot_mesh = trimesh.load(mesh_fn)
        rot_mesh.apply_transform(data_dict["object_pose"])

        plt = Plotter((1, 2))
        plt.at(0).show(Mesh([rot_mesh.vertices, rot_mesh.faces], c="y"),
                       Points(data_dict["partial_pointcloud"], c="b"),
                       vedo_utils.draw_origin(scale=0.1))
        plt.at(1).show(Points(data_dict["query_points"][data_dict["sdf"] < 0.0]),
                       vedo_utils.draw_origin(scale=0.1))
        plt.interactive().close()