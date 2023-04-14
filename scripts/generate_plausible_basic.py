import argparse
import os

import numpy as np
import torch
from tqdm import tqdm

import mmint_utils
from usdf.utils import utils


def generate_plausible_basic(dataset_cfg: dict, split: str, vis: bool = True):
    """
    Generate plausible pointclouds for each mesh in the dataset matching to the partial views generated from all other
    meshes.
    """
    d = "cuda" if torch.cuda.is_available() else "cpu"
    surface_N = 1000  # Number of points to use in partial pointcloud.
    N_angles = dataset_cfg["N_angles"]  # Number of angles each mesh is rendered from.

    meshes_dir = dataset_cfg["meshes_dir"]
    dataset_dir = dataset_cfg["dataset_dir"]
    partials_dir = os.path.join(dataset_dir, "partials")
    plausibles_dir = os.path.join(dataset_dir, "plausibles")
    mmint_utils.make_dir(plausibles_dir)

    # Load split info.
    split_fn = os.path.join(meshes_dir, "splits", dataset_cfg["splits"][split])
    meshes = np.atleast_1d(np.loadtxt(split_fn, dtype=str))
    meshes = [m.replace(".obj", "") for m in meshes]

    # Load partial pointclouds for each mesh at each angle.
    partial_pointclouds = dict()
    with tqdm(total=len(meshes) * N_angles) as pbar:
        for partial_mesh_name in meshes:
            mesh_partial_pointclouds = dict()
            for angle_idx in range(N_angles):
                # Load the partial view information for the given angle.
                partial_angle_dir = os.path.join(partials_dir, partial_mesh_name, "angle_%d" % angle_idx)
                pointcloud = utils.load_pointcloud(os.path.join(partial_angle_dir, "pointcloud.ply"))[:, :3]
                angle = mmint_utils.load_gzip_pickle(os.path.join(partial_angle_dir, "info.pkl.gzip"))["angle"]

                # Downsample partial surface pointcloud.
                # np.random.shuffle(pointcloud)
                # pointcloud = torch.from_numpy(pointcloud[:surface_N]).to(d).float()

                mesh_partial_pointclouds[angle] = torch.from_numpy(pointcloud).to(d).float()

                pbar.update(1)
            partial_pointclouds[partial_mesh_name] = mesh_partial_pointclouds

    # Evaluate distances between partial pointclouds.
    with tqdm(total=len(meshes) * N_angles * len(meshes) * N_angles) as pbar:
        for partial_mesh_name in meshes:
            mesh_partial_pointclouds = partial_pointclouds[partial_mesh_name]

            for angle, partial_pc in mesh_partial_pointclouds.items():

                # Compare to every other partial pointcloud.
                for match_mesh_name in meshes:
                    match_partial_pointclouds = partial_pointclouds[match_mesh_name]

                    for match_angle, match_partial_pc in match_partial_pointclouds.items():
                        pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate plausibles by comparing partial views.")
    parser.add_argument("dataset_cfg", type=str, help="Path to dataset config file.")
    parser.add_argument("split", type=str, help="Split to use.")
    parser.add_argument("--vis", "-v", action="store_true", help="Visualize results.")
    parser.set_defaults(vis=False)
    args = parser.parse_args()

    dataset_cfg_ = mmint_utils.load_cfg(args.dataset_cfg)["data"][args.split]

    generate_plausible_basic(dataset_cfg_, args.split, args.vis)
