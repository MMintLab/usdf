import argparse
import os

import numpy as np
import trimesh
from tqdm import trange

import transforms3d as tf3d

import mmint_utils
from usdf.utils.sdf_utils import generate_sdf_data


def generate_sdf_z_rot(dataset_cfg: dict, split: str, vis: bool = False):
    meshes_dir = dataset_cfg["meshes_dir"]
    dataset_dir = dataset_cfg["dataset_dir"]
    N_angles = dataset_cfg["N_angles"]  # Number of angles to generate SDF data from.
    N_random = dataset_cfg["N_random"]  # Number of random points to sample around mesh.
    N_off_surface = dataset_cfg["N_off_surface"]  # Number of points to sample off surface.
    off_surface_sigma_a = dataset_cfg[
        "off_surface_sigma_a"]  # Standard deviation of Gaussian to sample off surface points.
    off_surface_sigma_b = dataset_cfg[
        "off_surface_sigma_b"]  # Standard deviation of Gaussian to sample off surface points.
    sdfs_dir = os.path.join(dataset_dir, "sdfs")
    mmint_utils.make_dir(sdfs_dir)

    # Load split info.
    split_fn = os.path.join(dataset_dir, "splits", dataset_cfg["splits"][split])
    meshes = np.atleast_1d(np.loadtxt(split_fn, dtype=str))
    meshes = [m.replace(".obj", "") for m in meshes]

    with trange(len(meshes) * N_angles) as pbar:
        for idx in range(len(meshes)):
            mesh_name = meshes[idx]

            # Load the source mesh used to generate SDF.
            source_mesh_fn = os.path.join(meshes_dir, mesh_name + ".obj")
            source_mesh = trimesh.load(source_mesh_fn)

            # Create the SDF directory.
            example_sdf_dir = os.path.join(sdfs_dir, mesh_name)
            mmint_utils.make_dir(example_sdf_dir)

            for angle_idx in range(N_angles):
                example_sdf_angle_dir = os.path.join(example_sdf_dir, "angle_%d" % angle_idx)
                mmint_utils.make_dir(example_sdf_angle_dir)

                # Rotate source mesh.
                rot_mesh = source_mesh.copy()
                object_pose = np.eye(4)
                object_pose[:3, :3] = tf3d.euler.euler2mat(0, 0, angle_idx * 2 * np.pi / N_angles, axes="sxyz")
                rot_mesh.apply_transform(object_pose)

                # Generate SDF data for rotated mesh.
                sdf_dict = generate_sdf_data(rot_mesh, N_random, N_off_surface, off_surface_sigma_a,
                                             off_surface_sigma_b, vis)
                sdf_dict["object_pose"] = object_pose

                # Save result.
                mmint_utils.save_gzip_pickle(sdf_dict, os.path.join(example_sdf_angle_dir, "sdf_data.pkl.gzip"))

                pbar.update()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Gen SDF for z rotations of object.")
    parser.add_argument("dataset_cfg_fn", type=str, help="Path to dataset config file.")
    parser.add_argument("split", type=str, help="Split to generate.")
    parser.add_argument("-v", "--vis", action="store_true", help="Visualize the SDFs.")
    args = parser.parse_args()

    dataset_cfg_ = mmint_utils.load_cfg(args.dataset_cfg_fn)["data"][args.split]

    generate_sdf_z_rot(dataset_cfg_, args.split, args.vis)
