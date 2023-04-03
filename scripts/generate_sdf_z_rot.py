import argparse
import os

import numpy as np
import trimesh
from tqdm import trange
from vedo import Plotter, Mesh, Points

from usdf.utils import vedo_utils
from usdf.utils.sdf_utils import get_sdf_query_points, get_sdf_values
import transforms3d as tf3d

import mmint_utils


def generate_sdf_z_rot(dataset_cfg: dict, split: str, vis: bool = False):
    meshes_dir = dataset_cfg["meshes_dir"]
    dataset_dir = dataset_cfg["dataset_dir"]
    N_angles = dataset_cfg["N_angles"]  # Number of angles to generate SDF data from.
    sdfs_dir = os.path.join(dataset_dir, "sdfs")
    mmint_utils.make_dir(sdfs_dir)

    # Load split info.
    split_fn = os.path.join(meshes_dir, "splits", dataset_cfg["splits"][split])
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

                # Sample the SDF points to evaluate.
                query_points = get_sdf_query_points(rot_mesh, noise=0.01)

                # Calculate the SDF values.
                sdf_values = get_sdf_values(rot_mesh, query_points)

                if vis:
                    plt = Plotter((1, 2))
                    plt.at(0).show(Mesh([rot_mesh.vertices, rot_mesh.faces], c="red"),
                                   vedo_utils.draw_origin(scale=0.1))
                    plt.at(1).show(Points(query_points[sdf_values < 0.0]), vedo_utils.draw_origin(scale=0.1))
                    plt.interactive().close()

                # Save result.
                mmint_utils.save_gzip_pickle({
                    "query_points": query_points,
                    "sdf_values": sdf_values,
                    "object_pose": object_pose,
                }, os.path.join(example_sdf_angle_dir, "sdf_data.pkl.gzip"))

                pbar.update()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Gen SDF for z rotations of object.")
    parser.add_argument("dataset_cfg_fn", type=str, help="Path to dataset config file.")
    parser.add_argument("split", type=str, help="Split to generate.")
    parser.add_argument("-v", "--vis", action="store_true", help="Visualize the SDFs.")
    args = parser.parse_args()

    dataset_cfg_ = mmint_utils.load_cfg(args.dataset_cfg_fn)["data"][args.split]

    generate_sdf_z_rot(dataset_cfg_, args.split, args.vis)