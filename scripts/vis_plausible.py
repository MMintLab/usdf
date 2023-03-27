import argparse
import os

import chsel
import mmint_utils
import numpy as np
import torch
import trimesh
from usdf import utils
from vedo import Mesh, Plotter, Points
import torch.nn.functional as F


def vis_plausible(dataset_cfg: dict, split: str):
    # TODO: Move to config.
    N_angles = 8  # Number of angles each mesh is rendered from.
    order_by_rank = False
    meshes_dir = dataset_cfg["meshes_dir"]
    dataset_dir = dataset_cfg["dataset_dir"]
    partials_dir = os.path.join(dataset_dir, "partials")
    plausibles_dir = os.path.join(dataset_dir, "plausibles")

    # Load split info.
    split_fn = os.path.join(meshes_dir, "splits", dataset_cfg["splits"][split])
    meshes = np.atleast_1d(np.loadtxt(split_fn, dtype=str))
    meshes = [m.replace(".obj", "") for m in meshes]

    for partial_mesh_name in meshes:
        # Load the source mesh used to generate the partial view.
        source_mesh_fn = os.path.join(meshes_dir, partial_mesh_name + ".obj")
        source_mesh = trimesh.load(source_mesh_fn)

        # Directory for the plausibles of the partial view.
        example_plausibles_dir = os.path.join(plausibles_dir, partial_mesh_name)
        example_partials_dir = os.path.join(partials_dir, partial_mesh_name)
        for angle_idx in range(N_angles):
            # Directory for the plausibles of the partial view at the given angle.
            example_angle_plausibles_dir = os.path.join(example_plausibles_dir, "angle_%d" % angle_idx)
            example_angle_partials_dir = os.path.join(example_partials_dir, "angle_%d" % angle_idx)

            for match_mesh_name in meshes:
                # Load mesh used to match to partial view.
                mesh_fn_full = os.path.join(meshes_dir, match_mesh_name + ".obj")
                mesh_tri: trimesh.Trimesh = trimesh.load(mesh_fn_full)

                # Load the plausible results.
                res_dict = mmint_utils.load_gzip_pickle(
                    os.path.join(example_angle_plausibles_dir, match_mesh_name + ".pkl.gzip"))
                res = res_dict["res"]

                # Load the partial view.
                pointcloud = res_dict["pointcloud"]
                free_pointcloud = res_dict["free_pointcloud"]
                angle = res_dict["angle"]

                # res.RTs.R, res.RTs.T, res.RTs.s are the similarity transform parameters
                world_to_link = chsel.solution_to_world_to_link_matrix(res)
                link_to_world = world_to_link.inverse()

                if order_by_rank:
                    indices = torch.argsort(res.rmse)

                    for index in indices:
                        print("RMSE: %f" % res.rmse[index].item())
                        mesh_tri_est = mesh_tri.copy().apply_transform(link_to_world[index].cpu().numpy())
                        mesh_estimates = Mesh([mesh_tri_est.vertices, mesh_tri_est.faces], alpha=0.5)
                        plt = Plotter()
                        source_mesh_angle = source_mesh.copy().apply_transform(
                            trimesh.transformations.rotation_matrix(angle, [0, 0, 1]))
                        plt.at(0).show(
                            Points(pointcloud, c="green", alpha=0.1),
                            Points(free_pointcloud, c="blue", alpha=0.1),
                            Mesh([source_mesh_angle.vertices, source_mesh_angle.faces], c="red", alpha=0.2),
                            mesh_estimates
                        )
                else:
                    norm_inv_rmse = 1.0 - (res.rmse - res.rmse.min()) / (res.rmse.max() - res.rmse.min())

                    # Transform meshes based on results.
                    mesh_estimates = []
                    for i in range(link_to_world.shape[0]):
                        mesh_tri_est = mesh_tri.copy().apply_transform(link_to_world[i].cpu().numpy())
                        mesh_estimates.append(
                            Mesh([mesh_tri_est.vertices, mesh_tri_est.faces], alpha=norm_inv_rmse[i].item()))

                    plt = Plotter()
                    source_mesh_angle = source_mesh.copy().apply_transform(
                        trimesh.transformations.rotation_matrix(angle, [0, 0, 1]))
                    plt.at(0).show(
                        Points(pointcloud, c="green", alpha=0.5),
                        Points(free_pointcloud, c="blue", alpha=0.5),
                        Mesh([source_mesh_angle.vertices, source_mesh_angle.faces], c="red", alpha=0.2),
                        *mesh_estimates
                    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Visualize plausible reconstructions.")
    parser.add_argument("dataset_cfg_fn", type=str, help="Path to dataset config file.")
    parser.add_argument("split", type=str, help="Split to visualize.")
    args = parser.parse_args()

    dataset_cfg_ = mmint_utils.load_cfg(args.dataset_cfg_fn)["data"][args.split]
    vis_plausible(dataset_cfg_, args.split)
