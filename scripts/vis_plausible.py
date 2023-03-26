import argparse
import os

import chsel
import mmint_utils
import numpy as np
import trimesh
from usdf import utils
from vedo import Mesh, Plotter, Points


def vis_plausible(dataset_cfg: dict, split: str):
    meshes_dir = dataset_cfg["meshes_dir"]
    dataset_dir = dataset_cfg["dataset_dir"]
    partials_dir = os.path.join(dataset_dir, "partials")
    plausibles_dir = os.path.join(dataset_dir, "plausibles")

    # Load split info.
    split_fn = os.path.join(meshes_dir, "splits", split + ".txt")
    mesh_fns = np.loadtxt(split_fn, dtype=str)

    plausibles_fns = [f for f in os.listdir(plausibles_dir) if ".obj.pkl" in f]

    for plausible_fn in plausibles_fns:
        plausible_dict = mmint_utils.load_gzip_pickle(os.path.join(plausibles_dir, plausible_fn))

        mesh_fn = os.path.join(meshes_dir, os.path.splitext(plausible_fn)[0])
        mesh_tri = trimesh.load(mesh_fn)

        for partial_fn in plausible_dict.keys():
            source_mesh_fn = os.path.join(meshes_dir, partial_fn + ".obj")
            source_mesh = trimesh.load(source_mesh_fn)
            for angle in plausible_dict[partial_fn].keys():
                partial_angle_dir = os.path.join(partials_dir, partial_fn, f"{angle:.2f}")
                pointcloud = utils.load_pointcloud(os.path.join(partial_angle_dir, "pointcloud.ply"))[:, :3]
                free_pointcloud = utils.load_pointcloud(os.path.join(partial_angle_dir, "free_pointcloud.ply"))[:,
                                  :3]
                angle = mmint_utils.load_gzip_pickle(os.path.join(partial_angle_dir, "info.pkl.gzip"))["angle"]

                # Downsample partial pointcloud.
                np.random.shuffle(pointcloud)
                pointcloud = pointcloud[:2000]

                # Filter free points to lie in ball around object.
                free_pointcloud = free_pointcloud[np.linalg.norm(free_pointcloud, axis=1) < 1.1]
                np.random.shuffle(free_pointcloud)
                free_pointcloud = free_pointcloud[:2000]

                res_dict = plausible_dict[partial_fn][angle]
                res = res_dict["res"]

                # res.RTs.R, res.RTs.T, res.RTs.s are the similarity transform parameters
                # get 30 4x4 transform matrix for homogeneous coordinates
                world_to_link = chsel.solution_to_world_to_link_matrix(res)
                link_to_world = world_to_link.inverse()

                # Transform meshes based on results.
                mesh_estimates = []
                for i in range(link_to_world.shape[0]):
                    mesh_tri_est = mesh_tri.copy().apply_transform(link_to_world[i].cpu().numpy())
                    mesh_estimates.append(Mesh([mesh_tri_est.vertices, mesh_tri_est.faces], alpha=0.5))

                plt = Plotter()
                source_mesh_angle = source_mesh.copy().apply_transform(
                    trimesh.transformations.rotation_matrix(angle, [0, 0, 1]))
                plt.at(0).show(
                    Points(pointcloud, c="green", alpha=0.2),
                    Points(free_pointcloud, c="blue", alpha=0.2),
                    Mesh([source_mesh_angle.vertices, source_mesh_angle.faces], c="red"),
                    *mesh_estimates
                )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Visualize plausible reconstructions.")
    parser.add_argument("dataset_cfg_fn", type=str, help="Path to dataset config file.")
    parser.add_argument("split", type=str, help="Split to visualize.")
    args = parser.parse_args()

    dataset_cfg_ = mmint_utils.load_cfg(args.dataset_cfg_fn)
    vis_plausible(dataset_cfg_, args.split)
