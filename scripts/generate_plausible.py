import argparse
import copy
import os

import chsel
import mmint_utils
import numpy as np
import pytorch_volumetric as pv
import pytorch_kinematics as pk
import torch
import trimesh
from tqdm import tqdm
import open3d as o3d
from usdf import utils

from vedo import Plotter, Points, Mesh


def generate_plausible(dataset_cfg: dict, split: str, vis: bool = True):
    # TODO: Move to config.
    N = 2000  # Number of points to use in partial pointcloud.
    B = 32  # Batch size for CHSEL.
    use_cached_sdf: bool = True
    d = "cuda" if torch.cuda.is_available() else "cpu"

    meshes_dir = dataset_cfg["meshes_dir"]
    dataset_dir = dataset_cfg["dataset_dir"]
    partials_dir = os.path.join(dataset_dir, "partials")
    plausibles_dir = os.path.join(dataset_dir, "plausibles")
    mmint_utils.make_dir(plausibles_dir)

    # Load split info.
    split_fn = os.path.join(meshes_dir, "splits", split + ".txt")
    mesh_fns = np.loadtxt(split_fn, dtype=str)

    # Load partial data.
    partials_fns = os.listdir(partials_dir)

    # Apply CHSEL on each mesh in dataset to attempt registration to partial view.
    for mesh_fn in tqdm(mesh_fns):
        mesh_fn_full = os.path.join(meshes_dir, mesh_fn)
        mesh_tri: trimesh.Trimesh = trimesh.load(mesh_fn_full)
        obj = pv.MeshObjectFactory(mesh_fn_full)
        sdf = pv.MeshSDF(obj)
        if use_cached_sdf:
            sdf = pv.CachedSDF(os.path.splitext(mesh_fn)[0], resolution=0.02,
                               range_per_dim=obj.bounding_box(padding=0.1), gt_sdf=sdf, device=d)

        results = dict()
        for partial_fn in partials_fns:
            partial_views_dirs = os.listdir(os.path.join(partials_dir, partial_fn))
            partial_views_dirs.sort(key=lambda x: float(x))
            source_mesh_fn = os.path.join(meshes_dir, partial_fn + ".obj")
            source_mesh = trimesh.load(source_mesh_fn)

            results[partial_fn] = dict()

            for partial_view_dir in partial_views_dirs:
                partial_angle_dir = os.path.join(partials_dir, partial_fn, partial_view_dir)
                pointcloud = utils.load_pointcloud(os.path.join(partial_angle_dir, "pointcloud.ply"))[:, :3]
                free_pointcloud = utils.load_pointcloud(os.path.join(partial_angle_dir, "free_pointcloud.ply"))[:, :3]
                angle = mmint_utils.load_gzip_pickle(os.path.join(partial_angle_dir, "info.pkl.gzip"))["angle"]

                # Downsample partial pointcloud.
                np.random.shuffle(pointcloud)
                pointcloud = torch.from_numpy(pointcloud[:N]).to(d).float()

                # Filter free points to lie in ball around object.
                free_pointcloud = free_pointcloud[np.linalg.norm(free_pointcloud, axis=1) < 1.1]
                np.random.shuffle(free_pointcloud)
                free_pointcloud = torch.from_numpy(free_pointcloud[:N]).to(d).float()

                # Setup semantics of pointcloud.
                sem_sdf = np.zeros((N,))
                sem_free = [chsel.SemanticsClass.FREE] * N

                # Combine pointclouds to send to CHSEL.
                points = torch.cat([pointcloud, free_pointcloud], dim=0)
                semantics = sem_sdf.tolist() + sem_free

                # Visualize the points being used.
                if vis:
                    plt = Plotter()
                    # Rotate source mesh here based on angle.
                    source_mesh_angle = source_mesh.copy().apply_transform(
                        trimesh.transformations.rotation_matrix(angle, [0, 0, 1]))
                    plt.at(0).show(Points(pointcloud.cpu().numpy(), c="green"),
                                   Points(free_pointcloud.cpu().numpy(), c="blue", alpha=0.2),
                                   Mesh([source_mesh_angle.vertices, source_mesh_angle.faces]))

                # Initialize transforms to rotations around z.
                euler_angles = torch.zeros((B, 3), device=d)
                euler_angles[:, 2] = torch.linspace(0.0, 2 * np.pi, B + 1, device=d)[:-1]
                random_init_tsf = pk.Transform3d(pos=torch.zeros((B, 3), device=d),
                                                 rot=pk.euler_angles_to_matrix(euler_angles, "XYZ"), device=d)
                random_init_tsf = random_init_tsf.get_matrix()

                registration = chsel.CHSEL(sdf, points, semantics, qd_iterations=100, free_voxels_resolution=0.02)
                res, all_solutions = registration.register(iterations=15, batch=B, initial_tsf=random_init_tsf)

                if vis:
                    # print the sorted RMSE for each iteration
                    print(torch.sort(registration.res_history[-1].rmse).values)

                    # res.RTs.R, res.RTs.T, res.RTs.s are the similarity transform parameters
                    # get 30 4x4 transform matrix for homogeneous coordinates
                    world_to_link = chsel.solution_to_world_to_link_matrix(registration.res_history[-1])
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
                        Points(pointcloud.cpu().numpy(), c="green", alpha=0.2),
                        Points(free_pointcloud.cpu().numpy(), c="blue", alpha=0.2),
                        Mesh([source_mesh_angle.vertices, source_mesh_angle.faces], c="red"),
                        *mesh_estimates
                    )

                results[partial_fn][angle] = {
                    "res": res,
                    "all_solutions": all_solutions
                }

        # Save results.
        mmint_utils.save_gzip_pickle(results, os.path.join(plausibles_dir, mesh_fn + ".pkl"))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate plausible poses for partial views.")
    parser.add_argument("dataset_cfg_fn", type=str, help="Dataset config file.")
    parser.add_argument("split", type=str, help="Split to generate plausible poses for.")
    parser.add_argument("-v", "--vis", action="store_true", help="Visualize results.")
    args = parser.parse_args()

    dataset_cfg_ = mmint_utils.load_cfg(args.dataset_cfg_fn)
    generate_plausible(dataset_cfg_, args.split, args.vis)
