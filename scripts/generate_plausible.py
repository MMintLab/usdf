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
from usdf.utils import utils

from vedo import Plotter, Points, Mesh


def generate_plausible(dataset_cfg: dict, split: str, vis: bool = True):
    """
    Generate plausible pointclouds for each mesh in the dataset matching to the partial views generated from all other
    meshes.
    """
    # TODO: Move to config.
    surface_N = 1000  # Number of points to use in partial pointcloud.
    free_N = 2000  # Number of points to use in free pointcloud.
    B = 32  # Batch size for CHSEL.
    N_angles = 8  # Number of angles each mesh is rendered from.
    use_cached_sdf: bool = False
    d = "cuda" if torch.cuda.is_available() else "cpu"

    meshes_dir = dataset_cfg["meshes_dir"]
    dataset_dir = dataset_cfg["dataset_dir"]
    partials_dir = os.path.join(dataset_dir, "partials")
    plausibles_dir = os.path.join(dataset_dir, "plausibles")
    mmint_utils.make_dir(plausibles_dir)

    # Load split info.
    split_fn = os.path.join(meshes_dir, "splits", dataset_cfg["splits"][split])
    meshes = np.atleast_1d(np.loadtxt(split_fn, dtype=str))
    meshes = [m.replace(".obj", "") for m in meshes]

    # Apply CHSEL on each mesh in dataset to attempt registration to each partial view.
    with tqdm(total=len(meshes) * len(meshes) * N_angles) as pbar:

        # Outer loop: Meshes to match to partial views.
        for mesh_name in meshes:

            # Load mesh and build SDF.
            mesh_fn_full = os.path.join(meshes_dir, mesh_name + ".obj")
            mesh_tri: trimesh.Trimesh = trimesh.load(mesh_fn_full)
            obj = pv.MeshObjectFactory(mesh_fn_full)
            sdf = pv.MeshSDF(obj)
            if use_cached_sdf:
                sdf = pv.CachedSDF(mesh_name, resolution=0.02,
                                   range_per_dim=obj.bounding_box(padding=0.1), gt_sdf=sdf, device=d)

            # Inner loop: Partial views generated from all meshes.
            for partial_mesh_name in meshes:

                # Load the source mesh used to generate the partial view.
                source_mesh_fn = os.path.join(meshes_dir, partial_mesh_name + ".obj")
                source_mesh = trimesh.load(source_mesh_fn)

                # Directory for the plausibles of the partial view.
                plausibles_partial_dir = os.path.join(plausibles_dir, partial_mesh_name)
                mmint_utils.make_dir(plausibles_partial_dir)

                for angle_idx in range(N_angles):
                    # Directory for the plausibles of the partial view at the given angle.
                    plausibles_partial_angle_dir = os.path.join(plausibles_partial_dir, "angle_%d" % angle_idx)
                    mmint_utils.make_dir(plausibles_partial_angle_dir)

                    # Load the partial view information for the given angle.
                    partial_angle_dir = os.path.join(partials_dir, partial_mesh_name, "angle_%d" % angle_idx)
                    pointcloud = utils.load_pointcloud(os.path.join(partial_angle_dir, "pointcloud.ply"))[:, :3]
                    free_pointcloud = utils.load_pointcloud(os.path.join(partial_angle_dir, "free_pointcloud.ply"))[:,
                                      :3]
                    angle = mmint_utils.load_gzip_pickle(os.path.join(partial_angle_dir, "info.pkl.gzip"))["angle"]

                    # Downsample partial surface pointcloud.
                    np.random.shuffle(pointcloud)
                    pointcloud = torch.from_numpy(pointcloud[:surface_N]).to(d).float()

                    # Filter free points to lie in ball around object and downsample.
                    free_pointcloud = free_pointcloud[np.linalg.norm(free_pointcloud, axis=1) < 1.4]
                    np.random.shuffle(free_pointcloud)
                    # free_pointcloud = torch.from_numpy(free_pointcloud[:free_N]).to(d).float()
                    free_pointcloud = torch.from_numpy(free_pointcloud).to(d).float()

                    # Setup semantics of pointcloud.
                    sem_sdf = np.zeros((len(pointcloud),))
                    sem_free = [chsel.SemanticsClass.FREE] * len(free_pointcloud)

                    # Combine pointclouds to send to CHSEL.
                    points = torch.cat([pointcloud, free_pointcloud], dim=0)
                    semantics = sem_sdf.tolist() + sem_free

                    assert len(points) == len(semantics)

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
                    # random_init_tsf = pk.Transform3d(pos=torch.randn((B, 3), device=d),
                    #                                  rot=pk.random_rotations(B, device=d), device=d)
                    # random_init_tsf = random_init_tsf.get_matrix()

                    # Run CHSEL to register mesh to the partial view.
                    registration = chsel.CHSEL(sdf, points, semantics, qd_iterations=100, free_voxels_resolution=0.02)
                    res, all_solutions = registration.register(iterations=15, batch=B, initial_tsf=random_init_tsf)

                    if vis:
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
                            Points(pointcloud.cpu().numpy(), c="green", alpha=0.2),
                            Points(free_pointcloud.cpu().numpy(), c="blue", alpha=0.2),
                            Mesh([source_mesh_angle.vertices, source_mesh_angle.faces], c="red"),
                            *mesh_estimates
                        )

                    # Save the results.
                    mmint_utils.save_gzip_pickle(
                        {
                            "res": res,
                            "pointcloud": pointcloud.cpu().numpy(),  # Also save downsampled pointclouds.
                            "free_pointcloud": free_pointcloud.cpu().numpy(),
                            "angle": angle,
                        },
                        os.path.join(plausibles_partial_angle_dir, "%s.pkl.gzip" % mesh_name)
                    )

                    pbar.update()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate plausible poses for partial views.")
    parser.add_argument("dataset_cfg_fn", type=str, help="Dataset config file.")
    parser.add_argument("split", type=str, help="Split to generate plausible poses for.")
    parser.add_argument("-v", "--vis", action="store_true", help="Visualize results.")
    args = parser.parse_args()

    dataset_cfg_ = mmint_utils.load_cfg(args.dataset_cfg_fn)["data"][args.split]
    generate_plausible(dataset_cfg_, args.split, args.vis)
