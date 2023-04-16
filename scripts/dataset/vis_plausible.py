import argparse
import os

import chsel
import mmint_utils
import numpy as np
import torch
import trimesh
from usdf.utils import utils
from vedo import Mesh, Plotter, Points
import torch.nn.functional as F
import chsel
import pytorch_volumetric as pv
import pytorch_kinematics as pk
import matplotlib.pyplot as plt_
import open3d as o3d


def vis_plausible(dataset_cfg: dict, split: str):
    # TODO: Move to config.
    N_angles = 8  # Number of angles each mesh is rendered from.
    order_by_rank = True
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

            # Load the partial view information for the given angle.
            pointcloud = utils.load_pointcloud(os.path.join(example_angle_partials_dir, "pointcloud.ply"))[:, :3]
            np.random.shuffle(pointcloud)
            pointcloud = pointcloud[:1000]
            free_pointcloud = utils.load_pointcloud(os.path.join(example_angle_partials_dir, "free_pointcloud.ply"))[:,
                              :3]
            free_pointcloud = free_pointcloud[np.linalg.norm(free_pointcloud, axis=1) < 1.4]
            angle = mmint_utils.load_gzip_pickle(os.path.join(example_angle_partials_dir, "info.pkl.gzip"))["angle"]

            # plt = Plotter()
            # source_mesh_angle = source_mesh.copy().apply_transform(
            #     trimesh.transformations.rotation_matrix(angle, [0, 0, 1]))
            # plt.at(0).show(
            #     Points(pointcloud, c="green", alpha=0.5),
            #     Points(free_pointcloud, c="blue", alpha=0.5),
            #     Mesh([source_mesh_angle.vertices, source_mesh_angle.faces], c="red"),
            # )

            for match_mesh_name in meshes:
                # Load mesh used to match to partial view.
                mesh_fn_full = os.path.join(meshes_dir, match_mesh_name + ".obj")
                mesh_tri: trimesh.Trimesh = trimesh.load(mesh_fn_full)

                # Load the plausible results.
                res_dict = mmint_utils.load_gzip_pickle(
                    os.path.join(example_angle_plausibles_dir, match_mesh_name + ".pkl.gzip"))
                res = res_dict["res"]

                # Load the partial view.
                # pointcloud = res_dict["pointcloud"]
                # free_pointcloud = res_dict["free_pointcloud"]
                # angle = res_dict["angle"]

                # create the CHSEL wrapper
                d = "cuda"
                dtype = torch.float
                obj = pv.MeshObjectFactory(mesh_fn_full)
                obj_sdf = pv.MeshSDF(obj)
                positions = torch.from_numpy(np.concatenate([pointcloud, free_pointcloud])).to(device=d, dtype=dtype)
                sem_sdf = np.zeros((len(pointcloud),))
                sem_free = [chsel.SemanticsClass.FREE] * len(free_pointcloud)

                # Combine pointclouds to send to CHSEL.
                semantics = sem_sdf.tolist() + sem_free

                # extract the known free voxels from the given free points
                free_voxels = pv.VoxelSet(positions[len(pointcloud):], torch.ones(len(free_pointcloud), device=d))
                known_sdf_voxels = pv.VoxelSet(positions[:len(pointcloud)], torch.zeros(len(pointcloud), device=d))

                registration = chsel.CHSEL(obj_sdf, positions, semantics, qd_iterations=100,
                                           free_voxels_resolution=0.02,
                                           # cost=chsel.costs.VolumetricDoubleDirectCost,
                                           # free_voxels=free_voxels,
                                           # known_sdf_voxels=known_sdf_voxels,
                                           # surface_threshold=0.0, scale_known_freespace=20.
                                           )

                print(res.rmse)
                gt_world_to_link = pk.RotateAxisAngle(angle, "Z", device=d, dtype=dtype).get_matrix()
                Rs = torch.cat([res.RTs.R, gt_world_to_link[:, :3, :3]], dim=0)
                Ts = torch.cat([res.RTs.T, gt_world_to_link[:, :3, 3]], dim=0)
                rmse_sanity_check = registration.evaluate(Rs, Ts, None)
                print(rmse_sanity_check)

                # res.RTs.R, res.RTs.T, res.RTs.s are the similarity transform parameters
                world_to_link = chsel.solution_to_world_to_link_matrix(res)
                link_to_world = world_to_link.inverse()

                if order_by_rank:
                    indices = torch.arange(len(res.rmse))
                    # indices = torch.argsort(res.rmse, descending=True)

                    for index in indices:
                        print(f"RMSE: {res.rmse[index].item()} index: {index}")
                        mesh_tri_est = mesh_tri.copy().apply_transform(link_to_world[index].cpu().numpy())

                        interior_threshold = -registration.volumetric_cost.surface_threshold
                        world_frame_free_voxels, known_free = registration.volumetric_cost.free_voxels.get_known_pos_and_values()
                        world_frame_free_voxels = world_frame_free_voxels[known_free.view(-1) == 1]
                        # world_frame_free_voxels = positions[len(pointcloud):]
                        model_frame_free_pos = pk.Transform3d(matrix=world_to_link[index]).transform_points(
                            world_frame_free_voxels)
                        definitely_not_violating = obj_sdf.outside_surface(model_frame_free_pos,
                                                                           surface_level=interior_threshold)
                        violating = ~definitely_not_violating
                        # this full lookup is much, much slower than the cached version with points, but are about equivalent
                        # sdf_value, sdf_grad = obj_sdf(model_frame_free_pos[violating])
                        # # get all the SDF values for plotting
                        # # interior points will have sdf_value < 0
                        # loss = torch.zeros(model_frame_free_pos.shape[:-1], dtype=model_frame_free_pos.dtype,
                        #                    device=model_frame_free_pos.device)
                        # violation = interior_threshold - sdf_value
                        # loss[violating] = violation
                        sdf_value, sdf_grad = obj_sdf(model_frame_free_pos)
                        # print the max and min of the sdf values
                        print(f"min: {sdf_value.min()} max: {sdf_value.max()}")

                        # print(f"sum: {loss.sum()} max: {loss.max()} mean: {loss.mean()} quantile: {loss.quantile(0.9)}")

                        # get quantile of free points that have high loss
                        # quantile = 0.9
                        # loss = loss.cpu().numpy()
                        # loss_quantile = np.quantile(loss, quantile)
                        # free_pointcloud = free_pointcloud[loss > loss_quantile]
                        # loss = loss[loss > loss_quantile]

                        mesh_estimates = Mesh([mesh_tri_est.vertices, mesh_tri_est.faces], alpha=0.5)
                        plt = Plotter()
                        source_mesh_angle = source_mesh.copy().apply_transform(
                            trimesh.transformations.rotation_matrix(angle, [0, 0, 1]))
                        # color each free_pointcloud based on its loss
                        # create matplotlib colormap with norm based on loss
                        cmap = plt_.cm.get_cmap('viridis')
                        # norm = plt_.Normalize(loss.min(), loss.max())
                        # map loss to color
                        # colors = cmap(norm(loss))

                        # create norm based on sdf_value and map to color
                        sdf_value = sdf_value.cpu().numpy()
                        norm = plt_.Normalize(sdf_value.min(), sdf_value.max())
                        colors = cmap(norm(sdf_value))

                        # convert trimesh mesh to open3d mesh
                        source_mesh_angle = o3d.geometry.TriangleMesh()
                        source_mesh_angle.vertices = o3d.utility.Vector3dVector(source_mesh.vertices)
                        source_mesh_angle.triangles = o3d.utility.Vector3iVector(source_mesh.faces)
                        source_mesh_angle.compute_vertex_normals()
                        source_mesh_angle.paint_uniform_color([0.5, 0.5, 0.5])

                        # visualize the mesh and transformed mesh with open3d
                        # create o3d pointcloud from free_pointcloud with colors
                        pcd_colored = o3d.geometry.PointCloud()
                        pcd_colored.points = o3d.utility.Vector3dVector(world_frame_free_voxels.cpu().numpy())
                        pcd_colored.colors = o3d.utility.Vector3dVector(colors[..., :3])
                        # create o3d pointcloud from pointcloud
                        pcd_source = o3d.geometry.PointCloud()
                        pcd_source.points = o3d.utility.Vector3dVector(pointcloud)
                        # paint pcd_source green
                        pcd_source.paint_uniform_color([0, 1, 0])
                        o3d.visualization.draw_geometries([
                            source_mesh_angle,
                            pcd_colored,
                            # pcd_source
                        ])

                        # downsample_stride = 10
                        # plt.at(0).show(
                        #     Points(pointcloud, c="green", alpha=0.1),
                        #     Points(free_pointcloud[::downsample_stride], c=colors[..., :3][::downsample_stride], alpha=0.1),
                        #     # Points(free_pointcloud, c="blue", alpha=0.1),
                        #     Mesh([source_mesh_angle.vertices, source_mesh_angle.faces], c="red", alpha=0.2),
                        #     mesh_estimates
                        # )

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
