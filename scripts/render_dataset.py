import argparse
import os

import mmint_utils
import numpy as np
import pyrender
import trimesh
import transforms3d as tf3d
from tqdm import tqdm
from usdf.utils import utils
from usdf.utils.render_utils import depth_to_pointcloud, depth_to_free_points
from vedo import Plotter, Points, Mesh

os.environ["PYOPENGL_PLATFORM"] = "egl"


def render_dataset(dataset_cfg: dict, split: str, vis: bool = False):
    N = 8  # Number of angles to render mesh from.

    meshes_dir = dataset_cfg["meshes_dir"]
    dataset_dir = dataset_cfg["dataset_dir"]
    mmint_utils.make_dir(dataset_dir)

    partials_dir = os.path.join(dataset_dir, "partials")
    mmint_utils.make_dir(partials_dir)

    # Load split info.
    split_fn = os.path.join(meshes_dir, "splits", dataset_cfg["splits"][split])
    meshes = np.atleast_1d(np.loadtxt(split_fn, dtype=str))

    # Build scene.
    scene = pyrender.Scene()

    # Add camera to the scene.
    yfov = np.pi / 2.0
    camera = pyrender.PerspectiveCamera(yfov=yfov, aspectRatio=1.0)
    camera_pose = np.eye(4)
    camera_pose[:3, :3] = tf3d.euler.euler2mat(np.pi / 2.0, 0, 0, axes="sxyz")
    camera_pose[:3, 3] = [0, -2.0, 0]
    scene.add(camera, pose=camera_pose)

    # Add light to the scene.
    light = pyrender.SpotLight(color=np.ones(3), intensity=3.0, innerConeAngle=np.pi / 16.0, outerConeAngle=np.pi / 6.0)
    scene.add(light, pose=camera_pose)

    # Add renderer.
    r = pyrender.OffscreenRenderer(256, 256)

    with tqdm(total=len(meshes) * N) as pbar:
        for mesh_fn in meshes:
            mesh_path = os.path.join(meshes_dir, mesh_fn)
            mesh_tri: trimesh.Trimesh = trimesh.load(mesh_path)
            mesh = pyrender.Mesh.from_trimesh(mesh_tri)
            mesh_node = pyrender.Node(mesh=mesh, matrix=np.eye(4))
            scene.add_node(mesh_node)

            mesh_partials_path = os.path.join(partials_dir, mesh_fn[:-4])
            mmint_utils.make_dir(mesh_partials_path)

            for angle_idx, angle in enumerate(np.linspace(0, 2 * np.pi, N + 1)[:-1]):
                object_pose = np.eye(4)
                object_pose[:3, :3] = tf3d.euler.euler2mat(0, 0, angle, axes="sxyz")
                scene.set_pose(mesh_node, object_pose)

                # Render the scene.
                color, depth = r.render(scene)

                # Convert depth to pointcloud.
                pointcloud = depth_to_pointcloud(depth, yfov)[:, :3]
                pointcloud = utils.transform_pointcloud(pointcloud, camera_pose)

                # Recover free points.
                free_pointcloud = depth_to_free_points(depth, yfov, max_depth=3.0, n=200)
                free_pointcloud = utils.transform_pointcloud(free_pointcloud, camera_pose)

                if vis:
                    plt = Plotter()
                    mesh_tri_rot = mesh_tri.copy().apply_transform(object_pose)
                    plt.at(0).show(Points(pointcloud[:, :3], c="green"),
                                   Mesh([mesh_tri_rot.vertices, mesh_tri_rot.faces]),
                                   Points(free_pointcloud[:, :3], c="blue", alpha=0.1))

                # Save partials.
                mesh_partials_angle_path = os.path.join(mesh_partials_path, "angle_%d" % angle_idx)
                mmint_utils.make_dir(mesh_partials_angle_path)
                utils.save_pointcloud(pointcloud, os.path.join(mesh_partials_angle_path, "pointcloud.ply"))
                utils.save_pointcloud(free_pointcloud, os.path.join(mesh_partials_angle_path, "free_pointcloud.ply"))
                mmint_utils.save_gzip_pickle({"angle": angle}, os.path.join(mesh_partials_angle_path, "info.pkl.gzip"))

                pbar.update(1)

            scene.remove_node(mesh_node)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Render dataset.")
    parser.add_argument("dataset_cfg", type=str, help="Dataset configuration file.")
    parser.add_argument("split", type=str, help="Split to render.")
    parser.add_argument("-v", "--vis", action="store_true", help="Visualize pointclouds.")
    parser.set_defaults(vis=False)
    args = parser.parse_args()

    dataset_cfg_ = mmint_utils.load_cfg(args.dataset_cfg)["data"][args.split]

    render_dataset(dataset_cfg_, args.split, args.vis)
