import argparse
import os

import mmint_utils
import numpy as np
import pyrender
import trimesh
import transforms3d as tf3d
from tqdm import tqdm
from usdf import utils
from usdf.render_utils import depth_to_pointcloud

os.environ["PYOPENGL_PLATFORM"] = "egl"


def render_dataset(dataset_cfg: dict, split: str, vis: bool = False):
    N = 8  # Number of angles to render mesh from.

    meshes_dir = dataset_cfg["meshes_dir"]
    dataset_dir = dataset_cfg["dataset_dir"]
    mmint_utils.make_dir(dataset_dir)

    partials_dir = os.path.join(dataset_dir, "partials")
    mmint_utils.make_dir(partials_dir)

    # Load split info.
    split_fn = os.path.join(meshes_dir, "splits", split + ".txt")
    mesh_fns = np.loadtxt(split_fn, dtype=str)

    # Build scene.
    scene = pyrender.Scene()

    # Add camera to the scene.
    yfov = np.pi / 3.0
    camera = pyrender.PerspectiveCamera(yfov=yfov, aspectRatio=1.0)
    camera_pose = np.eye(4)
    camera_pose[:3, :3] = tf3d.euler.euler2mat(np.pi / 2.0, 0, 0, axes="sxyz")
    camera_pose[:3, 3] = [0, -2.0, 0]
    scene.add(camera, pose=camera_pose)

    # Add light to the scene.
    light = pyrender.SpotLight(color=np.ones(3), intensity=3.0, innerConeAngle=np.pi / 16.0, outerConeAngle=np.pi / 6.0)
    scene.add(light, pose=camera_pose)

    # Add renderer.
    r = pyrender.OffscreenRenderer(128, 128)

    for mesh_fn in tqdm(mesh_fns):
        partials_dict = dict()

        mesh_path = os.path.join(meshes_dir, mesh_fn)
        mesh_tri: trimesh.Trimesh = trimesh.load(mesh_path)
        mesh = pyrender.Mesh.from_trimesh(mesh_tri)
        mesh_node = pyrender.Node(mesh=mesh, matrix=np.eye(4))
        scene.add_node(mesh_node)

        for angle in np.linspace(0, 2 * np.pi, N + 1)[:-1]:
            object_pose = np.eye(4)
            object_pose[:3, :3] = tf3d.euler.euler2mat(0, 0, angle, axes="sxyz")
            scene.set_pose(mesh_node, object_pose)

            # Render the scene.
            color, depth = r.render(scene)

            # Convert depth to pointcloud.
            pointcloud = depth_to_pointcloud(depth, yfov)[:, :3]
            pointcloud = utils.transform_pointcloud(pointcloud, camera_pose)

            # Save partials.
            partials_dict[angle] = pointcloud

        # Write partials.
        partials_fn = os.path.join(partials_dir, mesh_fn[:-4] + ".pkl.gzip")
        mmint_utils.save_gzip_pickle(partials_dict, partials_fn)

        scene.remove_node(mesh_node)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Render dataset.")
    parser.add_argument("dataset_cfg", type=str, help="Dataset configuration file.")
    parser.add_argument("split", type=str, help="Split to render.")
    parser.add_argument("-v", "--vis", action="store_true", help="Visualize pointclouds.")
    parser.set_defaults(vis=False)
    args = parser.parse_args()

    dataset_cfg_ = mmint_utils.load_cfg(args.dataset_cfg)

    render_dataset(dataset_cfg_, args.split, args.vis)
