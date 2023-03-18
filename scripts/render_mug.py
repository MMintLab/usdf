import numpy as np
import trimesh
import pyrender
import transforms3d as tf3d
from matplotlib import pyplot as plt
from usdf import utils

from usdf.render_utils import depth_to_pointcloud
from vedo import Plotter, Points, Mesh

scene = pyrender.Scene()

# Add mesh to the scene.
mesh_tri: trimesh.Trimesh = trimesh.load("out/meshes/shapenet/mugs_proc/1a1c0a8d4bad82169f0594e65f756cf5.obj")
mesh = pyrender.Mesh.from_trimesh(mesh_tri)
mesh_node = pyrender.Node(mesh=mesh, matrix=np.eye(4))
scene.add_node(mesh_node)

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

for angle in np.linspace(0, 2 * np.pi, 10):
    object_pose = np.eye(4)
    object_pose[:3, :3] = tf3d.euler.euler2mat(0, 0, angle, axes="sxyz")
    scene.set_pose(mesh_node, object_pose)

    # Render the scene.
    r = pyrender.OffscreenRenderer(512, 512)
    color, depth = r.render(scene)

    # Convert depth to pointcloud.
    pointcloud = depth_to_pointcloud(depth, yfov)
    pointcloud = utils.transform_pointcloud(pointcloud, camera_pose)

    # Plot pointcloud.
    vedo_plt = Plotter()
    mesh_tri_rot = mesh_tri.copy().apply_transform(object_pose)
    vedo_plt.at(0).show(Points(pointcloud[:, :3]), Mesh([mesh_tri_rot.vertices, mesh_tri_rot.faces]))
