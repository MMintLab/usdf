import numpy as np
import trimesh
import pyrender
import transforms3d as tf3d
from matplotlib import pyplot as plt
from usdf import utils

from usdf.render_utils import depth_to_pointcloud
from vedo import Plotter, Points, Mesh

mesh_tri = trimesh.load("out/meshes/shapenet/mugs_proc/1a1c0a8d4bad82169f0594e65f756cf5.obj")
mesh = pyrender.Mesh.from_trimesh(mesh_tri)

# Add mesh to the scene.
scene = pyrender.Scene()
scene.add(mesh)

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

# Render the scene.
r = pyrender.OffscreenRenderer(512, 512)
color, depth = r.render(scene)

# plt.figure()
# plt.subplot(1, 2, 1)
# plt.axis('off')
# plt.imshow(color)
# plt.subplot(1, 2, 2)
# plt.axis('off')
# plt.imshow(depth, cmap=plt.cm.gray_r)
# plt.show()

# Convert depth to pointcloud.
pointcloud = depth_to_pointcloud(depth, yfov)
pointcloud = utils.transform_pointcloud(pointcloud, camera_pose)

# Plot pointcloud.
vedo_plt = Plotter()
vedo_plt.at(0).show(Points(pointcloud[:, :3]), Mesh([mesh_tri.vertices, mesh_tri.faces]))
