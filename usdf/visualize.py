import trimesh
from vedo import Plotter, Mesh, Points


def visualize_mesh(data_dict: dict, mesh: trimesh.Trimesh, gt_mesh: trimesh.Trimesh):
    plt = Plotter(shape=(2, 1))
    plt.at(0).show(
        Mesh([gt_mesh.vertices, gt_mesh.faces]),
        Points(data_dict["surface_pointcloud"], c="b"),
        Points(data_dict["free_pointcloud"], c="r", alpha=0.05),
    )
    plt.at(1).show(
        Mesh([mesh.vertices, mesh.faces]),
        Points(data_dict["surface_pointcloud"], c="b"),
        Points(data_dict["free_pointcloud"], c="r", alpha=0.05),
    )
    plt.interactive().close()
