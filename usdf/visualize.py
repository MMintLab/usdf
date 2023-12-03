import trimesh
from vedo import Plotter, Mesh, Points


def visualize_mesh(data_dict: dict, mesh: trimesh.Trimesh, gt_mesh: trimesh.Trimesh):
    vis_partial = "surface_pointcloud" in data_dict
    plt = Plotter(shape=(2, 1))
    plt.at(0).show(
        Mesh([gt_mesh.vertices, gt_mesh.faces]),
        Points(data_dict["surface_pointcloud"], c="b") if vis_partial else None,
        Points(data_dict["free_pointcloud"], c="r", alpha=0.05) if vis_partial else None,
    )
    plt.at(1).show(
        Mesh([mesh.vertices, mesh.faces]),
        Points(data_dict["surface_pointcloud"], c="b") if vis_partial else None,
        Points(data_dict["free_pointcloud"], c="r", alpha=0.05) if vis_partial else None,
    )
    plt.interactive().close()
