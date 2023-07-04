import argparse

import torch

import trimesh
from tqdm import trange
from vedo import Plotter, Mesh, Points
import matplotlib as mpl

from usdf.utils.model_utils import load_dataset_from_config
from usdf.utils.results_utils import load_gt_results, load_pred_results


def vis_results(dataset_cfg: str, gen_dir: str, mode: str = "test", offset: int = 0):
    # Load dataset.
    dataset_cfg, dataset = load_dataset_from_config(dataset_cfg, dataset_mode=mode)
    num_examples = len(dataset)
    dataset_cfg = dataset_cfg["data"][args.mode]

    # Load ground truth information.
    gt_meshes = load_gt_results(dataset, dataset_cfg, num_examples)

    # Load predicted information.
    predictions = load_pred_results(gen_dir, num_examples)

    for idx, (gt_mesh, prediction) in enumerate(zip(gt_meshes, predictions)):
        data_dict = dataset[idx]

        # pc = data_dict["partial_pointcloud"]

        plt = Plotter(shape=(1, 2))
        plt.at(0).show(
            Mesh([gt_mesh.vertices, gt_mesh.faces]),
            # Points(pc, c="b"),
            "Ground Truth"
        )

        pred_mesh, _, _ = prediction
        if type(pred_mesh) == trimesh.Trimesh:
            plt.at(1).show(Mesh([pred_mesh.vertices, pred_mesh.faces]),
                           "Predicted")  # , Points(pc, c="b"))
        elif type(pred_mesh) == dict:
            vertex_uncertainty = pred_mesh["uncertainty"]

            # Convert vertex uncertainty to color using jet colormap.
            jet = mpl.colormaps.get_cmap('jet')
            cNorm = mpl.colors.Normalize(vmin=vertex_uncertainty.min(), vmax=vertex_uncertainty.max())
            scalarMap = mpl.cm.ScalarMappable(norm=cNorm, cmap=jet)
            vertex_colors = (scalarMap.to_rgba(vertex_uncertainty) * 255).astype("uint8")

            mesh = Mesh([pred_mesh["vertices"], pred_mesh["faces"]])
            mesh.pointcolors = vertex_colors

            plt.at(1).show(mesh, "Predicted")
        else:
            raise ValueError("Unknown mesh type.")

        plt.interactive().close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize generated results.")
    parser.add_argument("dataset_cfg", type=str, help="Path to dataset config file.")
    parser.add_argument("gen_dir", type=str, help="Path to directory containing generated results.")
    parser.add_argument("--mode", "-m", type=str, default="test", help="Dataset mode to use.")
    parser.add_argument("--offset", "-o", type=int, default=0, help="Offset to use for visualization.")
    args = parser.parse_args()

    vis_results(args.dataset_cfg, args.gen_dir, args.mode, args.offset)
