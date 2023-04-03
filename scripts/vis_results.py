import argparse

from vedo import Plotter, Mesh

from usdf.utils.model_utils import load_dataset_from_config
from usdf.utils.results_utils import load_gt_results, load_pred_results


def vis_results(dataset_cfg: str, gen_dir: str, mode: str = "test"):
    # Load dataset.
    dataset_cfg, dataset = load_dataset_from_config(dataset_cfg, dataset_mode=mode)
    num_examples = len(dataset)

    # Load ground truth information.
    gt_meshes = load_gt_results(dataset, dataset_cfg, num_examples)

    # Load predicted information.
    pred_meshes = load_pred_results(gen_dir, num_examples)

    for idx in range(len(dataset)):
        data_dict = dataset[idx]

        plt = Plotter(shape=(1, 2))
        plt.at(0).show(Mesh([gt_meshes[idx].vertices, gt_meshes[idx].faces]), "Ground Truth")
        plt.at(1).show(Mesh([pred_meshes[idx].vertices, pred_meshes[idx].faces]), "Predicted")
        plt.interactive().close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize generated results.")
    parser.add_argument("dataset_cfg", type=str, help="Path to dataset config file.")
    parser.add_argument("gen_dir", type=str, help="Path to directory containing generated results.")
    parser.add_argument("--mode", type=str, default="test", help="Dataset mode to use.")
    args = parser.parse_args()

    vis_results(args.dataset_cfg, args.gen_dir, args.mode)
