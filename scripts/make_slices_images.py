import argparse
import os

import matplotlib as mpl
import cv2

from usdf.utils.model_utils import load_dataset_from_config
from usdf.utils.results_utils import load_pred_results, load_gt_results

import matplotlib.pyplot as plt


def make_slices_images(dataset_cfg: str, gen_dir: str, mode: str = "test"):
    # Load dataset.
    dataset_cfg, dataset = load_dataset_from_config(dataset_cfg, dataset_mode=mode)
    num_examples = len(dataset)
    dataset_cfg = dataset_cfg["data"][mode]

    # Load ground truth information.
    gt_meshes = load_gt_results(dataset, dataset_cfg, num_examples)

    # Load predicted information.
    pred_meshes, pred_slices = load_pred_results(gen_dir, num_examples)

    for idx, pred_slice in enumerate(pred_slices):
        uncertainty_image = pred_slice["uncertainty"]

        plt.imshow(uncertainty_image)
        plt.show()

        jet = mpl.colormaps.get_cmap('jet')
        cNorm = mpl.colors.Normalize(vmin=uncertainty_image.min(), vmax=uncertainty_image.max())
        scalarMap = mpl.cm.ScalarMappable(norm=cNorm, cmap=jet)
        color_image = (scalarMap.to_rgba(uncertainty_image.flatten()) * 255).astype("uint8").reshape(100, 100, -1)

        plt.imshow(color_image)
        plt.show()

        cv2.imwrite(os.path.join(gen_dir, "slice_img_%d.png" % idx), color_image[:, :, :3])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize generated results.")
    parser.add_argument("dataset_cfg", type=str, help="Path to dataset config file.")
    parser.add_argument("gen_dir", type=str, help="Path to directory containing generated results.")
    parser.add_argument("--mode", "-m", type=str, default="test", help="Dataset mode to use.")
    args = parser.parse_args()

    make_slices_images(args.dataset_cfg, args.gen_dir, args.mode)
