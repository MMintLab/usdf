import os

import numpy as np
import torch
import yaml
from vedo import Plotter, Video, Mesh
import vedo.pyplot as plt

from usdf import config
from usdf.utils.args_utils import get_model_dataset_arg_parser, load_model_dataset_from_args
from usdf.utils.model_utils import load_generation_cfg


def latent_interpolation(model_cfg, model, model_file, device, out_fn, gen_args: dict, tsne_dir):
    video_length = 5.0
    num_interpolations = 10

    model.eval()

    # Load generate cfg, if present.
    generation_cfg = load_generation_cfg(model_cfg, model_file)
    if gen_args is not None:
        generation_cfg.update(gen_args)

    # Load generator.
    generator = config.get_generator(model_cfg, model, generation_cfg, device)

    # Load TSNE embeddings.
    tsne_object_codes = np.load(os.path.join(tsne_dir, "object_codes.npy"))
    # Load full model embeddings.
    object_codes = model.object_code.weight.detach().cpu().numpy()
    assert len(tsne_object_codes) == len(object_codes)

    # Visualize interpolation.
    vedo_plot = Plotter(shape=(1, 2), sharecam=False)
    vedo_plot.at(0).camera.SetPosition(1.5, 1.5, 1.0)
    vedo_plot.at(0).camera.SetFocalPoint(0.0, 0.0, 0.0)
    vedo_plot.at(0).camera.SetViewUp(0.0, 0.0, 1.0)

    # Create plot of TSNE embeddings.
    tsne_plot = plt.plot(tsne_object_codes[:, 0], tsne_object_codes[:, 1], lw=0, marker=".", mc="blue", grid=False,
                         axes=False)
    vedo_plot.at(1).show(tsne_plot)

    video = Video(out_fn, backend="ffmpeg", fps=10)
    num_frames = int(video_length * video.fps)

    # Starting point for interpolation.
    idx1 = np.random.choice(len(tsne_object_codes), size=1, replace=False)[0]

    interp_line = None
    interp_marker = None
    vedo_mesh = None
    for interp_idx in range(num_interpolations):

        # Select next embedding to interpolate to.
        idx2 = np.random.choice(len(tsne_object_codes), size=1, replace=False)[0]

        # Draw line between embeddings.
        if interp_line is not None:
            vedo_plot.at(1).remove(interp_line)
        interp_line = plt.plot([tsne_object_codes[idx1, 0], tsne_object_codes[idx2, 0]],
                               [tsne_object_codes[idx1, 1], tsne_object_codes[idx2, 1]],
                               like=tsne_plot, lw=1, marker="o", mc="red", grid=False, axes=False)
        vedo_plot.at(1).show(interp_line)

        for frame_idx in range(num_frames + 1):
            # Interpolate in tsne embeddings.
            interp_tsne = tsne_object_codes[idx1] * (1.0 - frame_idx / num_frames) + tsne_object_codes[idx2] * (
                    frame_idx / num_frames)

            # Plot interpolated tsne embedding.
            if interp_marker is not None:
                vedo_plot.at(1).remove(interp_marker)
            interp_marker = plt.plot([interp_tsne[0]], [interp_tsne[1]], lw=0, marker="*", mc="green", grid=False,
                                     like=tsne_plot, axes=False)
            vedo_plot.at(1).show(interp_marker)

            # Generate mesh from interpolated embedding.
            interp_code = object_codes[idx1] * (1.0 - frame_idx / num_frames) + object_codes[idx2] * (
                    frame_idx / num_frames)
            mesh, _ = generator.generate_mesh({}, {"latent": torch.from_numpy(interp_code).to(device).unsqueeze(0)})
            if vedo_mesh is not None:
                vedo_plot.at(0).remove(vedo_mesh)
            vedo_mesh = Mesh([mesh.vertices, mesh.faces])
            vedo_plot.at(0).show(vedo_mesh)

            # Add frame to video.
            video.add_frame()

        # Update starting point.
        idx1 = idx2

    video.close()
    vedo_plot.close()


if __name__ == '__main__':
    parser = get_model_dataset_arg_parser()
    parser.add_argument("tsne_dir", type=str, help="Directory containing TSNE embeddings.")
    parser.add_argument("out_fn", type=str, help="Out filename for video.")
    parser.add_argument("--gen_args", type=yaml.safe_load, default=None, help="Generation args.")
    args = parser.parse_args()

    model_cfg_, model_, dataset_, device_ = load_model_dataset_from_args(args)
    latent_interpolation(model_cfg_, model_, args.model_file, device_, args.out_fn, args.gen_args, args.tsne_dir)
