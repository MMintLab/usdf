import os

import numpy as np
from sklearn.manifold import TSNE

import mmint_utils
from usdf.utils.args_utils import get_model_dataset_arg_parser, load_model_dataset_from_args


def fit_tsne(model, out_dir: str):
    """
    Fit a t-SNE model to the latent space of the given model.

    Args:
        model: DeepSDF model
        out_dir: out directory
    """
    mmint_utils.make_dir(out_dir)

    # Load embedding(s) for model.
    embeddings = [model.object_code.weight.detach().cpu().numpy()]
    out_files = [os.path.join(out_dir, "object_codes.npy")]
    if model.use_pose_code:
        embeddings.append(model.pose_code.weight.detach().cpu().numpy())
        out_files.append(os.path.join(out_dir, "pose_codes.npy"))

    for embedding, out_file in zip(embeddings, out_files):
        # Fit t-SNE model.
        tsne = TSNE(n_components=2, verbose=1)
        tsne_results = tsne.fit_transform(embedding)

        # Save results.
        np.save(out_file, tsne_results)


if __name__ == '__main__':
    parser = get_model_dataset_arg_parser()
    parser.add_argument("--out", "-o", type=str, help="Optional out directory to write generated results to.")
    args = parser.parse_args()

    model_cfg_, model_, dataset_, device_ = load_model_dataset_from_args(args)

    fit_tsne(model_, args.out)
