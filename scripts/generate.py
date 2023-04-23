import os.path

import mmint_utils
import yaml
from usdf import config
from usdf.utils.args_utils import get_model_dataset_arg_parser, load_model_dataset_from_args
from usdf.utils.model_utils import load_generation_cfg
from tqdm import trange

from usdf.utils.results_utils import write_results


def generate(model_cfg, model, model_file, dataset, device, out_dir, gen_args: dict):
    model.eval()

    # Load generate cfg, if present.
    generation_cfg = load_generation_cfg(model_cfg, model_file)
    if gen_args is not None:
        generation_cfg.update(gen_args)

    # Load generator.
    generator = config.get_generator(model_cfg, model, generation_cfg, device)

    # Determine what to generate.
    generate_mesh = generator.generates_mesh
    generate_pointcloud = generator.generates_pointcloud
    generate_slice = generator.generates_slice

    # Create output directory.
    if out_dir is not None:
        mmint_utils.make_dir(out_dir)

    # Dump any generation arguments to out directory.
    mmint_utils.dump_cfg(os.path.join(out_dir, "metadata.yaml"), generation_cfg)

    # Go through dataset and generate!
    for idx in trange(len(dataset)):
        data_dict = dataset[idx]
        metadata = {}
        mesh = pointcloud = slice_ = None

        if generate_mesh:
            mesh, metadata_mesh = generator.generate_mesh(data_dict, metadata)
            metadata = mmint_utils.combine_dict(metadata, metadata_mesh)

        if generate_pointcloud:
            pointcloud, metadata_pc = generator.generate_pointcloud(data_dict, metadata)
            metadata = mmint_utils.combine_dict(metadata, metadata_pc)

        if generate_slice:
            slice_, metadata_slice = generator.generate_slice(data_dict, metadata)
            metadata = mmint_utils.combine_dict(metadata, metadata_slice)

        write_results(out_dir, mesh, pointcloud, slice_, metadata, idx)


if __name__ == '__main__':
    parser = get_model_dataset_arg_parser()
    parser.add_argument("--out", "-o", type=str, help="Optional out directory to write generated results to.")
    # TODO: Add visualization?
    parser.add_argument("--gen_args", type=yaml.safe_load, default=None, help="Generation args.")
    args = parser.parse_args()

    # Seed for repeatability.
    # torch.manual_seed(10)
    # np.random.seed(10)
    # random.seed(10)

    model_cfg_, model_, dataset_, device_ = load_model_dataset_from_args(args)

    out = args.out
    if out is None:
        out = os.path.join(model_cfg_["training"]["out_dir"], "out", args.mode)
        mmint_utils.make_dir(out)

    generate(model_cfg_, model_, args.model_file, dataset_, device_, out, args.gen_args)
