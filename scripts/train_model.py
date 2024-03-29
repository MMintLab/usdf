import argparse

import torch
import json

import mmint_utils as utils
import usdf.config as config


def train_model(config_file: str, cuda_id: int = 0, no_cuda: bool = False, verbose: bool = False,
                config_args: dict = None):
    # Read config.
    cfg = utils.load_cfg(config_file)

    # If any customization is passed via command line - add in here.
    if config_args is not None:
        cfg = utils.combine_dict(cfg, config_args)

    is_cuda = (torch.cuda.is_available() and not no_cuda)
    device = torch.device("cuda:%d" % cuda_id if is_cuda else "cpu")

    # Setup datasets.
    print('Loading train dataset:')
    train_dataset = config.get_dataset('train', cfg)
    print('Train dataset size: %d' % len(train_dataset))

    if "validation" in cfg["data"]:
        print('Loading validation dataset:')
        validation_dataset = config.get_dataset('validation', cfg)
        print('Validation dataset size: %d' % len(validation_dataset))
    else:
        validation_dataset = None

    # Create model:
    print('Loading model:')
    model = config.get_model(cfg, train_dataset, device=device)
    print(model)

    # Get trainer.
    trainer = config.get_trainer(cfg, model, device=device)
    trainer.train(train_dataset, validation_dataset)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a model.')
    parser.add_argument('config', type=str, help='Path to config file.')
    parser.add_argument('--cuda_id', type=int, default=0, help="Cuda device id to use.")
    parser.add_argument('--no_cuda', action='store_true', help='Do not use cuda.')
    parser.add_argument('-v', '--verbose', dest='verbose', action='store_true', help='Be verbose.')
    parser.set_defaults(verbose=False)
    parser.add_argument('--config_args', type=json.loads, default=None,
                        help='Config elements to overwrite. Use for easy hyperparameter search.')
    args = parser.parse_args()

    train_model(args.config, args.cuda_id, args.no_cuda, args.verbose, args.config_args)
