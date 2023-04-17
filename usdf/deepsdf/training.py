import os

import numpy as np
import torch.utils.data
from tensorboardX import SummaryWriter
from torch import optim
import usdf.loss as usdf_losses

import mmint_utils
from usdf.training import BaseTrainer


class Trainer(BaseTrainer):

    def pretrain(self, *args, **kwargs):
        # Nothing to do.
        pass

    def train(self, train_dataset: torch.utils.data.Dataset, validation_dataset: torch.utils.data.Dataset):
        # Shorthands:
        out_dir = self.cfg['training']['out_dir']
        lr = self.cfg['training']['learning_rate']
        max_epochs = self.cfg['training']['epochs']
        epochs_per_save = self.cfg['training']['epochs_per_save']
        self.train_loss_weights = self.cfg['training']['loss_weights']  # TODO: Better way to set this?

        # Output + vis directory
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        logger = SummaryWriter(os.path.join(out_dir, 'logs/train'))

        # Dump config to output directory.
        mmint_utils.dump_cfg(os.path.join(out_dir, 'config.yaml'), self.cfg)

        # Get optimizer (TODO: Parameterize?)
        # TODO: Different learn rates for model vs. embedding?
        optimizer = optim.Adam(self.model.parameters(), lr=lr)

        # Load model + optimizer if a partially trained copy of it exists.
        epoch_it, it = self.load_partial_train_model(
            {"model": self.model, "optimizer": optimizer}, out_dir, "model.pt")

        # Training loop
        while True:
            epoch_it += 1

            if epoch_it > max_epochs:
                print("Backing up and stopping training. Reached max epochs.")
                save_dict = {
                    'model': self.model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch_it': epoch_it,
                    'it': it,
                }
                torch.save(save_dict, os.path.join(out_dir, 'model.pt'))
                break

            loss = None

            example_idcs = np.arange(len(train_dataset))
            np.random.shuffle(example_idcs)
            example_idcs = torch.from_numpy(example_idcs).to(self.device)

            for example_idx in example_idcs:
                it += 1

                # For this training, we use just a single example per run.
                batch = train_dataset[example_idx]
                loss = self.train_step(batch, it, optimizer, logger, self.compute_train_loss)

            print('[Epoch %02d] it=%03d, loss=%.4f'
                  % (epoch_it, it, loss))

            if epoch_it % epochs_per_save == 0:
                save_dict = {
                    'model': self.model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch_it': epoch_it,
                    'it': it,
                }
                torch.save(save_dict, os.path.join(out_dir, 'model.pt'))

    def compute_train_loss(self, data, it):
        example_idx = torch.from_numpy(data["example_idx"]).to(self.device)
        query_points = torch.from_numpy(data["query_points"]).to(self.device).float().unsqueeze(0)
        sdf_labels = torch.from_numpy(data["sdf"]).to(self.device).float().unsqueeze(0)

        # Run model forward.
        z_object = self.model.encode_example(example_idx)
        out_dict = self.model(query_points, z_object)

        # Compute loss.
        loss_dict = dict()

        # SDF loss.
        pred_sdf = out_dict["sdf"]
        sdf_loss = usdf_losses.sdf_loss(sdf_labels, pred_sdf, clip=0.1)
        loss_dict["sdf_loss"] = sdf_loss

        # Latent embedding loss: well-formed embedding.
        embedding_loss = usdf_losses.l2_loss(out_dict["embedding"], squared=True)
        loss_dict["embedding_loss"] = embedding_loss

        # Calculate total weighted loss.
        loss = 0.0
        for loss_key in loss_dict.keys():
            loss += float(self.train_loss_weights[loss_key]) * loss_dict[loss_key]
        loss_dict["loss"] = loss

        return loss_dict, out_dict
