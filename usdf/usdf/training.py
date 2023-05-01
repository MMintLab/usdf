import os

import mmint_utils
import numpy as np
import torch
from torch import optim
import torch.utils.data
from torch.utils.tensorboard import SummaryWriter
from usdf.data.uncertainty_dataset import UncertaintyDataset
from usdf.training import BaseTrainer
import usdf.loss as usdf_losses
import torch.nn.functional as F
import shutil


class Trainer(BaseTrainer):

    def pretrain(self, *args, **kwargs):
        # Nothing to do.
        pass

    def train(self, train_dataset: UncertaintyDataset, validation_dataset: UncertaintyDataset):
        # Shorthands:
        out_dir = self.cfg['training']['out_dir']
        lr = self.cfg['training']['learning_rate']
        max_epochs = self.cfg['training']['epochs']
        epochs_per_save = self.cfg['training']['epochs_per_save']
        self.train_loss_weights = self.cfg['training']['loss_weights']  # TODO: Better way to set this?
        batch_size = self.cfg['training']['batch_size']
        grad_clip = self.cfg['training'].get('grad_clip', None)

        # Output + vis directory
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        logger = SummaryWriter(os.path.join(out_dir, 'logs/train'))

        # Dump config to output directory.
        mmint_utils.dump_cfg(os.path.join(out_dir, 'config.yaml'), self.cfg)

        # Get optimizer (TODO: Parameterize?)
        # TODO: If decoder-only, change lr for each component.
        optimizer = optim.Adam(self.model.parameters(), lr=lr)

        # Setup gradient clipping.
        if grad_clip is not None:
            for p in self.model.parameters():
                p.register_hook(lambda grad: torch.clamp(grad, -grad_clip, grad_clip))

        # Load model + optimizer if a partially trained copy of it exists.
        epoch_it, it = self.load_partial_train_model(
            {"model": self.model, "optimizer": optimizer}, out_dir, "model.pt")

        # Setup data loader.
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

        # Training loop
        while True:
            epoch_it += 1

            if epoch_it > max_epochs:
                print("Backing up and stopping training. Reached max epochs.")
                shutil.copyfile(os.path.join(out_dir, 'model.pt'), os.path.join(out_dir, 'model_backup.pt'))

                save_dict = {
                    'model': self.model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch_it': epoch_it,
                    'it': it,
                }
                torch.save(save_dict, os.path.join(out_dir, 'model.pt'))
                break

            loss = None

            for batch in train_loader:
                it += 1

                loss = self.train_step(batch, it, optimizer, logger, self.compute_train_loss)

            print('[Epoch %02d] it=%03d, loss=%.4f'
                  % (epoch_it, it, loss))

            if epoch_it % epochs_per_save == 0:
                # Backup previous checkpoint.
                if os.path.exists(os.path.join(out_dir, 'model.pt')):
                    shutil.copyfile(os.path.join(out_dir, 'model.pt'), os.path.join(out_dir, 'model_backup.pt'))

                save_dict = {
                    'model': self.model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch_it': epoch_it,
                    'it': it,
                }
                torch.save(save_dict, os.path.join(out_dir, 'model.pt'))

    def compute_train_loss(self, data, it):
        partial_pc = data["partial_pointcloud"].to(self.device).float()
        example_idx = data["example_idx"].to(self.device)
        query_points = data["query_points"].to(self.device).float()
        sdf_labels = data["sdf"].to(self.device).float()

        # Run model forward.
        z_object = self.model.encode_example(example_idx, partial_pc)

        if torch.isinf(z_object).any() or torch.isnan(z_object).any():
            print("NaN or inf in latent embedding. Exiting..")
            exit()

        out_dict = self.model(query_points, z_object)

        # Compute loss.
        loss_dict = dict()

        # SDF loss.
        dist = out_dict["dist"]
        sdf_loss = -dist.log_prob(sdf_labels).mean()
        loss_dict["sdf_loss"] = sdf_loss

        # Latent embedding loss: well-formed embedding.
        embedding_loss = usdf_losses.l2_loss(out_dict["embedding"], squared=True)
        loss_dict["embedding_loss"] = embedding_loss

        # Network regularization.
        reg_loss = self.model.regularization_loss(out_dict)
        loss_dict["reg_loss"] = reg_loss

        # Calculate total weighted loss.
        loss = 0.0
        for loss_key in loss_dict.keys():
            loss += float(self.train_loss_weights[loss_key]) * loss_dict[loss_key]
        loss_dict["loss"] = loss

        return loss_dict, out_dict
