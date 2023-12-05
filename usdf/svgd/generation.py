import numpy as np
import torch
from torch import nn
from tqdm import trange
from vedo import Plotter, Mesh, Points

import pytorch_kinematics as pk

from usdf.generation import BaseGenerator
from usdf.utils.infer_utils import inference_by_optimization
from usdf.utils.marching_cubes import create_mesh
import usdf.loss as usdf_losses

from usdf.deepsdf.generation import Generator as DeepSDFGenerator
from usdf.svgd.svgd_utils import SVGD, RBF, CostProbWrapper


class Generator(DeepSDFGenerator):



    ####################################################################################################################
    # Inference Helpers                                                                                                #
    ####################################################################################################################

    def infer_latent(self, data_dict, num_examples=1):
        latent, pose = self.init_latent(num_examples)

        # Set up as parameters for opt.
        latent.requires_grad = True
        pose.requires_grad = True

        # Surface point cloud.
        surface_pointcloud = torch.from_numpy(data_dict["surface_pointcloud"]).to(self.device).float().unsqueeze(0)
        surface_pointcloud = surface_pointcloud.repeat(latent.shape[0], latent.shape[1], 1, 1)
        surface_pointcloud.requires_grad = True

        # Free point cloud.
        free_pointcloud = torch.from_numpy(data_dict["free_pointcloud"]).to(self.device).float().unsqueeze(0)
        free_pointcloud = free_pointcloud.repeat(latent.shape[0], latent.shape[1], 1, 1)
        free_pointcloud.requires_grad = True

        X = torch.cat([latent, pose], dim=-1)  # (..., latent_size + pose_size)
        X = torch.flatten(X, end_dim=-2)  # (B, latent_size + pose_size)
        X = X.detach().requires_grad_(True)
        opt = torch.optim.Adam([X], lr=1e-3)
        kernel = RBF()
        cost_prob = CostProbWrapper(self.inference_loss, surface_pointcloud, free_pointcloud,
                                    latent_size=latent.shape[-1])
        svgd = SVGD(cost_prob, kernel, opt)

        iter_idx = 0
        range_ = trange(self.iter_limit)
        for iter_idx in range_:
            if iter_idx % self.vis_every == 0:
                latent, pose = cost_prob._split_X(X)
                self.vis_function(latent, pose, data_dict)

            svgd.step(X)

            range_.set_postfix(loss=cost_prob.loss_value.item())

        latent, pose = cost_prob._split_X(X)
        _, final_loss = self.inference_loss(latent, pose, surface_pointcloud, free_pointcloud)

        return latent, pose, {"final_loss": final_loss, "iters": iter_idx + 1}

