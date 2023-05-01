from enum import Enum

import torch
import torch.nn as nn
from torch.distributions import Normal, Categorical, MixtureSameFamily

from usdf.models import meta_modules, point_net
from usdf.loss import hypo_weight_loss


class Dist(Enum):
    GAUSSIAN = "gaussian"
    GMM = "gmm"


class USDF(nn.Module):

    def __init__(self, num_examples: int, z_object_size: int, use_encoder: bool, num_components: int = 1,
                 distribution: Dist = Dist.GAUSSIAN, device=None):
        super().__init__()
        self.z_object_size = z_object_size
        self.device = device
        self.use_encoder = use_encoder
        self.num_components = num_components  # Only used for GMM.
        self.distribution = distribution

        out_features = -1
        if self.distribution == Dist.GAUSSIAN:
            out_features = 2
        elif self.distribution == Dist.GMM:
            out_features = 3 * self.num_components

        # Setup the DeepSDF module.
        self.object_model = meta_modules.virdo_hypernet(in_features=3, out_features=out_features,
                                                        hyper_in_features=self.z_object_size, hl=4).to(self.device)

        if self.use_encoder:
            self.object_encoder = point_net.PointNet(self.z_object_size, "max").to(self.device)
        else:
            # Setup latent embeddings (used during training).
            self.object_code = nn.Embedding(num_examples, self.z_object_size, dtype=torch.float32).requires_grad_(
                True).to(
                self.device)
            nn.init.normal_(self.object_code.weight, mean=0.0, std=0.1)

    def encode_example(self, example_idx: torch.Tensor, partial_pc: torch.Tensor):
        if self.use_encoder:
            return self.object_encoder(partial_pc)
        else:
            return self.object_code(example_idx)

    def forward(self, query_points: torch.Tensor, z_object: torch.Tensor):
        # model_out = self.object_model(query_points, z_object)

        model_in = {
            "coords": query_points,
            "embedding": z_object,
        }
        model_out = self.object_model(model_in)

        dist_feats = model_out["model_out"]

        # Build distribution from predictions.
        dist = None
        if self.distribution == Dist.GAUSSIAN:
            sdf_means = dist_feats[..., 0]
            sdf_var = torch.exp(dist_feats[..., 1])  # Ensure positive.

            dist = Normal(sdf_means, sdf_var)
        elif self.distribution == Dist.GMM:
            mixture_weights = torch.softmax(dist_feats[..., :self.num_components], dim=-1)
            sdf_means = dist_feats[..., self.num_components:2 * self.num_components]
            sdf_var = torch.exp(dist_feats[..., 2 * self.num_components:3 * self.num_components])  # Ensure positive.

            mix = Categorical(mixture_weights)
            comp = Normal(sdf_means, sdf_var)
            dist = MixtureSameFamily(mix, comp)

        out_dict = {
            "query_points": query_points,
            "dist": dist,
            "hypo_params": model_out["hypo_params"],
            "embedding": z_object,
        }
        return out_dict

    def regularization_loss(self, out_dict: dict):
        hypo_params = out_dict["hypo_params"]
        hypo_loss = hypo_weight_loss(hypo_params)
        return hypo_loss
