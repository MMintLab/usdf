import torch
import torch.nn as nn
from usdf.models import meta_modules, mlp, point_net
from usdf.loss import hypo_weight_loss
from usdf.models.deepsdf import DeepSDFObjectModule


class USDF(nn.Module):

    def __init__(self, num_examples: int, z_object_size: int, use_encoder: bool, device=None):
        super().__init__()
        self.z_object_size = z_object_size
        self.device = device
        self.use_encoder = use_encoder

        # Setup the DeepSDF module.
        # self.object_model = DeepSDFObjectModule(z_object_size=self.z_object_size, out_dim=2,
        #                                         final_activation="none").to(self.device)

        self.object_model = meta_modules.virdo_hypernet(in_features=3, out_features=2,
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

        sdf_means = model_out["model_out"][..., 0]
        sdf_var = torch.exp(model_out["model_out"][..., 1])  # Ensure positive.

        out_dict = {
            "query_points": query_points,
            "sdf_means": sdf_means,
            "sdf_var": sdf_var,
            "hypo_params": model_out["hypo_params"],
            "embedding": z_object,
        }
        return out_dict

    def regularization_loss(self, out_dict: dict):
        hypo_params = out_dict["hypo_params"]
        hypo_loss = hypo_weight_loss(hypo_params)
        return hypo_loss
