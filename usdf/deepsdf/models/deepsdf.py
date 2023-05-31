import torch
from torch import nn

from usdf.models import meta_modules
import usdf.loss as usdf_loss


class DeepSDF(nn.Module):

    def __init__(self, num_examples: int, z_object_size: int, use_angle: bool, device=None):
        super().__init__()
        self.z_object_size = z_object_size
        self.device = device
        self.use_angle = use_angle

        # Setup the DeepSDF module.
        self.object_model = meta_modules.virdo_hypernet(
            in_features=3, out_features=1,
            hyper_in_features=2 if self.use_angle else self.z_object_size,
            hl=3
        ).to(self.device)

        # Setup latent embeddings (used during training).
        if not self.use_angle:
            self.object_code = nn.Embedding(num_examples, self.z_object_size, dtype=torch.float32).requires_grad_(
                True).to(
                self.device)
            nn.init.normal_(self.object_code.weight, mean=0.0, std=0.1)

    def encode_example(self, example_idx: torch.Tensor, angle: torch.Tensor):
        if self.use_angle:
            embed = angle
            embed = embed.unsqueeze(-1)

            # Sinusoidal embedding.
            embed = torch.cat([torch.sin(embed), torch.cos(embed)], dim=-1)
        else:
            embed = self.object_code(example_idx)

        return embed

    def forward(self, query_points: torch.Tensor, z_object: torch.Tensor):
        model_in = {
            "coords": query_points,
            "embedding": z_object,
        }

        model_out = self.object_model(model_in)
        sdf = model_out["model_out"].squeeze(-1)
        hypo_params = model_out["hypo_params"]

        out_dict = {
            "query_points": query_points,
            "sdf": sdf,
            "embedding": z_object,
            "hypo_params": hypo_params,
        }
        return out_dict

    def regularization_loss(self, out_dict: dict):
        sdf_hypo_params = out_dict["hypo_params"]
        sdf_hypo_loss = usdf_loss.hypo_weight_loss(sdf_hypo_params)
        return sdf_hypo_loss
