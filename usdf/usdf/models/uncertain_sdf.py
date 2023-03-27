import torch
import torch.nn as nn
from usdf.models import meta_modules


class USDF(nn.Module):

    def __init__(self, num_examples: int, z_object_size: int, device=None):
        super().__init__()
        self.z_object_size = z_object_size
        self.device = device

        # Setup the object module.
        self.object_model = meta_modules.virdo_hypernet(in_features=3, out_features=2,
                                                        hyper_in_features=self.z_object_size, hl=2).to(self.device)

        # Setup latent embeddings (used during training).
        self.object_code = nn.Embedding(num_examples, self.z_object_size, dtype=torch.float32).requires_grad_(True).to(
            self.device)
        nn.init.normal_(self.object_code.weight, mean=0.0, std=0.1)

    def forward(self, query_points: torch.Tensor, z_object: torch.Tensor):
        model_in = {
            "coords": query_points,
            "embedding": z_object,
        }
        model_out = self.object_model(model_in)

        out_dict = {
            "query_points": model_out["model_in"],
            "sdf": model_out["model_out"],
            "hypo_params": model_out["hypo_params"],
            "embedding": z_object,
        }
        return out_dict
