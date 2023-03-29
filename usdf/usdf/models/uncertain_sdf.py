import torch
import torch.nn as nn
from usdf.models import meta_modules, mlp
from usdf.loss import hypo_weight_loss


class USDF(nn.Module):

    def __init__(self, num_examples: int, z_object_size: int, device=None):
        super().__init__()
        self.z_object_size = z_object_size
        self.device = device

        # Setup the object module.
        # self.object_model = meta_modules.virdo_hypernet(in_features=3, out_features=2,
        #                                                 hyper_in_features=self.z_object_size, hl=2).to(self.device)
        self.object_model = mlp.MLP(3 + z_object_size, 2, hidden_sizes=[8, 8]).to(self.device)

        # Setup latent embeddings (used during training).
        self.object_code = nn.Embedding(num_examples, self.z_object_size, dtype=torch.float32).requires_grad_(True).to(
            self.device)
        nn.init.normal_(self.object_code.weight, mean=0.0, std=0.1)

    def encode_example(self, example_idx: torch.Tensor):
        return self.object_code(example_idx)

    def forward(self, query_points: torch.Tensor, z_object: torch.Tensor):
        # model_in = {
        #     "coords": query_points,
        #     "embedding": z_object,
        # }
        z_object_ = z_object.unsqueeze(1).repeat(1, query_points.shape[1], 1)
        model_in = torch.cat([query_points, z_object_], dim=-1)
        model_out = self.object_model(model_in)

        sdf_means = model_out[..., 0]
        sdf_var = torch.exp(model_out[..., 1])  # Ensure positive.

        out_dict = {
            "query_points": query_points,
            "sdf_means": sdf_means,
            "sdf_var": sdf_var,
            # "hypo_params": model_out["hypo_params"],
            "embedding": z_object,
        }
        return out_dict

    def regularization_loss(self, out_dict: dict):
        # hypo_params = out_dict["hypo_params"]
        # hypo_loss = hypo_weight_loss(hypo_params)
        # return hypo_loss
        return 0.0
