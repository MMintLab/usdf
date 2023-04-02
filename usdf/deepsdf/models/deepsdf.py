import torch
from torch import nn

from usdf.models import mlp


class DeepSDF(nn.Module):

    def __init__(self, num_examples: int, z_object_size: int, device=None):
        super().__init__()
        self.z_object_size = z_object_size
        self.device = device

        # Setup the DeepSDF module.
        # Note: Might need to make this more complex.
        self.object_model = mlp.build_mlp(3 + self.z_object_size, 1, hidden_sizes=[8, 8, 8]).to(self.device)

        # Setup latent embeddings (used during training).
        self.object_code = nn.Embedding(num_examples, self.z_object_size, dtype=torch.float32).requires_grad_(True).to(
            self.device)
        nn.init.normal_(self.object_code.weight, mean=0.0, std=0.1)

    def encode_example(self, example_idx: torch.Tensor):
        return self.object_code(example_idx)

    def forward(self, query_points: torch.Tensor, z_object: torch.Tensor):
        z_object_ = z_object.unsqueeze(1).repeat(1, query_points.shape[1], 1)
        model_in = torch.cat([query_points, z_object_], dim=-1)
        model_out = self.object_model(model_in)

        out_dict = {
            "query_points": query_points,
            "sdf": model_out,
            "embedding": z_object,
        }
        return out_dict
