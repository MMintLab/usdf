import torch
from torch import nn

from usdf.models import mlp, meta_modules
from usdf.models.deepsdf import DeepSDFObjectModule
import usdf.loss as usdf_loss


class DeepSDF(nn.Module):

    def __init__(self, num_examples: int, z_object_size: int, device=None):
        super().__init__()
        self.z_object_size = z_object_size
        self.device = device

        # Setup the DeepSDF module.
        # Note: Might need to make this more complex.
        # self.object_model = DeepSDFObjectModule(z_object_size=self.z_object_size).to(self.device)
        # self.object_model = mlp.MLP(input_size=self.z_object_size + 3, output_size=1,
        #                             hidden_sizes=[128, 128, 128]).to(self.device)
        self.object_model = meta_modules.virdo_hypernet(in_features=3, out_features=1,
                                                        hyper_in_features=self.z_object_size, hl=3).to(self.device)

        # Setup latent embeddings (used during training).
        self.object_code = nn.Embedding(num_examples, self.z_object_size, dtype=torch.float32).requires_grad_(True).to(
            self.device)
        nn.init.normal_(self.object_code.weight, mean=0.0, std=0.1)

    def encode_example(self, example_idx: torch.Tensor):
        return self.object_code(example_idx)

    def forward(self, query_points: torch.Tensor, z_object: torch.Tensor):
        # z_object_ = z_object.unsqueeze(1).repeat(1, query_points.shape[1], 1)
        # model_in = torch.cat([query_points, z_object_], dim=-1)
        # model_out = self.object_model(model_in)

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
