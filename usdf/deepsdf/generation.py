import numpy as np
import torch
from torch import nn

from usdf.generation import BaseGenerator
from usdf.utils.infer_utils import inference_by_optimization
from usdf.utils.marching_cubes import create_mesh
import usdf.loss as usdf_losses


def get_surface_loss_fn(embed_weight: float):
    def surface_loss_fn(model, latent, data_dict, device):
        # Pull out relevant data.
        surface_coords_ = torch.from_numpy(data_dict["partial_pointcloud"]).to(device).float().unsqueeze(0)
        surface_coords_ = surface_coords_.repeat(latent.shape[0], 1, 1)

        if model.use_angle:
            latent = model.encode_example(torch.tensor([data_dict["example_idx"]]).to(device), latent)
        elif model.sinusoidal_embed:
            assert latent.shape[-1] == 1
            latent = torch.cat([torch.sin(latent), torch.cos(latent)], dim=-1)

        # Predict with updated latents.
        pred_dict_ = model.forward(surface_coords_, latent)

        # loss = 0.0

        # Loss: all points on surface should have SDF = 0.0.
        sdf_loss = torch.mean(torch.abs(pred_dict_["sdf"]), dim=-1)

        # Latent embedding loss: shouldn't drift too far from data.
        embedding_loss = usdf_losses.l2_loss(pred_dict_["embedding"], squared=True, reduce=False)

        loss = sdf_loss + (embed_weight * embedding_loss)
        return loss.mean(), loss

    return surface_loss_fn


def get_init_function(init_mode: str = "random"):
    if init_mode == "random":
        def init_function(embedding: nn.Embedding, device=None):
            torch.nn.init.normal_(embedding.weight, mean=0.0, std=0.1)

        return init_function
    elif init_mode == "1d_interpolation":
        def init_function(embedding: nn.Embedding, device=None):
            embedding.weight.data = torch.arange(0.0, 2 * np.pi, 2 * np.pi / embedding.weight.shape[0],
                                                 device=device, requires_grad=True).unsqueeze(-1)

        return init_function


class Generator(BaseGenerator):

    def __init__(self, cfg: dict, model: nn.Module, generation_cfg: dict, device: torch.device = None):
        super().__init__(cfg, model, generation_cfg, device)
        self.generates_mesh = True

        self.gen_from_known_latent = generation_cfg.get("gen_from_known_latent", False)
        self.mesh_resolution = generation_cfg.get("mesh_resolution", 64)
        self.num_latent = generation_cfg.get("num_latent", 1)
        self.embed_weight = generation_cfg.get("embed_weight", 0.0)
        self.init_mode = generation_cfg.get("init_mode", "random")

    def generate_latent(self, data):
        latent_metadata = {}
        if self.gen_from_known_latent:
            latent = self.model.encode_example(torch.tensor([data["example_idx"]]).to(self.device),
                                               torch.tensor([data["angle"]]).to(self.device).float())
        else:
            z_object_, latent_metadata = inference_by_optimization(self.model,
                                                                   get_surface_loss_fn(self.embed_weight),
                                                                   get_init_function(self.init_mode),
                                                                   self.model.z_object_size,
                                                                   self.num_latent,
                                                                   data,
                                                                   device=self.device,
                                                                   verbose=True)

            latent = z_object_.weight[torch.argmin(latent_metadata["final_loss"])].unsqueeze(0)

            if self.model.use_angle:
                latent = self.model.encode_example(torch.tensor([data["example_idx"]]).to(self.device),
                                                   latent.squeeze(-1))
            elif self.model.sinusoidal_embed:
                assert latent.shape[-1] == 1
                latent = torch.cat([torch.sin(latent), torch.cos(latent)], dim=-1)

        return latent, latent_metadata

    def generate_mesh(self, data, metadata):
        # Check if we have been provided with the latent already.
        if "latent" in metadata:
            latent = metadata["latent"]
        else:
            latent = self.generate_latent(data)

        # Setup function to map from query points to SDF values.
        def sdf_fn(query_points):
            return torch.where(torch.norm(query_points, dim=-1) <= 1.1,
                               self.model.forward(query_points.unsqueeze(0), latent)["sdf"][0],
                               torch.tensor(1.1).to(self.device).float())

        mesh = create_mesh(sdf_fn, n=self.mesh_resolution)
        return mesh, {"latent": latent}

    def generate_pointcloud(self, data, metadata):
        pass
