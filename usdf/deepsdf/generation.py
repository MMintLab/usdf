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


class Generator(BaseGenerator):

    def __init__(self, cfg: dict, model: nn.Module, generation_cfg: dict, device: torch.device = None):
        super().__init__(cfg, model, generation_cfg, device)
        self.generates_mesh = True

        self.gen_from_known_latent = generation_cfg.get("gen_from_known_latent", False)
        self.mesh_resolution = generation_cfg.get("mesh_resolution", 64)
        self.num_latent = generation_cfg.get("num_latent", 1)

    def generate_latent(self, data):
        if self.gen_from_known_latent:
            latent = self.model.encode_example(torch.tensor([data["example_idx"]]).to(self.device))
        else:
            z_object_, latent_metadata = inference_by_optimization(self.model, get_surface_loss_fn(0.01),
                                                                   self.model.z_object_size,
                                                                   self.num_latent, data, device=self.device,
                                                                   verbose=False)
            latent = z_object_.weight[torch.argmin(latent_metadata["final_loss"])].unsqueeze(0)
        return latent

    def generate_mesh(self, data, metadata):
        # Check if we have been provided with the latent already.
        if "latent" in metadata:
            latent = metadata["latent"]
        else:
            latent = self.generate_latent(data)

        # Setup function to map from query points to SDF values.
        def sdf_fn(query_points):
            return self.model.forward(query_points.unsqueeze(0), latent)["sdf"]

        mesh = create_mesh(sdf_fn, n=self.mesh_resolution)
        return mesh, {"latent": latent}

    def generate_pointcloud(self, data, metadata):
        pass
