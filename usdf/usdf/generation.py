import torch
from torch import nn

from usdf.deepsdf.generation import get_surface_loss_fn
from usdf.generation import BaseGenerator
from usdf.utils.infer_utils import inference_by_optimization
from usdf.utils.marching_cubes import create_mesh


class Generator(BaseGenerator):

    def __init__(self, cfg: dict, model: nn.Module, generation_cfg: dict, device: torch.device = None):
        super().__init__(cfg, model, generation_cfg, device)

        # Does model have encoder?
        self.has_encoder = self.model.use_encoder
        self.gen_from_known_latent = generation_cfg.get("gen_from_known_latent", False)

        self.generates_mesh = True

    def generate_latent(self, data):
        if self.has_encoder:
            # Encoder.
            partial_pc = torch.from_numpy(data["partial_pointcloud"]).to(self.device).float().unsqueeze(0)
            latent = self.model.encode_example(None, partial_pc)
        else:
            # Decoder-only.
            example_idx = torch.from_numpy(data["example_idx"]).to(self.device)
            if self.gen_from_known_latent:
                latent = self.model.encode_example(example_idx, None)
            else:
                z_object_, _ = inference_by_optimization(self.model, get_surface_loss_fn(100.0),
                                                         self.model.z_object_size,
                                                         1, data, device=self.device, verbose=True)
                latent = z_object_.weight
        return latent

    def generate_mesh(self, data, metadata):
        # Check if we have been provided with the latent already.
        if "latent" in metadata:
            latent = metadata["latent"]
        else:
            latent = self.generate_latent(data)

        # Setup function to map from query points to SDF values.
        def sdf_fn(query_points):
            return self.model.forward(query_points.unsqueeze(0), latent)["sdf_means"]

        mesh = create_mesh(sdf_fn)

        # At each mesh point, calculate uncertainty.
        mesh_points = torch.from_numpy(mesh.vertices).to(self.device).float().unsqueeze(0)
        pred_dict = self.model.forward(mesh_points, latent)
        uncertainty = pred_dict["sdf_var"].detach().cpu().numpy()[0]

        return {"vertices": mesh.vertices, "faces": mesh.faces, "uncertainty": uncertainty}, \
            {"latent": latent}

    def generate_pointcloud(self, data, metadata):
        pass
