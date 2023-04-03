import torch
from torch import nn

from usdf.generation import BaseGenerator
from usdf.utils.marching_cubes import create_mesh


class Generator(BaseGenerator):

    def __init__(self, cfg: dict, model: nn.Module, generation_cfg: dict, device: torch.device = None):
        super().__init__(cfg, model, generation_cfg, device)
        self.generates_mesh = True

        self.gen_from_known_latent = generation_cfg.get("gen_from_known_latent", False)
        print(self.gen_from_known_latent)

    def generate_mesh(self, data, metadata):
        # Check if we have been provided with the latent already.
        if "latent" in metadata:
            latent = metadata["latent"]
        # else:
        #     raise NotImplementedError()

        if self.gen_from_known_latent:
            latent = self.model.encode_example(torch.from_numpy(data["example_idx"]).to(self.device))

        # Setup function to map from query points to SDF values.
        def sdf_fn(query_points):
            return self.model.forward(query_points.unsqueeze(0), latent)["sdf"]

        mesh = create_mesh(sdf_fn)
        return mesh, {"latent": latent}

    def generate_pointcloud(self, data, metadata):
        pass
