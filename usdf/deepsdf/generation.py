import numpy as np
import torch
from torch import nn
from vedo import Plotter, Mesh, Points

import pytorch_kinematics as pk

from usdf.generation import BaseGenerator
from usdf.utils.infer_utils import inference_by_optimization
from usdf.utils.marching_cubes import create_mesh
import usdf.loss as usdf_losses


def get_surface_loss_fn(embed_weight: float, alpha: float, infer_pose: bool):
    def surface_loss_fn(model, latent, data_dict, device):
        # Surface point cloud.
        surface_pointcloud = torch.from_numpy(data_dict["surface_pointcloud"]).to(device).float().unsqueeze(0)
        surface_pointcloud = surface_pointcloud.repeat(latent.shape[0], latent.shape[1], 1, 1)

        # Free point cloud.
        free_pointcloud = torch.from_numpy(data_dict["free_pointcloud"]).to(device).float().unsqueeze(0)
        free_pointcloud = free_pointcloud.repeat(latent.shape[0], latent.shape[1], 1, 1)

        if infer_pose:
            # Pull out pose.
            pose = latent[..., :9]
            pos = pose[..., :3]
            rot_6d = pose[..., 3:]
            rot = pk.rotation_6d_to_matrix(rot_6d)

            # Pull out latent.
            latent = latent[..., 9:]

            # Transform points.
            surface_pointcloud = (rot @ surface_pointcloud.transpose(3, 2)).transpose(2, 3) + pos
            free_pointcloud = (rot @ free_pointcloud.transpose(3, 2)).transpose(2, 3) + pos

        # Predict with updated latents.
        surface_pred_dict = model.forward(surface_pointcloud, latent)
        free_pred_dict = model.forward(free_pointcloud, latent)

        # Loss: all points on surface should have SDF = 0.0.
        surface_loss = torch.sum(torch.abs(surface_pred_dict["sdf"]), dim=-1)

        # Loss: all points in free space should have SDF > alpha.
        free_loss = torch.sum(torch.max(torch.zeros_like(free_pred_dict["sdf"]), alpha - free_pred_dict["sdf"]),
                              dim=-1)

        # Latent embedding loss: shouldn't drift too far from data.
        embedding_loss = usdf_losses.l2_loss(latent, squared=True, reduce=False)

        loss = surface_loss + free_loss + embed_weight * embedding_loss
        return loss.mean(), loss

    return surface_loss_fn


def get_init_function(data, init_mode: str = "random", infer_pose: bool = True):
    # Use partial pointcloud mean to initialize pose.
    if infer_pose:
        pos = torch.zeros(3)
        # pos = -data["surface_pointcloud"].mean(axis=0)

    if init_mode == "random":
        def init_function(num_examples: int, num_latent: int, latent_size: int, device=None):
            embed_weight = torch.zeros([num_examples, num_latent, latent_size], dtype=torch.float32, device=device)

            if infer_pose:
                # rot_batch = pk.matrix_to_rotation_6d(
                #     pk.random_rotations(num_examples * num_latent, requires_grad=True)).to(device).reshape(
                #     [num_examples, num_latent, 6])
                rot_batch = pk.matrix_to_rotation_6d(
                    torch.tile(torch.eye(3), (num_examples * num_latent, 1, 1))
                ).to(device).reshape([num_examples, num_latent, 6])
                pos_batch = torch.from_numpy(np.tile(pos, (num_examples, num_latent, 1))).to(device).float()
                pose = torch.cat([pos_batch, rot_batch], dim=-1).requires_grad_(True)
                embed_weight[..., :9] = pose
                torch.nn.init.normal_(embed_weight[..., 9:], mean=0.0, std=0.1)
            else:
                torch.nn.init.normal_(embed_weight, mean=0.0, std=0.1)

            return embed_weight

        return init_function


def get_vis_function(generator, data):
    def vis_function(latent):
        mesh, _ = generator.generate_mesh_from_latent(latent[0])

        plt = Plotter(shape=(1, 1))
        plt.at(0).show(
            Mesh([mesh.vertices, mesh.faces]),
            Points(data["surface_pointcloud"], c="b"),
            Points(data["free_pointcloud"], c="r", alpha=0.05),
        )
        plt.close()

    return vis_function


class Generator(BaseGenerator):

    def __init__(self, cfg: dict, model: nn.Module, generation_cfg: dict, device: torch.device = None):
        super().__init__(cfg, model, generation_cfg, device)

        self.generates_mesh = True
        self.generates_mesh_set = False

        self.gen_from_known_latent = generation_cfg.get("gen_from_known_latent", False)
        self.infer_pose = generation_cfg.get("infer_pose", True)
        self.mesh_resolution = generation_cfg.get("mesh_resolution", 64)
        self.embed_weight = generation_cfg.get("embed_weight", 10.0)
        self.num_latent = generation_cfg.get("num_latent", 1)
        self.init_mode = generation_cfg.get("init_mode", "random")
        self.iter_limit = generation_cfg.get("iter_limit", 1500)
        self.alpha = generation_cfg.get("alpha", -0.01)

    def generate_latent(self, data, return_single: bool = False):
        latent_metadata = {}
        if self.gen_from_known_latent:
            latent = self.model.encode_example(torch.tensor([data["example_idx"]]).to(self.device),
                                               torch.tensor([data["mesh_idx"]]).to(self.device), None)
        else:
            latent, latent_metadata = inference_by_optimization(
                self.model,
                get_surface_loss_fn(self.embed_weight, self.alpha,
                                    self.infer_pose),
                get_init_function(data, self.init_mode,
                                  self.infer_pose),
                self.model.z_object_size + 9 if self.infer_pose else self.model.z_object_size,
                1,
                self.num_latent,
                data,
                # vis_fn=get_vis_function(self, data),
                inf_params={"iter_limit": self.iter_limit},
                device=self.device,
                verbose=True
            )

            if return_single:
                latent = latent[0, torch.argmin(latent_metadata["final_loss"][0])].unsqueeze(0)

        return latent, latent_metadata

    def generate_mesh(self, data, metadata):
        # Generate a single latent code for the given data.
        latent, _ = self.generate_latent(data, True)

        # Generate mesh from latent code.
        return self.generate_mesh_from_latent(latent)

    def generate_mesh_from_latent(self, latent):
        # Setup function to map from query points to SDF values.
        def sdf_fn(query_points):
            query_points = query_points.unsqueeze(0)

            if self.infer_pose:
                # Pull out pose.
                pose = latent[..., :9]
                pos = pose[..., :3]
                rot_6d = pose[..., 3:]
                rot = pk.rotation_6d_to_matrix(rot_6d)

                # Pull out latent.
                latent_ = latent[..., 9:]

                # Transform points.
                query_points = (rot @ query_points.transpose(1, 2)).transpose(2, 1) + pos
            else:
                latent_ = latent

            return self.model.forward(query_points, latent_)["sdf"][0]

        mesh = create_mesh(sdf_fn, n=self.mesh_resolution)
        return mesh, {"latent": latent}

    def generate_mesh_set(self, data, metadata):
        pass
