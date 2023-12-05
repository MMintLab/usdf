import numpy as np
import torch
from torch import nn
from tqdm import trange
from vedo import Plotter, Mesh, Points

import pytorch_kinematics as pk

from usdf.generation import BaseGenerator
from usdf.utils.infer_utils import inference_by_optimization
from usdf.utils.marching_cubes import create_mesh
import usdf.loss as usdf_losses


class Generator(BaseGenerator):

    def __init__(self, cfg: dict, model: nn.Module, generation_cfg: dict, device: torch.device = None):
        super().__init__(cfg, model, generation_cfg, device)

        self.generates_mesh = False
        self.generates_mesh_set = True

        self.gen_from_known_latent = generation_cfg.get("gen_from_known_latent", False)
        self.infer_pose = generation_cfg.get("infer_pose", False)
        self.mesh_resolution = generation_cfg.get("mesh_resolution", 128)
        self.embed_weight = generation_cfg.get("embed_weight", 0.01)
        self.num_latent = generation_cfg.get("num_latent", 8)
        self.use_full_pointcloud = generation_cfg.get("use_full_pointcloud", False)
        self.init_mode = generation_cfg.get("init_mode", "random")
        self.iter_limit = generation_cfg.get("iter_limit", 1000)
        self.alpha = generation_cfg.get("alpha", -0.01)
        self.vis_every = generation_cfg.get("vis_every", 250)

    def generate_latent(self, data, return_single: bool = False):
        latent_metadata = {}
        if self.gen_from_known_latent:
            latent = self.model.encode_example(torch.tensor([data["example_idx"]]).to(self.device),
                                               torch.tensor([data["mesh_idx"]]).to(self.device), None)[0]
            pose = None
        else:
            latent, pose, latent_metadata = self.infer_latent(data, num_examples=1)

            if return_single:
                latent = latent[0, torch.argmin(latent_metadata["final_loss"][0])]
                pose = pose[0, torch.argmin(latent_metadata["final_loss"][0])]
            else:
                latent = latent[0]
                pose = pose[0]

        return (latent, pose), latent_metadata

    ####################################################################################################################
    # Inference Helpers                                                                                                #
    ####################################################################################################################

    def infer_latent(self, data_dict, num_examples=1):
        latent, pose = self.init_latent(num_examples)

        # Set up as parameters for opt.
        latent.requires_grad = True
        pose.requires_grad = True

        # Full point cloud.
        full_pointcloud = torch.from_numpy(data_dict["full_pointcloud"]).to(self.device).float().unsqueeze(0)
        full_pointcloud = full_pointcloud.repeat(latent.shape[0], latent.shape[1], 1, 1)
        full_pointcloud.requires_grad = True

        # Surface point cloud.
        surface_pointcloud = torch.from_numpy(data_dict["surface_pointcloud"]).to(self.device).float().unsqueeze(0)
        surface_pointcloud = surface_pointcloud.repeat(latent.shape[0], latent.shape[1], 1, 1)
        surface_pointcloud.requires_grad = True

        # Free point cloud.
        free_pointcloud = torch.from_numpy(data_dict["free_pointcloud"]).to(self.device).float().unsqueeze(0)
        free_pointcloud = free_pointcloud.repeat(latent.shape[0], latent.shape[1], 1, 1)
        free_pointcloud.requires_grad = True

        if self.infer_pose:
            opt = torch.optim.Adam([latent, pose], lr=3e-2)
        else:
            opt = torch.optim.Adam([latent], lr=3e-2)

        iter_idx = 0
        range_ = trange(self.iter_limit)
        for iter_idx in range_:
            opt.zero_grad()

            if iter_idx % self.vis_every == 0 and False:
                self.vis_function(latent, pose, data_dict)

            loss, loss_ind = self.inference_loss(latent, pose,
                                                 full_pointcloud if self.use_full_pointcloud else surface_pointcloud,
                                                 free_pointcloud)

            loss.backward()
            opt.step()

            range_.set_postfix(loss=loss.item())

        _, final_loss = self.inference_loss(latent, pose,
                                            full_pointcloud if self.use_full_pointcloud else surface_pointcloud,
                                            free_pointcloud)

        return latent, pose, {"final_loss": final_loss, "iters": iter_idx + 1}

    def init_latent(self, num_examples):
        # latent_init = torch.zeros([num_examples, self.num_latent, self.model.z_object_size], dtype=torch.float32,
        #                           device=self.device)
        # torch.nn.init.normal_(latent_init[..., 9:], mean=0.0, std=0.1)
        # latent_init = self.model.object_code.weight[0].unsqueeze(0).repeat(num_examples, self.num_latent, 1)
        latent_init = torch.randn([num_examples, self.num_latent, self.model.z_object_size], dtype=torch.float32,
                                  device=self.device) * 0.

        pos = torch.zeros(3)
        if self.infer_pose:
            # rot_batch = pk.matrix_to_rotation_6d(
            #     torch.tile(torch.eye(3), (num_examples * self.num_latent, 1, 1))
            # ).to(self.device).reshape([num_examples, self.num_latent, 6])
            rot_batch = pk.matrix_to_rotation_6d(
                pk.random_rotations(num_examples * self.num_latent)).to(self.device).reshape(
                [num_examples, self.num_latent, 6])
            pos_batch = torch.from_numpy(np.tile(pos, (num_examples, self.num_latent, 1))).to(self.device).float()
            pose_init = torch.cat([pos_batch, rot_batch], dim=-1)
        else:
            rot_batch = pk.matrix_to_rotation_6d(
                torch.tile(torch.eye(3), (num_examples * self.num_latent, 1, 1))
            ).to(self.device).reshape([num_examples, self.num_latent, 6])
            pos_batch = torch.from_numpy(np.tile(pos, (num_examples, self.num_latent, 1))).to(self.device).float()
            pose_init = torch.cat([pos_batch, rot_batch], dim=-1)

        return latent_init, pose_init

    def inference_loss(self, latent, pose, surface_pointcloud, free_pointcloud):
        # Pull out pose.
        pos = pose[..., :3]
        rot_6d = pose[..., 3:]
        rot = pk.rotation_6d_to_matrix(rot_6d)

        # Transform points.
        surface_pointcloud_tf = ((rot @ surface_pointcloud.transpose(3, 2)).transpose(2, 3) +
                                 pos.unsqueeze(2).repeat(1, 1, surface_pointcloud.shape[2], 1))
        free_pointcloud_tf = ((rot @ free_pointcloud.transpose(3, 2)).transpose(2, 3) +
                              pos.unsqueeze(2).repeat(1, 1, free_pointcloud.shape[2], 1))

        # Predict with updated latents.
        surface_pred_dict = self.model.forward(surface_pointcloud_tf, latent)
        free_pred_dict = self.model.forward(free_pointcloud_tf, latent)

        # Loss: all points on surface should have SDF = 0.0.
        epsilon = 3.5e-4
        surface_loss = torch.mean(
            torch.max(torch.abs(surface_pred_dict["sdf"]) - epsilon, torch.zeros_like(surface_pred_dict["sdf"])),
            dim=-1)
        # surface_loss = torch.mean(
        #     torch.max(torch.zeros_like(surface_pred_dict["sdf"]), self.alpha + surface_pred_dict["sdf"]), dim=-1)

        # Loss: all points in free space should have SDF > alpha.
        free_loss = torch.mean(torch.max(torch.zeros_like(free_pred_dict["sdf"]), self.alpha - free_pred_dict["sdf"]),
                               dim=-1)

        # Latent embedding loss: shouldn't drift too far from data.
        embedding_loss = usdf_losses.l2_loss(latent, squared=True, reduce=False)

        loss = surface_loss + free_loss + self.embed_weight * embedding_loss
        return loss.mean(), loss

    def vis_function(self, latent, pose, data_dict):
        meshes = []
        for mesh_idx in range(self.num_latent):
            mesh, _ = self.generate_mesh_from_latent(latent[0, mesh_idx], pose[0, mesh_idx])
            meshes.append(mesh)

        plot_shape = int(np.ceil(np.sqrt(self.num_latent)))
        plt = Plotter(shape=(plot_shape, plot_shape))
        for mesh_idx in range(self.num_latent):
            mesh = meshes[mesh_idx]
            plot_x = mesh_idx // plot_shape
            plot_y = mesh_idx % plot_shape
            plt.at(plot_x, plot_y).show(
                Mesh([mesh.vertices, mesh.faces]),
                Points(data_dict["full_pointcloud"] if self.use_full_pointcloud else data_dict["surface_pointcloud"],
                       c="b"),
                Points(data_dict["free_pointcloud"], c="r", alpha=0.05),
            )
        plt.interactive().close()

    ####################################################################################################################

    def generate_mesh(self, data, metadata):
        # Generate a single latent code for the given data.
        l, _ = self.generate_latent(data, True)
        latent, pose = l

        # Generate mesh from latent code.
        return self.generate_mesh_from_latent(latent, pose)

    def generate_mesh_from_latent(self, latent, pose):
        latent = latent.unsqueeze(0)
        if pose is not None:
            pose = pose.unsqueeze(0)

        # Setup function to map from query points to SDF values.
        def sdf_fn(query_points):
            query_points = query_points.unsqueeze(0)

            if pose is not None:
                # Pull out pose.
                pos = pose[..., :3]
                rot_6d = pose[..., 3:]
                rot = pk.rotation_6d_to_matrix(rot_6d)

                # Transform points.
                query_points = (rot @ query_points.transpose(1, 2)).transpose(2, 1) + pos

            return self.model.forward(query_points, latent)["sdf"][0]

        mesh = create_mesh(sdf_fn, n=self.mesh_resolution)
        return mesh, {"latent": latent, "pose": pose}

    def generate_mesh_set(self, data, metadata):
        # Generate a single latent code for the given data.
        l, l_metadata = self.generate_latent(data, False)
        latent, pose = l

        # Generate meshes from latent code.
        meshes = []
        for mesh_idx in range(self.num_latent):
            mesh, _ = self.generate_mesh_from_latent(latent[mesh_idx], pose[mesh_idx])
            meshes.append(mesh)

        return meshes, {"latent": latent, "pose": pose, "final_loss": l_metadata["final_loss"]}
