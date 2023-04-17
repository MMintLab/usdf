import os

import torch
import trimesh

import mmint_utils
from usdf.utils import utils


def write_results(out_dir, mesh, pointcloud, slice, idx):
    if mesh is not None:
        if type(mesh) == trimesh.Trimesh:
            mesh_fn = os.path.join(out_dir, "mesh_%d.obj" % idx)
            mesh.export(mesh_fn)
        elif type(mesh) == dict:
            mesh_fn = os.path.join(out_dir, "mesh_%d.pkl.gzip" % idx)
            mmint_utils.save_gzip_pickle(mesh, mesh_fn)

    if pointcloud is not None:
        pc_fn = os.path.join(out_dir, "pointcloud_%d.ply" % idx)
        utils.save_pointcloud(pointcloud, pc_fn)

    if slice is not None:
        slice_fn = os.path.join(out_dir, "slice_%d.pkl.gzip" % idx)
        mmint_utils.save_gzip_pickle(slice, slice_fn)


def load_pred_results(out_dir, n, device=None):
    meshes = []
    slices = []

    for idx in range(n):
        mesh_fn = os.path.join(out_dir, "mesh_%d.obj" % idx)
        mesh_dict_fn = os.path.join(out_dir, "mesh_%d.pkl.gzip" % idx)
        if os.path.exists(mesh_fn):
            meshes.append(trimesh.load(mesh_fn))
        elif os.path.exists(mesh_dict_fn):
            meshes.append(mmint_utils.load_gzip_pickle(mesh_dict_fn))
        else:
            meshes.append(None)

        # pc_fn = os.path.join(out_dir, "pointcloud_%d.ply" % idx)
        # if os.path.exists(pc_fn):
        #     pointclouds.append(torch.from_numpy(utils.load_pointcloud(pc_fn)).to(device))
        # else:
        #     pointclouds.append(None)

        slice_fn = os.path.join(out_dir, "slice_%d.pkl.gzip" % idx)
        if os.path.exists(slice_fn):
            slices.append(mmint_utils.load_gzip_pickle(slice_fn))
        else:
            slices.append(None)

    return meshes, slices


def load_gt_results(dataset, dataset_cfg, n, device=None):
    # TODO: This is particular to dataset - can we offload somehow? Maybe move to dataset itself.
    meshes_dir = dataset_cfg["meshes_dir"]
    meshes = []

    for idx in range(n):
        data_dict = dataset[idx]

        mesh_fn = os.path.join(meshes_dir, dataset.meshes[data_dict["mesh_idx"][0]] + ".obj")
        example_mesh = trimesh.load(mesh_fn)

        object_pose = data_dict["object_pose"]
        example_mesh.apply_transform(object_pose)

        meshes.append(example_mesh)

    return meshes
