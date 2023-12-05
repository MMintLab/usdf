import os
from torch.multiprocessing import Pool

import torch
import trimesh
from tqdm import tqdm, trange

import mmint_utils
from usdf.utils import utils


def write_results(out_dir, mesh, mesh_set, metadata, idx):
    if mesh is not None:
        if type(mesh) == trimesh.Trimesh:
            mesh_fn = os.path.join(out_dir, "mesh_%d.obj" % idx)
            mesh.export(mesh_fn)
        elif type(mesh) == dict:
            mesh_fn = os.path.join(out_dir, "mesh_%d.pkl.gzip" % idx)
            mmint_utils.save_gzip_pickle(mesh, mesh_fn)

    if mesh_set is not None:
        for i, mesh in enumerate(mesh_set):
            if type(mesh) == trimesh.Trimesh:
                mesh_fn = os.path.join(out_dir, "mesh_%d_%d.obj" % (idx, i))
                mesh.export(mesh_fn)
            elif type(mesh) == dict:
                mesh_fn = os.path.join(out_dir, "mesh_%d_%d.pkl.gzip" % (idx, i))
                mmint_utils.save_gzip_pickle(mesh, mesh_fn)

    if metadata is not None:
        metadata_fn = os.path.join(out_dir, "metadata_%d.pkl.gzip" % idx)
        mmint_utils.save_gzip_pickle(metadata, metadata_fn)


# Predicted Results

class PredResultsLoader:

    def __init__(self, out_dir):
        self.out_dir = out_dir

    def __call__(self, idx):
        mesh_fn = os.path.join(self.out_dir, "mesh_%d.obj" % idx)
        mesh_dict_fn = os.path.join(self.out_dir, "mesh_%d.pkl.gzip" % idx)
        mesh = None
        if os.path.exists(mesh_fn):
            try:
                mesh = trimesh.load(mesh_fn)
            except:  # Sometimes mesh has no vertices!
                pass
        elif os.path.exists(mesh_dict_fn):
            mesh = mmint_utils.load_gzip_pickle(mesh_dict_fn)

        metadata_fn = os.path.join(self.out_dir, "metadata_%d.pkl.gzip" % idx)
        metadata = None
        if os.path.exists(metadata_fn):
            metadata = mmint_utils.load_gzip_pickle(metadata_fn)

        return mesh, metadata


def load_pred_results_all(out_dir, n, device=None):
    meshes = []
    slices = []
    metadatas = []
    torch.multiprocessing.set_start_method('spawn', force=True)

    with Pool(16) as p:
        results = tqdm(p.imap(PredResultsLoader(out_dir), range(n)), total=n)
        results = list(results)

    for mesh, slice_, metadata in results:
        meshes.append(mesh)
        slices.append(slice_)
        metadatas.append(metadata)

    return meshes, slices, metadatas


def load_pred_results(out_dir, n, device=None):
    for idx in range(n):
        mesh_fn = os.path.join(out_dir, "mesh_%d.obj" % idx)
        mesh_dict_fn = os.path.join(out_dir, "mesh_%d.pkl.gzip" % idx)
        mesh = None
        if os.path.exists(mesh_fn):
            try:
                mesh = trimesh.load(mesh_fn)
            except:  # Sometimes mesh has no vertices!
                pass
        elif os.path.exists(mesh_dict_fn):
            mesh = mmint_utils.load_gzip_pickle(mesh_dict_fn)

        metadata_fn = os.path.join(out_dir, "metadata_%d.pkl.gzip" % idx)
        metadata = None
        if os.path.exists(metadata_fn):
            metadata = mmint_utils.load_gzip_pickle(metadata_fn)

        yield mesh, metadata


# Ground Truth Results

class GTResultsLoader:

    def __init__(self, meshes_dir):
        self.meshes_dir = meshes_dir

    def __call__(self, inputs):
        mesh_name, object_pose = inputs
        mesh_fn = os.path.join(self.meshes_dir, mesh_name + ".obj")
        example_mesh = trimesh.load(mesh_fn)
        example_mesh.apply_transform(object_pose)

        return example_mesh


def load_gt_results_all(dataset, dataset_cfg, n, device=None):
    # TODO: This is particular to dataset - can we offload somehow? Maybe move to dataset itself.
    meshes_dir = dataset_cfg["meshes_dir"]

    inputs = []
    for i in trange(n):
        data_dict = dataset[i]
        inputs.append((dataset.meshes[data_dict["mesh_idx"]], data_dict["mesh_pose"]))

    torch.multiprocessing.set_start_method('spawn', force=True)
    with Pool(16) as p:
        meshes = tqdm(p.imap(GTResultsLoader(meshes_dir), inputs), total=n)
        meshes = list(meshes)

    return meshes


def load_gt_results(dataset, dataset_cfg, n, device=None):
    # TODO: This is particular to dataset - can we offload somehow? Maybe move to dataset itself.
    meshes_dir = dataset_cfg["meshes_dir"]

    for idx in range(n):
        data_dict = dataset[idx]

        mesh_fn = os.path.join(meshes_dir, dataset.meshes[data_dict["mesh_idx"]] + ".obj")
        example_mesh = trimesh.load(mesh_fn)

        object_pose = data_dict["mesh_pose"]
        example_mesh.apply_transform(object_pose)

        yield example_mesh
