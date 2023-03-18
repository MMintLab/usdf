import argparse
import os

import mmint_utils
import numpy as np
import trimesh


def view_meshes(dataset_cfg: dict, split):
    meshes_dir = dataset_cfg["meshes_dir"]

    # Load split info.
    split_fn = os.path.join(meshes_dir, "splits", split + ".txt")
    mesh_fns = np.loadtxt(split_fn, dtype=str)

    for mesh_fn in mesh_fns:
        print(mesh_fn)

        mesh_path = os.path.join(meshes_dir, mesh_fn)
        mesh = trimesh.load(mesh_path)
        mesh.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Render dataset.")
    parser.add_argument("dataset_cfg", type=str, help="Dataset configuration file.")
    parser.add_argument("split", type=str, help="Split to render.")
    args = parser.parse_args()

    dataset_cfg_ = mmint_utils.load_cfg(args.dataset_cfg)

    view_meshes(dataset_cfg_, args.split)
