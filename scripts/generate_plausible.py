import argparse
import os

import chsel
import mmint_utils
import numpy as np
import pytorch_volumetric as pv
import pytorch_kinematics as pk
import torch
from tqdm import tqdm


def generate_plausible(dataset_cfg: dict, split: str, vis: bool = True):
    N = 1000  # Number of points to use in partial pointcloud.
    B = 32  # Batch size for CHSEL.
    d = "cuda" if torch.cuda.is_available() else "cpu"

    meshes_dir = dataset_cfg["meshes_dir"]
    dataset_dir = dataset_cfg["dataset_dir"]
    partials_dir = os.path.join(dataset_dir, "partials")
    plausibles_dir = os.path.join(dataset_dir, "plausibles")
    mmint_utils.make_dir(plausibles_dir)

    # Load split info.
    split_fn = os.path.join(meshes_dir, "splits", split + ".txt")
    mesh_fns = np.loadtxt(split_fn, dtype=str)

    # Load partial data.
    partials_fns = os.listdir(partials_dir)

    # Apply CHSEL on each mesh in dataset to attempt registration to partial view.
    for mesh_fn in tqdm(mesh_fns):
        mesh_fn_full = os.path.join(meshes_dir, mesh_fn)
        obj = pv.MeshObjectFactory(mesh_fn_full)
        sdf = pv.MeshSDF(obj)

        results = dict()
        for partial_fn in partials_fns:
            partial_views = mmint_utils.load_gzip_pickle(os.path.join(partials_dir, partial_fn))

            results[partial_fn] = dict()

            for angle, partial_pc in partial_views.items():
                # Downsample partial pointcloud.
                np.random.shuffle(partial_pc)
                partial_pc = torch.from_numpy(partial_pc[:N]).to(d).float()

                # Setup semantics of pointcloud.
                sem_sdf = np.zeros((N,))

                # Initialize transforms to rotations around z.
                euler_angles = torch.zeros((B, 3), device=d)
                euler_angles[:, 2] = torch.linspace(0.0, 2 * np.pi, B + 1, device=d)[:-1]
                random_init_tsf = pk.Transform3d(pos=torch.zeros((B, 3), device=d),
                                                 rot=pk.euler_angles_to_matrix(euler_angles, "XYZ"), device=d)
                random_init_tsf = random_init_tsf.get_matrix()

                registration = chsel.CHSEL(sdf, partial_pc, sem_sdf, qd_iterations=100, free_voxels_resolution=0.02)
                res, all_solutions = registration.register(iterations=15, batch=B, initial_tsf=random_init_tsf)

                results[partial_fn][angle] = {
                    "res": res,
                    "all_solutions": all_solutions
                }

        # Save results.
        mmint_utils.save_gzip_pickle(results, os.path.join(plausibles_dir, mesh_fn + ".pkl"))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate plausible poses for partial views.")
    parser.add_argument("dataset_cfg_fn", type=str, help="Dataset config file.")
    parser.add_argument("split", type=str, help="Split to generate plausible poses for.")
    parser.add_argument("--vis", action="store_true", help="Visualize results.")
    args = parser.parse_args()

    dataset_cfg_ = mmint_utils.load_cfg(args.dataset_cfg_fn)
    generate_plausible(dataset_cfg_, args.split, args.vis)
