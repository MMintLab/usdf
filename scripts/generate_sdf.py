import argparse
import os

import chsel
import mmint_utils
import numpy as np
import trimesh
from tqdm import tqdm
from vedo import Plotter, Mesh, Points

from usdf.utils.sdf_utils import get_sdf_query_points, get_sdf_values


def generate_sdf(dataset_cfg: dict, split: str, vis: bool = False):
    # TODO: Move to config.
    N_angles = 8  # Number of angles each mesh is rendered from.
    B = 32  # Number of plausible poses generated.
    meshes_dir = dataset_cfg["meshes_dir"]
    dataset_dir = dataset_cfg["dataset_dir"]
    partials_dir = os.path.join(dataset_dir, "partials")
    plausibles_dir = os.path.join(dataset_dir, "plausibles")
    sdfs_dir = os.path.join(dataset_dir, "sdfs")
    mmint_utils.make_dir(sdfs_dir)

    # Load split info.
    split_fn = os.path.join(meshes_dir, "splits", dataset_cfg["splits"][split])
    meshes = np.atleast_1d(np.loadtxt(split_fn, dtype=str))
    meshes = [m.replace(".obj", "") for m in meshes]

    with tqdm(total=len(meshes) * len(meshes) * N_angles) as pbar:
        for partial_mesh_name in meshes:
            # Load the source mesh used to generate the partial view.
            # We use this mesh to choose the SDF points to evaluate.
            source_mesh_fn = os.path.join(meshes_dir, partial_mesh_name + ".obj")
            source_mesh = trimesh.load(source_mesh_fn)

            example_plausibles_dir = os.path.join(plausibles_dir, partial_mesh_name)
            example_sdf_dir = os.path.join(sdfs_dir, partial_mesh_name)
            mmint_utils.make_dir(example_sdf_dir)
            for angle_idx in range(N_angles):
                example_angle_plausibles_dir = os.path.join(example_plausibles_dir, "angle_%d" % angle_idx)
                example_angle_sdf_dir = os.path.join(example_sdf_dir, "angle_%d" % angle_idx)
                mmint_utils.make_dir(example_angle_sdf_dir)

                # Sample the SDF points to evaluate.
                query_points = get_sdf_query_points(source_mesh, noise=0.01)

                if vis:
                    plt = Plotter()
                    plt.at(0).show(
                        Mesh([source_mesh.vertices, source_mesh.faces]),
                        Points(query_points),
                    )
                    plt.interactive().close()

                for mesh_idx, match_mesh_name in enumerate(meshes):
                    # Load the plausible results.
                    res_dict = mmint_utils.load_gzip_pickle(
                        os.path.join(example_angle_plausibles_dir, match_mesh_name + ".pkl.gzip"))
                    res = res_dict["res"]

                    # Convert the solution(s) to a transform matrix.
                    world_to_link = chsel.solution_to_world_to_link_matrix(res)
                    link_to_world = world_to_link.inverse().cpu().numpy()

                    # Load the match mesh.
                    match_mesh_fn = os.path.join(meshes_dir, match_mesh_name + ".obj")
                    match_mesh = trimesh.load(match_mesh_fn)

                    sdf_values = np.zeros((query_points.shape[0], link_to_world.shape[0]), dtype=np.float32)

                    # Transform match mesh to its plausible pose.
                    for pose_idx in range(world_to_link.shape[0]):
                        match_mesh_transformed = match_mesh.copy()
                        match_mesh_transformed.apply_transform(link_to_world[pose_idx])

                        # Compute SDF values for the match mesh.
                        plausible_sdf_values = get_sdf_values(match_mesh_transformed, query_points)
                        sdf_values[:, pose_idx] = plausible_sdf_values

                        if vis:
                            plt = Plotter((1, 2))
                            plt.at(0).show(Mesh([source_mesh.vertices, source_mesh.faces], c="red"),
                                           Mesh([match_mesh_transformed.vertices, match_mesh_transformed.faces],
                                                c="blue"))
                            plt.at(1).show(Points(query_points[plausible_sdf_values < 0.0]))
                            plt.interactive().close()

                    # Save the SDF values.
                    mmint_utils.save_gzip_pickle({
                        "query_points": query_points,  # TODO: This only needs to be saved once.
                        "sdf_values": sdf_values,
                    }, os.path.join(example_angle_sdf_dir, "%s.pkl.gzip" % match_mesh_name))

                    pbar.update(1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Visualize plausible reconstructions.")
    parser.add_argument("dataset_cfg_fn", type=str, help="Path to dataset config file.")
    parser.add_argument("split", type=str, help="Split to visualize.")
    parser.add_argument("-v", "--vis", action="store_true", help="Visualize the SDFs.")
    parser.set_defaults(vis=False)
    args = parser.parse_args()

    dataset_cfg_ = mmint_utils.load_cfg(args.dataset_cfg_fn)["data"][args.split]
    generate_sdf(dataset_cfg_, args.split)
