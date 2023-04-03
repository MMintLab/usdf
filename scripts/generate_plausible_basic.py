import os

def generate_plausible_basic(dataset_cfg: dict, split: str, vis: bool = True):
    """
    Generate plausible pointclouds for each mesh in the dataset matching to the partial views generated from all other
    meshes.
    """
    surface_N = 1000  # Number of points to use in partial pointcloud.
    N_angles = dataset_cfg["N_angles"]  # Number of angles each mesh is rendered from.

    meshes_dir = dataset_cfg["meshes_dir"]
    dataset_dir = dataset_cfg["dataset_dir"]
    partials_dir = os.path.join(dataset_dir, "partials")
    plausibles_dir = os.path.join(dataset_dir, "plausibles")