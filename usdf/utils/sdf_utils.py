import numpy as np
import trimesh
import open3d as o3d


def sample_points_from_ball(n_points, ball_radius=1.1):
    """
    Sample points evenly from inside unit ball by sampling points in the unit cube
    and rejecting points outside the unit ball.

    Args:
        n_points: number points to return
        ball_radius: radius of ball to sample from
    Returns: point cloud of sampled query points
    """
    points = np.empty((0, 3), dtype=np.float32)

    while len(points) < n_points:
        # Sample points in the unit cube.
        new_points = np.random.uniform(-1, 1, size=(n_points, 3))

        # Reject points outside the unit ball.
        mask = np.linalg.norm(new_points, axis=1) <= ball_radius
        points = np.concatenate([points, new_points[mask]], axis=0)

    return points[:n_points]


def get_sdf_query_points(mesh: trimesh.Trimesh, n_random: int = 10000, n_off_surface: int = 10000,
                         noise: float = 0.004):
    if n_random > 0:
        query_points_random = sample_points_from_ball(n_random)
    else:
        query_points_random = np.empty([0, 3], dtype=float)

    if n_off_surface > 0:
        query_points_surface = mesh.sample(n_off_surface)
        query_points_surface += np.random.normal(0.0, noise, size=query_points_surface.shape)
    else:
        query_points_surface = np.empty([0, 3], dtype=float)

    return np.concatenate([query_points_random, query_points_surface])


def get_sdf_values(mesh: trimesh.Trimesh, query_points: np.ndarray):
    # Convert mesh to open3d mesh.
    mesh_o3d = o3d.geometry.TriangleMesh(
        vertices=o3d.utility.Vector3dVector(mesh.vertices),
        triangles=o3d.utility.Vector3iVector(mesh.faces)
    )

    # Build o3d scene with triangle mesh.
    tri_mesh_legacy = o3d.t.geometry.TriangleMesh.from_legacy(mesh_o3d)
    scene = o3d.t.geometry.RaycastingScene()
    _ = scene.add_triangles(tri_mesh_legacy)

    # Compute SDF to surface.
    query_points_ = o3d.core.Tensor(query_points, dtype=o3d.core.Dtype.Float32)
    signed_distance = scene.compute_signed_distance(query_points_)
    signed_distance_np = signed_distance.numpy()

    return signed_distance_np
