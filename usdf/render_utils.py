import numpy as np


def depth_to_pointcloud(depth, fov):
    """
    Deproject depth image to pointcloud.
    Credit to: https://github.com/mmatl/pyrender/issues/14#issuecomment-485881479

    Args:
        depth: depth image
        fov: field of view in radians
    Returns: pointcloud
    """
    fy = fx = 0.5 / np.tan(fov * 0.5)  # assume aspectRatio is one.
    height = depth.shape[0]
    width = depth.shape[1]

    mask = np.where(depth > 0)

    x = mask[1]
    y = mask[0]

    normalized_x = (x.astype(np.float32) - width * 0.5) / width
    normalized_y = (y.astype(np.float32) - height * 0.5) / height
    normalized_y = -normalized_y

    world_x = normalized_x * depth[y, x] / fx
    world_y = normalized_y * depth[y, x] / fy
    world_z = -depth[y, x]
    ones = np.ones(world_z.shape[0], dtype=np.float32)

    return np.vstack((world_x, world_y, world_z, ones)).T
