import numpy as np
import skimage.measure
import time
import torch
import trimesh


# From the DeepSDF repository https://github.com/facebookresearch/DeepSDF

def create_mesh(decoder, n=256, max_batch=40 ** 3, offset=None, scale=None):
    """
    Decoder is a function that takes in a batch of points and returns a batch of SDF values.
    """
    start = time.time()

    # TODO: How best to determine these?
    # min_bounds = [-0.053, -0.053, 0.02]
    # max_bounds = [0.053, 0.053, 0.076]
    min_bounds = [-1.1, -1.1, -1.1]
    max_bounds = [1.1, 1.1, 1.1]
    diff = 2.2

    # NOTE: the voxel_origin is actually the (bottom, left, down) corner, not the middle
    voxel_origin = min_bounds
    voxel_size = diff / (n - 1)

    overall_index = torch.arange(0, n ** 3, 1, out=torch.LongTensor())
    samples = torch.zeros(n ** 3, 4)

    # transform first 3 columns
    # to be the x, y, z index
    samples[:, 2] = overall_index % n
    samples[:, 1] = torch.div(overall_index.long(), n, rounding_mode="floor") % n
    samples[:, 0] = torch.div(torch.div(overall_index.long(), n, rounding_mode="floor"), n, rounding_mode="floor") % n

    # transform first 3 columns
    # to be the x, y, z coordinate
    samples[:, 0] = (samples[:, 0] * voxel_size) + voxel_origin[0]
    samples[:, 1] = (samples[:, 1] * voxel_size) + voxel_origin[1]
    samples[:, 2] = (samples[:, 2] * voxel_size) + voxel_origin[2]

    num_samples = n ** 3

    samples.requires_grad = False

    head = 0

    while head < num_samples:
        # print(head)
        sample_subset = samples[head: min(head + max_batch, num_samples), 0:3].cuda()
        sdf = decoder(sample_subset)

        samples[head: min(head + max_batch, num_samples), 3] = (
            sdf.detach().cpu()
        )
        head += max_batch

        del sdf
        torch.cuda.empty_cache()

    sdf_values = samples[:, 3]
    sdf_values = sdf_values.reshape(n, n, n)

    end = time.time()
    # print("sampling takes: %f" % (end - start))

    return convert_sdf_samples_to_mesh(
        sdf_values.data.cpu(),
        voxel_origin,
        voxel_size,
        offset,
        scale,
    )


def convert_sdf_samples_to_mesh(
        pytorch_3d_sdf_tensor,
        voxel_grid_origin,
        voxel_size,
        offset=None,
        scale=None,
):
    """
    Convert sdf samples to .ply
    :param pytorch_3d_sdf_tensor: a torch.FloatTensor of shape (n,n,n)
    :voxel_grid_origin: a list of three floats: the bottom, left, down origin of the voxel grid
    :voxel_size: float, the size of the voxels
    :ply_filename_out: string, path of the filename to save to
    This function adapted from: https://github.com/RobotLocomotion/spartan
    """
    numpy_3d_sdf_tensor = pytorch_3d_sdf_tensor.numpy()

    verts, faces, normals, values = np.zeros((0, 3)), np.zeros((0, 3)), np.zeros((0, 3)), np.zeros(0)
    try:
        verts, faces, normals, values = skimage.measure.marching_cubes(
            numpy_3d_sdf_tensor, level=0.0, spacing=[voxel_size] * 3
        )
    except ValueError as e:
        print("Error in mesh marching cubes: %s" % str(e))

    # transform from voxel coordinates to camera coordinates
    # note x and y are flipped in the output of marching_cubes
    mesh_points = np.zeros_like(verts)
    mesh_points[:, 0] = voxel_grid_origin[0] + verts[:, 0]
    mesh_points[:, 1] = voxel_grid_origin[1] + verts[:, 1]
    mesh_points[:, 2] = voxel_grid_origin[2] + verts[:, 2]

    # apply additional offset and scale
    if scale is not None:
        mesh_points = mesh_points / scale
    if offset is not None:
        mesh_points = mesh_points - offset

    mesh = trimesh.Trimesh(vertices=mesh_points, faces=faces, normals=normals)
    return mesh
