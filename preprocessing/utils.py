## utils.py
# some useful functions!
import numpy as np
import torch
import os
import sys
import warnings

def getDirs(parent_dir):
    ls = []
    for dir_name in os.listdir(parent_dir):
        path = os.path.join(parent_dir, dir_name)
        if os.path.isdir(path):
            ls.append(dir_name)
    return ls

def getFiles(targetdir):
    ls = []
    for fname in os.listdir(targetdir):
        path = os.path.join(targetdir, fname)
        if os.path.isdir(path):
            continue
        ls.append(fname)
    return ls

def sample_ct_on_nodes(node_coords, ct_subvolume, patch_size=5):
    #
    # IMPORTANT: expects node coordinates in CT voxels coordinate system
    #
    low_bound = patch_size // 2
    high_bound = (patch_size + 1) // 2 
    patches_tensor = torch.zeros((node_coords.size(0), patch_size, patch_size, patch_size))
    # sample a 3D patch on the ct subvolume for every node in the mesh
    for node_idx, node_coord in enumerate(node_coords):
        # use clamping to deal with the case that the patch is outside the ct_subvolume -> raise a warning though?
        cent_coord_unclamped = node_coord
        cent_coord = torch.clamp(cent_coord_unclamped, min=torch.tensor([low_bound,low_bound,low_bound]), max=torch.tensor(ct_subvolume.shape) - high_bound)
        if (torch.round(cent_coord_unclamped) != torch.round(cent_coord)).any():
            warnings.warn("Node patch outside of the CT subvolume!", category=RuntimeWarning)
            exit()
        cent_coord = torch.round(cent_coord).type(torch.int)
        cc_lo, cc_hi = cent_coord[0] - low_bound, cent_coord[0] + high_bound
        ap_lo, ap_hi = cent_coord[1] - low_bound, cent_coord[1] + high_bound
        lr_lo, lr_hi = cent_coord[2] - low_bound, cent_coord[2] + high_bound
        patches_tensor[node_idx] = ct_subvolume[cc_lo:cc_hi, ap_lo:ap_hi, lr_lo:lr_hi]
    return patches_tensor

def windowLevelNormalize(image, level, window):
    minval = level - window/2
    maxval = level + window/2
    wld = np.clip(image, minval, maxval)
    wld -= minval
    wld *= (1 / window)
    return wld

def clean_mesh(mesh, max_components_to_keep="auto"):
    # get number to keep
    if max_components_to_keep == "auto":
        _, num_tris_per_cluster, _ = mesh.cluster_connected_triangles()
        max_components_to_keep = len(num_tris_per_cluster)
    # identify connected triangles
    cluster_ind_per_tri, num_tris_per_cluster, _ = mesh.cluster_connected_triangles()
    if len(num_tris_per_cluster) < (max_components_to_keep + 1):
        return mesh # mesh needs no cleaning
    # identify triangles to toss
    num_tris = np.asarray(mesh.triangles).shape[0]
    cut_ind = max_components_to_keep - 1
    tris_to_yeet = list(filter(lambda tri_index: cluster_ind_per_tri[tri_index] > cut_ind, range(num_tris)))
    # remove triangles
    mesh.remove_triangles_by_index(triangle_indices=tris_to_yeet)
    # remove nodes
    mesh.remove_unreferenced_vertices()
    # return cleaned mesh
    return mesh

def get_bounds(seg, margin=(5, 10, 10)):
    # input: numpy binary segmentation, output: tuple of start positions (cc,ap,lr) and tuple of extents of bounding box 
    bounds = np.argwhere((seg == 1))
    cc = np.array((min(bounds[:,0]) - margin[0], max(bounds[:,0]) + margin[0]))
    ap = np.array((min(bounds[:,1]) - margin[1], max(bounds[:,1]) + margin[1]))
    lr = np.array((min(bounds[:,2]) - margin[2], max(bounds[:,2]) + margin[2]))
    # clip to avoid going out of range
    cc = np.clip(cc, 0, seg.shape[0]-1)
    ap = np.clip(ap, 0, seg.shape[1]-1)
    lr = np.clip(lr, 0, seg.shape[2]-1)
    # put the tuples together
    starts = (cc[0], ap[0], lr[0])
    extents = (cc[1]-cc[0], ap[1]-ap[0], lr[1]-lr[0])
    return starts, extents

def get_class_signed(dist):
    # signed 0: -2.5mm-, 1: -2.5 - -1mm, 2: -1 - 1mm, 3: 1 - 2.5mm, 4: 2.5mm+
    if dist < -2.5:
        return 0
    if -2.5 < dist < -1:
        return 1
    if -1 < dist < 1:
        return 2
    if 1 < dist < 2.5:
        return 3
    return 4

def get_class_unsigned(dist):
    # unsigned 0: 0 - 1mm, 1: 1 - 2.5mm, 2: 2.5 - 5mm, 3: 5mm+
    if -1 < dist < 1:
        return 0
    if -2.5 < dist < 2.5:
        return 1
    if -5 < dist < 5:
        return 2
    return 3