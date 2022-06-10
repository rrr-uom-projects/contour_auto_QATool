import os
import numpy as np
import SimpleITK as sitk
from skimage.transform import rescale
from skimage.measure import marching_cubes
import open3d as o3d
from utils import getDirs, getFiles
from tqdm import tqdm

import torch
import torch_geometric
from torch_geometric.transforms import FaceToEdge, ToUndirected

segs_to_get = ['Parotid-Lt.nrrd', 'Parotid-Rt.nrrd']
desired_spacing = [2.5, 1, 1]
source_dir = "/path/to/directory/containing/the/nikolov_et_al/nrrd/data"    ## TODO: update path variable here ##
output_dir = "/path/to/directory/in/which/to/put/preprocessed/data/"        ## TODO: update path variable here ##
crop_midpoints = crop_midpoints.astype(int)
pat_dirs = list(filter(lambda x: x != "TCGA-CV-A6JY", sorted(getDirs(source_dir))))

# resample all CTs to isotropic (1 x 1 x 2.5)
for pdx, pat_dir in enumerate(tqdm(pat_dirs)):
    CT = sitk.ReadImage(os.path.join(source_dir, pat_dir, "CT_IMAGE.nrrd"))
    CT_spacing = np.array(CT.GetSpacing())
    npy_CT = sitk.GetArrayFromImage(CT)
    # resample CT
    scale_factor = CT_spacing[[2,0,1]] / desired_spacing
    npy_CT = rescale(npy_CT, scale=scale_factor, order=3, preserve_range=True, anti_aliasing=True)
    seg_fnames = getFiles(os.path.join(source_dir, pat_dir, "segmentations"))
    for seg_idx, seg_fname in enumerate(seg_fnames):       
        if (seg_fname in segs_to_get) or (seg_fname.replace('_','-') in segs_to_get):      
            seg = sitk.ReadImage(os.path.join(source_dir, pat_dir, "segmentations", seg_fname))
            assert((np.array(seg.GetSpacing()) == CT_spacing).all())
            npy_seg = sitk.GetArrayFromImage(seg)
            # resample
            npy_seg = rescale(npy_seg, scale=scale_factor, order=0, preserve_range=True, anti_aliasing=False)
            
            # flip the right to match left
            if seg_fname[8]=="R":
                npy_seg = np.flip(npy_seg, axis=2).copy()

            # crop ct and seg mask to sub-volume (48, 112, 96) (for efficiency)
            inds = np.argwhere((npy_seg == 1))
            cc, ap, lr = (max(inds[:,0]) + min(inds[:,0])) // 2, (max(inds[:,1]) + min(inds[:,1])) // 2, (max(inds[:,2]) + min(inds[:,2])) // 2
            if seg_fname[8] == "R":
                cropped_CT = np.flip(npy_CT, axis=2).copy()[cc-24:cc+24, ap-56:ap+56, lr-48:lr+48]
            else:
                cropped_CT = npy_CT.copy()[cc-24:cc+24, ap-56:ap+56, lr-48:lr+48]
            npy_seg = npy_seg[cc-24:cc+24, ap-56:ap+56, lr-48:lr+48]

            # get surface
            verts, faces, normals, values = marching_cubes(volume=npy_seg, level=0.5, spacing=(1,1,1)) # keep spacing in voxel space
            
            # use open3d to smooth mesh
            seg_mesh = o3d.geometry.TriangleMesh()
            seg_mesh.vertices = o3d.utility.Vector3dVector(verts)
            seg_mesh.triangles = o3d.utility.Vector3iVector(faces)
            seg_mesh.vertex_normals = o3d.utility.Vector3dVector(normals)
            seg_mesh = seg_mesh.filter_smooth_taubin(number_of_iterations=100)
            seg_mesh = seg_mesh.simplify_quadric_decimation(target_number_of_triangles=1000)
            seg_mesh = seg_mesh.filter_smooth_taubin(number_of_iterations=10)

            # sample points uniformly on the surface
            pcd = seg_mesh.sample_points_uniformly(number_of_points=100000)
            uniform_points = np.asarray(pcd.points)
            np.save(os.path.join(output_dir, "pretrain_uniform_points", f"{pat_dir}_{seg_fname[8]}.npy"), uniform_points)

            # save the image patches
            np.save(os.path.join(output_dir, "pretrain_ct_patches", f"{pat_dir}_{seg_fname[8]}.npy"), cropped_CT)