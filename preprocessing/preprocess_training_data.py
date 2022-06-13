import os
from os.path import join
import numpy as np
import SimpleITK as sitk

import open3d as o3d
from utils import *
from tqdm import tqdm

import torch
import torch_geometric
from torch_geometric.transforms import FaceToEdge, ToUndirected

from skimage.transform import rescale
from skimage.measure import marching_cubes
from scipy.ndimage import distance_transform_edt
from scipy.signal import fftconvolve
from scipy.interpolate import interpn

segs_to_get = ['Parotid-Lt.nrrd', 'Parotid-Rt.nrrd']
desired_spacing = [2.5, 1, 1]
num_deformations_to_generate_per_seg = 100
source_dir = "/path/to/directory/containing/the/nikolov_et_al/nrrd/data"    ## TODO: update path variable here ##
output_dir = "/path/to/directory/in/which/to/put/preprocessed/data/"        ## TODO: update path variable here ##
pat_dirs = list(filter(lambda x: x != "TCGA-CV-A6JY", sorted(getDirs(source_dir))))

global_signed_classes = torch.zeros((5))
# resample all CTs to isotropic (1 x 1 x 2.5)
for pdx, pat_dir in enumerate(tqdm(pat_dirs)):
    CT = sitk.ReadImage(join(source_dir, pat_dir, "CT_IMAGE.nrrd"))
    CT_spacing = np.array(CT.GetSpacing())
    npy_CT = sitk.GetArrayFromImage(CT)
    # resample CT
    scale_factor = CT_spacing[[2,0,1]] / desired_spacing
    npy_CT = rescale(npy_CT, scale=scale_factor, order=3, preserve_range=True, anti_aliasing=True)
    # normalise CT
    npy_CT = windowLevelNormalize(npy_CT, level=40, window=350)
    seg_fnames = getFiles(join(source_dir, pat_dir, "segmentations"))
    for seg_idx, seg_fname in enumerate(seg_fnames):       
        if (seg_fname in segs_to_get) or (seg_fname.replace('_','-') in segs_to_get):
            seg = sitk.ReadImage(join(source_dir, pat_dir, "segmentations", seg_fname))
            assert((np.array(seg.GetSpacing()) == CT_spacing).all())
            npy_seg = sitk.GetArrayFromImage(seg)
            # resample
            npy_seg = rescale(npy_seg, scale=scale_factor, order=0, preserve_range=True, anti_aliasing=False)
            assert(npy_seg.shape==npy_CT.shape)

            # flip the right to match left
            if seg_fname[8] == "R":
                flipped_npy_CT = np.flip(npy_CT, axis=2).copy()
                npy_seg = np.flip(npy_seg, axis=2).copy()

            # get distance xfm
            dist_xfm = distance_transform_edt((~(npy_seg.astype(bool))).astype(float), sampling=(2.5,1,1)) - distance_transform_edt(npy_seg, sampling=(2.5,1,1))

            # get gold standard
            verts, faces, normals, values = marching_cubes(volume=dist_xfm, level=0., spacing=(2.5,1,1))

            # use open3d to smooth mesh
            seg_mesh = o3d.geometry.TriangleMesh()
            seg_mesh.vertices = o3d.utility.Vector3dVector(verts)
            seg_mesh.triangles = o3d.utility.Vector3iVector(faces)
            seg_mesh.vertex_normals = o3d.utility.Vector3dVector(normals)
            seg_mesh = seg_mesh.filter_smooth_taubin(number_of_iterations=100)
            seg_mesh = seg_mesh.simplify_quadric_decimation(target_number_of_triangles=1000)
            seg_mesh = seg_mesh.filter_smooth_taubin(number_of_iterations=10)
            seg_mesh.remove_unreferenced_vertices()
            #o3d.io.write_triangle_mesh(join(output_dir, f"{pat_dir}_{seg_fname[8]}_GS.ply"), seg_mesh)
            
            # grab the smoothed vertices and triangles
            verts_smooth = np.asarray(seg_mesh.vertices)
            triangles_smooth = np.asarray(seg_mesh.triangles)
            np.save(join(output_dir, "triangles_smooth", "GS", f"{pat_dir}_{seg_fname[8]}.npy"), triangles_smooth)
            pos = torch.from_numpy(verts_smooth).to(torch.float)
            face = torch.from_numpy(triangles_smooth).t().contiguous()
            data = ToUndirected()(FaceToEdge()(torch_geometric.data.Data(pos=pos, face=face)))
            torch.save(data, join(output_dir, "graph_objects", "GS", f"{pat_dir}_{seg_fname[8]}.pt"))

            # sample points uniformly on the surface to use for nearest neighbour loss function
            pcd = seg_mesh.sample_points_uniformly(number_of_points=100000)
            uniform_points = np.asarray(pcd.points)
            np.save(join(output_dir, "GS_uniform_points", f"{pat_dir}_{seg_fname[8]}.npy"), uniform_points)

            # Generate training data using realistic deformations!
            for deformation_num in range(num_deformations_to_generate_per_seg):
                nan_nodes = True
                while nan_nodes:
                    ## First generate gaussian noise map and convolve with a point spread function
                    # compute size of noise volume to be applied
                    starts, extents = get_bounds(npy_seg, margin=(5, 10, 10))
                    gaussian_noise_map = np.random.normal(scale=0.035, size=extents)
                    sigma = 7.5     # width of kernel
                    xx, yy, zz = np.meshgrid(np.arange(-25,26,1), np.arange(-25,26,1), np.arange(-25,26,1))
                    kernel = np.exp(-(xx**2 + yy**2 + zz**2)/(2*sigma**2))
                    noise = fftconvolve(gaussian_noise_map, kernel, mode="same")
                    full_noise = np.zeros_like(npy_seg)
                    full_noise[starts[0]:starts[0] + extents[0], starts[1]:starts[1] + extents[1], starts[2]:starts[2] + extents[2]] = noise

                    # Augment distance transform 
                    deformed_dist_xfm = dist_xfm + full_noise

                    # get surface
                    verts, faces, normals, values = marching_cubes(volume=deformed_dist_xfm, level=0., spacing=(2.5,1,1)) # keep spacing in voxel space
                    
                    # use open3d to smooth mesh
                    seg_mesh = o3d.geometry.TriangleMesh()
                    seg_mesh.vertices = o3d.utility.Vector3dVector(verts)
                    seg_mesh.triangles = o3d.utility.Vector3iVector(faces)
                    seg_mesh.vertex_normals = o3d.utility.Vector3dVector(normals)

                    # clean mesh using connected components
                    seg_mesh = clean_mesh(mesh=seg_mesh, max_components_to_keep="auto")

                    # smooth mesh
                    seg_mesh = seg_mesh.filter_smooth_taubin(number_of_iterations=100)
                    seg_mesh = seg_mesh.simplify_quadric_decimation(target_number_of_triangles=1000)
                    seg_mesh = seg_mesh.filter_smooth_taubin(number_of_iterations=10)
                    seg_mesh.remove_unreferenced_vertices()

                    # grab the smoothed vertices and triangles
                    verts_smooth = np.asarray(seg_mesh.vertices)
                    triangles_smooth = np.asarray(seg_mesh.triangles)
                    pos = torch.from_numpy(verts_smooth).to(torch.float)
                    face = torch.from_numpy(triangles_smooth).t().contiguous()
                    data = ToUndirected()(FaceToEdge()(torch_geometric.data.Data(pos=pos, face=face)))

                    # test if node is NaN somehow
                    if pos.isnan().any():
                        print(f"Nan-node detected, let's try this again... (seg: {pat_dir}, def_num: {deformation_num})")
                    else:
                        np.save(join(output_dir, "triangles_smooth", f"{pat_dir}_{seg_fname[8]}_{deformation_num}.npy"), triangles_smooth)
                        torch.save(data, join(output_dir, "graph_objects", f"{pat_dir}_{seg_fname[8]}_{deformation_num}.pt"))
                        nan_nodes = False
                
                # save the image patches
                node_coords = data.pos / torch.tensor(desired_spacing)
                if seg_fname[8] == "R":
                    patches_tensor = sample_ct_on_nodes(node_coords, torch.tensor(flipped_npy_CT))
                else:
                    patches_tensor = sample_ct_on_nodes(node_coords, torch.tensor(npy_CT))
                # save node patches
                torch.save(patches_tensor, join(output_dir, "ct_patches", f"{pat_dir}_{seg_fname[8]}_{deformation_num}.pt"))

                # compute signed dists
                points = (np.arange(npy_seg.shape[0]), np.arange(npy_seg.shape[1]), np.arange(npy_seg.shape[2]))
                signed_dists = interpn(points=points, values=dist_xfm, xi=node_coords.numpy())

                # create classes for each node to be classified into 
                # signed class bins 0: -2.5mm-, 1: -2.5 - -1mm, 2: -1 - 1mm, 3: 1 - 2.5mm, 4: 2.5mm+
                n_classes = 5
                node_classes_signed = torch.zeros(size=(pos.size(0), n_classes), dtype=int)
                for node_idx in range(pos.size(0)):
                    node_classes_signed[node_idx, get_class_signed(dist=signed_dists[node_idx])] = 1
                global_signed_classes += torch.tensor([node_classes_signed[:,i].sum() for i in range(n_classes)])
                torch.save(node_classes_signed, join(output_dir, "signed_classes", f"{pat_dir}_{seg_fname[8]}_{deformation_num}.pt"))

torch.save(global_signed_classes, join(output_dir, "all_signed_classes.pt"))