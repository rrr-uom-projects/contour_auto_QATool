# dataset.py
import os
from os.path import join
import numpy as np
import torch
import random
from torch_geometric.data import Dataset, Data
from utils import getFiles, windowLevelNormalize

class MyData(Data):
    def __cat_dim__(self, key, value, *args, **kwargs):
        if key in ['indiv_num_nodes', 'fname']:
            return None
        else:
            return super().__cat_dim__(key, value, *args, **kwargs)

#################################################
################ qaTool datasets ################
#################################################

class qaTool_classifier_dataset(Dataset):
    def __init__(self, mesh_dir, gs_classes_dir, ct_patches_dir, mesh_inds, perturbations_to_sample_per_epoch):
        self.mesh_dir = mesh_dir
        self.GS_mesh_dir = join(self.mesh_dir, "GS")
        self.gs_classes_dir = gs_classes_dir
        self.ct_patches_dir = ct_patches_dir
        self.mesh_inds = mesh_inds
        self.availableMeshes = [sorted(getFiles(self.GS_mesh_dir))[ind] for ind in mesh_inds]   ## Note: using self.GS_mesh_dir here to get stripped fnames, not a bug"
        self.perturbs_to_sample = perturbations_to_sample_per_epoch
        # not sure what these are
        self._indices = None
        self.transform = None
        # generate sets of deformation indices for each sample to prevent overlap
        inds = np.arange(0,100,1)
        np.random.shuffle(inds)
        self.def_ind_sets = inds.reshape(self.perturbs_to_sample, 100 // self.perturbs_to_sample)

    def len(self):
        return len(self.availableMeshes) * self.perturbs_to_sample

    def get(self, idx):
        # get 1 data sample
        if torch.is_tensor(idx):
           idx = idx.tolist()
        # find which mesh to grab
        mesh_idx = idx // self.perturbs_to_sample
        mesh_to_get = self.availableMeshes[mesh_idx]
        perturbation_to_get = random.choice(self.def_ind_sets[idx % self.perturbs_to_sample])

        # load relevant data
        mesh = torch.load(join(self.mesh_dir, mesh_to_get.replace('.pt', f'_{perturbation_to_get}.pt')))
        gs_classes = torch.argmax(torch.load(join(self.gs_classes_dir, mesh_to_get.replace('.pt', f'_{perturbation_to_get}.pt'))), dim=1)

        # get nodes and edges
        node_coords = mesh.pos.type(torch.float)
        edge_index = mesh.edge_index.type(torch.long)

        # load ct patches
        patches_tensor = torch.load(join(self.ct_patches_dir, mesh_to_get.replace('.pt', f'_{perturbation_to_get}.pt')))

        # generate edge_attr here for SplineConv
        # edge_attr: pseudo (Tensor) - Edge attributes, ie. pseudo coordinates, of shape (number_of_edges x number_of_edge_attributes) in the fixed interval [0, 1].
        # I'm going to start using pseudo 3D cartesian coordinates for each edge. I shall shift the origin to (0.5,0.5,0.5) to suit the fixed interval condition.
        # I.e. on the edge (n_i, n_j), n_i will be at (0.5,0.5,0.5), n_j will be at (x,y,z) 0 <= x,y,z <= 1 . The edge attr will be (x,y,z)
        # To achieve this, claculate the relative position for each edge pair in normal space, then scale and shift.
        edge_attr = torch.zeros((edge_index.size(1), 3))
        # first grab raw relative positions for each edge
        for edge_idx, edge in enumerate(edge_index.t()):
            n_i = node_coords[edge[0]]
            n_j = node_coords[edge[1]]
            relative_pos = n_i - n_j
            edge_attr[edge_idx] = relative_pos
        # now scale and shift
        max_val = torch.max(edge_attr)
        min_val = torch.min(edge_attr)
        scale_val = 1 / (max_val - min_val)
        edge_attr *= scale_val
        edge_attr += 0.5    # makes use of undirected graph property (symmetric edges, rel_pos vecs will be on [-0.5,0.5])
        edge_attr = torch.clamp(edge_attr, 0, 1)    # clamp just in case, shouldn't make a difference though

        graph = MyData(pos=node_coords, edge_index=edge_index, edge_attr=edge_attr, y=gs_classes, patches_tensor=patches_tensor, indiv_num_nodes=node_coords.size(0), fname=mesh_to_get.replace('.pt', f'_{perturbation_to_get}'))
        return graph

class qaTool_classifier_dataset_ablation(qaTool_classifier_dataset):
    def get(self, idx):
        # get 1 data sample
        if torch.is_tensor(idx):
           idx = idx.tolist()
        # find which mesh to grab
        mesh_idx = idx // self.perturbs_to_sample
        mesh_to_get = self.availableMeshes[mesh_idx]
        perturbation_to_get = random.choice(self.def_ind_sets[idx % self.perturbs_to_sample])

        # load relevant data
        mesh = torch.load(join(self.mesh_dir, mesh_to_get.replace('.pt', f'_{perturbation_to_get}.pt')))
        gs_classes = torch.argmax(torch.load(join(self.gs_classes_dir, mesh_to_get.replace('.pt', f'_{perturbation_to_get}.pt'))), dim=1)

        # get nodes and edges
        node_coords = mesh.pos.type(torch.float)
        edge_index = mesh.edge_index.type(torch.long)

        # load ct patches
        patches_tensor = torch.full_like(torch.load(join(self.ct_patches_dir, mesh_to_get.replace('.pt', f'_{perturbation_to_get}.pt'))), fill_value=0)

        # generate edge_attr here for SplineConv
        # edge_attr: pseudo (Tensor) - Edge attributes, ie. pseudo coordinates, of shape (number_of_edges x number_of_edge_attributes) in the fixed interval [0, 1].
        # I'm going to start using pseudo 3D cartesian coordinates for each edge. I shall shift the origin to (0.5,0.5,0.5) to suit the fixed interval condition.
        # I.e. on the edge (n_i, n_j), n_i will be at (0.5,0.5,0.5), n_j will be at (x,y,z) 0 <= x,y,z <= 1 . The edge attr will be (x,y,z)
        # To achieve this, claculate the relative position for each edge pair in normal space, then scale and shift.
        edge_attr = torch.zeros((edge_index.size(1), 3))
        # first grab raw relative positions for each edge
        for edge_idx, edge in enumerate(edge_index.t()):
            n_i = node_coords[edge[0]]
            n_j = node_coords[edge[1]]
            relative_pos = n_i - n_j
            edge_attr[edge_idx] = relative_pos
        # now scale and shift
        max_val = torch.max(edge_attr)
        min_val = torch.min(edge_attr)
        scale_val = 1 / (max_val - min_val)
        edge_attr *= scale_val
        edge_attr += 0.5    # makes use of undirected graph property (symmetric edges, rel_pos vecs will be on [-0.5,0.5])
        edge_attr = torch.clamp(edge_attr, 0, 1)    # clamp just in case, shouldn't make a difference though

        graph = MyData(pos=node_coords, edge_index=edge_index, edge_attr=edge_attr, y=gs_classes, patches_tensor=patches_tensor, indiv_num_nodes=node_coords.size(0), fname=mesh_to_get.replace('.pt', f'_{perturbation_to_get}'))
        return graph


#################################################
############ patchPredictor dataset #############
#################################################

class patchPredictor_dataset(torch.utils.data.Dataset):
    def __init__(self, ct_subvolume_dir, uniform_points_dir, samples_per_epoch, inds, seed=None):
        super(patchPredictor_dataset).__init__()
        all_pat_names = sorted(getFiles(ct_subvolume_dir))
        pat_names = [all_pat_names[ind] for ind in inds]
        self.uniform_pointsets = np.array([np.load(join(uniform_points_dir, pat_name)) for pat_name in pat_names])
        self.ct_subvolumes = torch.tensor([windowLevelNormalize(np.load(join(ct_subvolume_dir, pat_name)), level=40, window=350) for pat_name in pat_names], dtype=torch.float16).to('cuda')
        self.num_pats = len(pat_names)
        self.len = samples_per_epoch
        self.ct_subvolume_size = self.ct_subvolumes[0].size()
        if seed != None:
            random.seed(seed)
        self.pat_names = pat_names
    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
           idx = idx.tolist()
        # select ct_subvolume
        ct_subvolume = self.ct_subvolumes[idx % self.num_pats]
        # generate data
        if random.choice([True, False]):
            # generate positive example
            uniform_pointset = self.uniform_pointsets[idx % self.num_pats]
            point_idx = random.randint(0,19999)
            point = np.round(uniform_pointset[point_idx]).astype(int)
            patch = ct_subvolume[point[0]-2:point[0]+3, point[1]-2:point[1]+3, point[2]-2:point[2]+3]
            sample = {"patch": torch.unsqueeze(patch, dim=0), "label": torch.tensor([1,0], dtype=torch.float).to('cuda')}
            return sample
        else:
            # generate negative example
            cc_mid = random.randint(2, self.ct_subvolume_size[0]-4)
            ap_mid = random.randint(2, self.ct_subvolume_size[1]-4)
            lr_mid = random.randint(2, self.ct_subvolume_size[2]-4)
            patch = ct_subvolume[cc_mid-2:cc_mid+3, ap_mid-2:ap_mid+3, lr_mid-2:lr_mid+3]
            sample = {"patch": torch.unsqueeze(patch, dim=0), "label": torch.tensor([0,1], dtype=torch.float).to('cuda')}
            return sample