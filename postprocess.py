import torch
import torch_geometric
import numpy as np
from os.path import join
from utils import getFiles
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib
matplotlib.rcParams.update({"font.size": 16})

graph_dir = join("/path/to/directory/containing/preprocessed/data/", "graph_objects/")  ## TODO: update path variable here ##
preds_dir = "/path/to/soft_predictions/directory/"  ## TODO: update path variable here ##
fnames = sorted(getFiles(preds_dir))
mean_proportions = np.zeros((3, len(fnames)))
for fdx, fname in enumerate(fnames):
    graph = torch.load(join(graph_dir, fname.replace('.npy','.pt')))
    pred = np.load(join(preds_dir, fname))
    hard_pred = np.argmax(pred, axis=1)
    num_nodes = graph.pos.size(0)
    proportions = np.zeros((num_nodes))
    mean_proportions[0, fdx] = ((hard_pred[graph.edge_index[0]]==hard_pred[graph.edge_index[1]]).sum() + (hard_pred[graph.edge_index[0]]==hard_pred[graph.edge_index[1]]-1).sum() + (hard_pred[graph.edge_index[0]]==hard_pred[graph.edge_index[1]]+1).sum()) / graph.edge_index.size(1)

preds_dir = "/path/to/GNN_ablation/soft_predictions/directory/" ## TODO: update path variable here ##
for fdx, fname in enumerate(fnames):
    graph = torch.load(join(graph_dir, fname.replace('.npy','.pt')))
    pred = np.load(join(preds_dir, fname))
    hard_pred = np.argmax(pred, axis=1)
    num_nodes = graph.pos.size(0)
    proportions = np.zeros((num_nodes))
    mean_proportions[1, fdx] = ((hard_pred[graph.edge_index[0]]==hard_pred[graph.edge_index[1]]).sum() + (hard_pred[graph.edge_index[0]]==hard_pred[graph.edge_index[1]]-1).sum() + (hard_pred[graph.edge_index[0]]==hard_pred[graph.edge_index[1]]+1).sum()) / graph.edge_index.size(1)

preds_dir = join("/path/to/directory/containing/preprocessed/data/", "signed_classes/") ## TODO: update path variable here ##
for fdx, fname in enumerate(fnames):
    graph = torch.load(join(graph_dir, fname.replace('.npy','.pt')))
    pred = torch.load(join(preds_dir, fname.replace('.npy','.pt'))).numpy()
    hard_pred = np.argmax(pred, axis=1)
    num_nodes = graph.pos.size(0)
    proportions = np.zeros((num_nodes))
    mean_proportions[2, fdx] = ((hard_pred[graph.edge_index[0]]==hard_pred[graph.edge_index[1]]).sum() + (hard_pred[graph.edge_index[0]]==hard_pred[graph.edge_index[1]]-1).sum() + (hard_pred[graph.edge_index[0]]==hard_pred[graph.edge_index[1]]+1).sum()) / graph.edge_index.size(1)
    
fig, ax = plt.subplots(figsize=(8, 4))
colors = ['g','r','y']
labels = ["Full model", "No GNN", "Ground truth"]
ax.hist(mean_proportions[0], bins=np.arange(0.8,1,0.002), color=colors[0], alpha=0.75)
ax.hist(mean_proportions[1], bins=np.arange(0.8,1,0.002), color=colors[1], alpha=0.75)
ax.hist(mean_proportions[2], bins=np.arange(0.8,1,0.002), color=colors[2], alpha=0.75)
ax.set_xlabel("Proportion of nodes with matching (+/- 1) neigbours")
ax.set_ylabel("Count")
ax.set_xticks([0.8,0.85,0.9,0.95,1])
m_s = []
for color, label in zip(colors, labels):
    m_s.append(mlines.Line2D([],[], color=color, marker='s', linestyle='None', mew=0, mec='k', markersize=15, label=label))
ax.legend(handles=m_s, fontsize="16", loc='upper left')
ax.set_xlim(0.8,1)
plt.tight_layout()
plt.savefig("neighbours_fig.pdf")

for i in range(3):
    print(np.mean(mean_proportions[i]), np.std(mean_proportions[i]))

source = join("/path/to/results/directory/", "lr1e3_cosS_bs16_aggradd") ## TODO: update path variable here ##
res_full = np.load(source + ".npy")
res_encAbl = np.load(source + "_blankCTAbl_0.npy")
res_GNNAbl = np.load(source + "_GNNAbl.npy")
res_preAbl = np.load(source + "_noPre.npy")
stack = np.stack((res_full, res_encAbl, res_GNNAbl, res_preAbl))

def wide_precision(array):
    internal = (array[0,0] + array[1,0]) / np.sum(array[:,0])
    external = (array[4,4] + array[3,4]) / np.sum(array[:,4])
    return np.array([internal, external])

def wide_recall(array):
    internal = (array[0,0] + array[0,1]) / np.sum(array[0,:])
    external = (array[4,4] + array[4,1]) / np.sum(array[4,:])
    return np.array([internal, external])

def precision(array):
    internal = (array[0,0]) / np.sum(array[:,0])
    external = (array[4,4]) / np.sum(array[:,4])
    return np.array([internal, external])

def recall(array):
    internal = array[0,0] / np.sum(array[0,:])
    external = array[4,4] / np.sum(array[4,:])
    return np.array([internal, external])

def F1(precision, recall):
    return (2 * precision * recall) / (precision + recall)

print("Precision = proportion of true positives")
print("Recall = proportion of positives identified")
labels = ["full", "encAbl", "GNNAbl", "noPre"]
for ldx, label in enumerate(labels):
    print("--- ", label, " ---")
    print("Precision: ", wide_precision(stack[ldx]))
    print("Recall: ", recall(stack[ldx]))
    print("F1: ", F1(wide_precision(stack[ldx]), recall(stack[ldx])))