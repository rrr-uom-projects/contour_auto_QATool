import matplotlib.pyplot as plt
import numpy as np
import torch
import open3d as o3d
import pyvista
from copy import deepcopy
from os.path import join

def pyvistarise(mesh):
    return pyvista.PolyData(np.asarray(mesh.vertices), np.insert(np.asarray(mesh.triangles), 0, 3, axis=1), deep=True, n_faces=len(mesh.triangles))

def visualise_pertubations(mesh, verts, classes, pred_classes):
    # make copy and shift
    mesh_copy = deepcopy(mesh)
    mesh.vertices = o3d.utility.Vector3dVector((np.asarray(mesh.vertices)[:] - np.array([0,30,0])))
    mesh_copy.vertices = o3d.utility.Vector3dVector((np.asarray(mesh_copy.vertices)[:] + np.array([0,30,0])))
    # convert to pyvista
    pyv_mesh = pyvistarise(mesh)
    pyv_mesh_copy = pyvistarise(mesh_copy)
    # plot
    pyvista.global_theme.background = 'white'
    plotter = pyvista.Plotter()
    plotter.add_mesh(pyv_mesh, color=[0.5, 0.706, 1.], opacity=0.75)
    plotter.add_mesh(pyv_mesh_copy, color=[0.5, 0.706, 1.], opacity=0.75)
    # add class points
    # ground truth
    cmap = plt.cm.get_cmap('bwr')
    num_classes = classes.shape[1]
    for pc_idx in range(num_classes):
        if (pc_idx==np.argmax(classes, axis=1)).any():
            this_pc_verts = verts[pc_idx==np.argmax(classes, axis=1)] - np.array([0,30,0])
            print(this_pc_verts)
            pc = pyvista.PolyData(this_pc_verts)
            plotter.add_mesh(pc, render_points_as_spheres=True, color=cmap(pc_idx/(num_classes-1))[:3], point_size=15)
    # pred
    num_classes = pred_classes.shape[1]
    for pc_idx in range(num_classes):
        if (pc_idx==np.argmax(pred_classes, axis=1)).any():
            this_pc_verts = verts[pc_idx==np.argmax(pred_classes, axis=1)] + np.array([0,30,0])
            pc = pyvista.PolyData(this_pc_verts)
            plotter.add_mesh(pc, render_points_as_spheres=True, color=cmap(pc_idx/(num_classes-1))[:3], point_size=15)
    # add labels
    poly = pyvista.PolyData([np.array([140,155,300]), np.array([140,215,300])])
    poly["My Labels"] = ["Ground truth", "Prediction"]
    plotter.add_point_labels(poly, "My Labels", point_size=0, font_size=36)        
    # render
    plotter.store_image = True
    plotter.camera.position = (267.23733688822307, 185.79964888533257, -37.21093737148561)
    plotter.camera.focal_point = (182.16842651367188, 203.39258575439453, 303.5139617919922)
    plotter.camera.up = (0.9701952368133866, -0.0017313380567325239, 0.2423183957794992)
    camera_pos = plotter.show(screenshot="./QA_tool/visualisations/viz_gndtruth_and_pred.png", window_size=[1600, 1000], auto_close=True)
    print(camera_pos)

def main():
    patient_fname = "patient_fname"                                 ## TODO: update name variable here ##
    source_dir = "/path/to/directory/containing/preprocessed/data/" ## TODO: update path variable here ##
    
    verts = torch.load(join(source_dir, "graph_objects/", f"{patient_fname}.pt")).pos.numpy()
    faces = np.load(join(source_dir, "triangles_smooth/", f"{patient_fname}.npy"))
    perturbed_mesh = o3d.geometry.TriangleMesh()
    perturbed_mesh.vertices = o3d.utility.Vector3dVector(verts)
    perturbed_mesh.triangles = o3d.utility.Vector3iVector(faces)

    perturbed_mesh.compute_vertex_normals()
    classes = torch.load(join(source_dir, "signed_classes/", f"{patient_fname}.pt")).numpy()
    pred_classes = np.load(join("/path/to/predicted/classes/", f"{patient_fname}.npy"))     ## TODO: update path variable here ##
    visualise_pertubations(perturbed_mesh, verts, classes, pred_classes)

if __name__ == '__main__':
    main()

