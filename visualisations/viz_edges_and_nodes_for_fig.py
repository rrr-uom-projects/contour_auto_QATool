import matplotlib.pyplot as plt
import numpy as np
import torch
import open3d as o3d
import pyvista
from copy import deepcopy

def pyvistarise(mesh):
    return pyvista.PolyData(np.asarray(mesh.vertices), np.insert(np.asarray(mesh.triangles), 0, 3, axis=1), deep=True, n_faces=len(mesh.triangles))

def visualise_pertubations(mesh, verts, classes):
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
    plotter.add_mesh(pyv_mesh, show_edges=True, line_width=3, edge_color=[0,0,0,1], color=[0.5, 0.706, 1.])
    plotter.add_mesh(pyv_mesh_copy, color=[0.5, 0.706, 1.], opacity=0.75)
    # add vertices with ground truth classes
    cmap = plt.cm.get_cmap('bwr')
    num_classes = classes.shape[1]
    for pc_idx in range(num_classes):
        if (pc_idx==np.argmax(classes, axis=1)).any():
            this_pc_verts = verts[pc_idx==np.argmax(classes, axis=1)] + np.array([0,30,0])
            print(this_pc_verts)
            pc = pyvista.PolyData(this_pc_verts)
            plotter.add_mesh(pc, render_points_as_spheres=True, color=cmap(pc_idx/(num_classes-1))[:3], point_size=15)
    # render
    plotter.store_image = True
    plotter.camera.position = (267.23733688822307, 185.79964888533257, -37.21093737148561)
    plotter.camera.focal_point = (182.16842651367188, 203.39258575439453, 303.5139617919922)
    plotter.camera.up = (0.9701952368133866, -0.0017313380567325239, 0.2423183957794992)
    camera_pos = plotter.show(screenshot="./QA_tool/visualisations/viz_edges_and_nodes_for_fig.png",window_size=[1600, 1000], auto_close=True)
    print(camera_pos)

def main():
    def_num = "7"
    pat_num = "629"

    verts = torch.load(f"C:/PhD/ESTRO22/QA_tool_data/more_var_partial/graph_objects/0522c0{pat_num}_R_{def_num}.pt").pos.numpy()
    faces = np.load(f"C:/PhD/ESTRO22/QA_tool_data/more_var_partial/triangles_smooth/0522c0{pat_num}_R_{def_num}.npy")
    perturbed_mesh = o3d.geometry.TriangleMesh()
    perturbed_mesh.vertices = o3d.utility.Vector3dVector(verts)
    perturbed_mesh.triangles = o3d.utility.Vector3iVector(faces)

    perturbed_mesh.compute_vertex_normals()
    classes = torch.load(f"C:/PhD/ESTRO22/QA_tool_data/more_var_partial/signed_classes/0522c0{pat_num}_R_{def_num}.pt").numpy()
    visualise_pertubations(perturbed_mesh, verts, classes)

if __name__ == '__main__':
    main()

