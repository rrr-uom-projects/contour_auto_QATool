import matplotlib.pyplot as plt
import numpy as np
import torch
import open3d as o3d
import pyvista
from copy import deepcopy

def pyvistarise(mesh):
    return pyvista.PolyData(np.asarray(mesh.vertices), np.insert(np.asarray(mesh.triangles), 0, 3, axis=1), deep=True, n_faces=len(mesh.triangles))

def visualise_pertubations(pre_mesh, post_mesh, verts, classes):
    # copy and shift
    post_mesh_copy = deepcopy(post_mesh)
    post_mesh_copy.vertices = o3d.utility.Vector3dVector((np.asarray(post_mesh_copy.vertices)[:] + np.array([0,30,0])))
    post_mesh.vertices = o3d.utility.Vector3dVector((np.asarray(post_mesh.vertices)[:] - np.array([0,30,0])))
    pre_mesh.vertices = o3d.utility.Vector3dVector((np.asarray(pre_mesh.vertices)[:] - np.array([0,30,0])))
    # pyvistarise
    pyv_pre_mesh = pyvistarise(pre_mesh)
    pyv_post_mesh = pyvistarise(post_mesh)
    pyv_post_mesh_copy = pyvistarise(post_mesh_copy)

    pyvista.global_theme.background = 'white'
    plotter = pyvista.Plotter()
    plotter.add_mesh(pyv_pre_mesh.extract_all_edges(), color='orange', line_width=2)
    plotter.add_mesh(pyv_post_mesh, color=[0.5, 0.706, 1.], opacity=0.5)
    plotter.add_mesh(pyv_post_mesh_copy, color=[0.5, 0.706, 1.], opacity=0.75)
    # add vertices with ground truth classes
    cmap = plt.cm.get_cmap('bwr')
    num_classes = classes.shape[1]
    for pc_idx in range(num_classes):
        if (pc_idx==np.argmax(classes, axis=1)).any():
            this_pc_verts = verts[pc_idx==np.argmax(classes, axis=1)]
            pc = pyvista.PolyData(this_pc_verts + np.array([0,30,0]))
            plotter.add_mesh(pc, render_points_as_spheres=True, color=cmap(pc_idx/(num_classes-1))[:3], opacity=1, point_size=15)
    plotter.store_image = True
    plotter.camera.position = (267.23733688822307, 185.79964888533257, -37.21093737148561)
    plotter.camera.focal_point = (182.16842651367188, 203.39258575439453, 303.5139617919922)
    plotter.camera.up = (0.9701952368133866, -0.0017313380567325239, 0.2423183957794992)
    camera_pos = plotter.show(screenshot="./QA_tool/visualisations/viz_perturbs.png",window_size=[1600, 1000], auto_close=True)
    print(camera_pos)

def main():
    def_num = "7"
    pat_num = "629"

    verts = torch.load(f"C:/PhD/ESTRO22/QA_tool_data/more_var_partial/graph_objects/GS/0522c0{pat_num}_R.pt").pos.numpy()
    faces = np.load(f"C:/PhD/ESTRO22/QA_tool_data/more_var_partial/triangles_smooth/GS/0522c0{pat_num}_R.npy")
    gs_mesh = o3d.geometry.TriangleMesh()
    gs_mesh.vertices = o3d.utility.Vector3dVector(verts)
    gs_mesh.triangles = o3d.utility.Vector3iVector(faces)

    verts = torch.load(f"C:/PhD/ESTRO22/QA_tool_data/more_var_partial/graph_objects/0522c0{pat_num}_R_{def_num}.pt").pos.numpy()
    faces = np.load(f"C:/PhD/ESTRO22/QA_tool_data/more_var_partial/triangles_smooth/0522c0{pat_num}_R_{def_num}.npy")
    perturbed_mesh = o3d.geometry.TriangleMesh()
    perturbed_mesh.vertices = o3d.utility.Vector3dVector(verts)
    perturbed_mesh.triangles = o3d.utility.Vector3iVector(faces)

    gs_mesh.compute_vertex_normals()
    perturbed_mesh.compute_vertex_normals()
    classes = torch.load(f"C:/PhD/ESTRO22/QA_tool_data/more_var_partial/signed_classes/0522c0{pat_num}_R_{def_num}.pt").numpy()
    visualise_pertubations(gs_mesh, perturbed_mesh, verts, classes)

if __name__ == '__main__':
    main()

