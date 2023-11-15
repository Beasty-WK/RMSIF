import os
import shutil
from default_config.rna_opts import rna_opts
from input_output.protonate import protonate
import pymesh
from IPython.core.debugger import set_trace
from sklearn.neighbors import KDTree
from triangulation.fixmesh import fix_mesh
from triangulation.computeMSMS import computeMSMS
from triangulation.computeCharges import computeCharges, assignChargesToNewMesh
from triangulation.compute_normal import compute_normal
from triangulation.computer_theta import compute_theta
from input_output.save_ply import save_ply
from input_output.pdbqt_to_pdb import pdbqt_to_pdb
import numpy as np
import warnings

warnings.filterwarnings("ignore")


tmp_dir =  " " # pdbqt files dir
rna_pdb_dir = " " # pdb files dir
out_ply = " " # out ply files dir

pdb_ids = []
pdb_filename = []
DiSion = {}
RNA_pdb_names = os.listdir(rna_pdb_dir)

def standardization(data):
    mu = np.mean(data, axis=0)
    sigma = np.std(data, axis=0)
    return (data - mu) / sigma

def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range

for i, d in enumerate(RNA_pdb_names):
    d = d.split(".")
    d = d[0]
    pdb_ids.append(d)


faile_RNA = []

number = 0
for pdb_id in pdb_ids:
    pdb_filename = rna_pdb_dir + pdb_id + ".pdb"
    number = number + 1
    if number >= 1:
        if not os.path.exists(tmp_dir):
            os.makedirs(tmp_dir)


        protonated_file = tmp_dir+"/"+pdb_id+".pdbqt"


        # protonate(pdb_filename, protonated_file)   # 执行质子化操作

        pdbqt_filename = protonated_file



        try:
            vertices1, faces1, normals1, names1, areas1, Mg_coords = computeMSMS(protonated_file, protonate=True, Mg=None, pdb_id=pdb_id)
        except:
            faile_RNA.append(pdb_id)
            continue
            # set_trace()



        mesh = pymesh.form_mesh(vertices1, faces1)

        out_Ply_files = out_ply + pdb_id + "/" + pdb_id
        out_ply_file = out_ply + pdb_id
        if not os.path.exists(out_ply_file):
            os.makedirs(out_ply_file)


        # Fix the mesh
        regular_mesh = fix_mesh(mesh, rna_opts['mesh_res'])


        Mg_coords_new = Mg_coords
        Mg_coords = np.array(Mg_coords).reshape(1, -1)
        kdt = KDTree(Mg_coords)
        dist, r = kdt.query(regular_mesh.vertices)  # 各顶点到Mg离子的距离
        np.save(out_Ply_files + ".npy", dist.T[0])

        dists = normalization(dist).T[0]  # 归一化

        charge = computeCharges(names1)             #顶点电荷值

        vertex_charges = assignChargesToNewMesh(regular_mesh.vertices, vertices1, charge, rna_opts)

        assert (len(dist) == len(regular_mesh.vertices))
        value_interacte = sum(dist) / len(dist)
        # value_interacte = float(DiSion_MG_RNA) + 4.0

        iface = np.zeros(len(regular_mesh.vertices))
        iface_v = np.where(dist <= value_interacte)[0]  
        iface[iface_v] = 1.0



        # Compute the normals
        vertex_normal = compute_normal(regular_mesh.vertices, regular_mesh.faces)
        # Compute the theta
        vertex_theta = compute_theta(regular_mesh.vertices, vertex_normal, Mg_coords_new)


        save_ply(out_Ply_files + ".ply", regular_mesh.vertices, \
                 regular_mesh.faces, normals=vertex_normal, iface=iface, charges=vertex_charges, dists=dists, vertex_theta=vertex_theta)


np.save("MGbind_neg.npy", faile_RNA)







