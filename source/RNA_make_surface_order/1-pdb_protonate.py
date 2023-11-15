import os
import shutil
from default_config.rna_opts import rna_opts
from input_output.protonate import protonate
import pymesh
from IPython.core.debugger import set_trace
from sklearn.neighbors import KDTree
from triangulation.fixmesh import fix_mesh
from triangulation.computeMSMS import computeMSMS
from triangulation.compute_normal import compute_normal
from input_output.save_ply import save_ply
from input_output.pdbqt_to_pdb import pdbqt_to_pdb
import numpy as np
from default_config.read_RNA_pdb import pdb_read_file
from default_config.replace_rna_chain import DISion_rna


tmp_dir = ""  # pdbqt files dir
rna_pdb_dir = "" # pdb files dir
pdb_ids = []
pdb_filename = []

RNA_pdb_names = os.listdir(rna_pdb_dir)

for i, d in enumerate(RNA_pdb_names):
    d = d.split(".")
    d = d[0]
    pdb_ids.append(d)
error_list = []
number = 0
for pdb_id in pdb_ids:
    pdb_filename = rna_pdb_dir + pdb_id + ".pdb"
    number = number + 1
    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir)

    protonated_file = tmp_dir+"/"+pdb_id+".pdbqt"

    try:
        protonate(pdb_filename, protonated_file)  # Unable Mg atom.
    except:
        error_list.append(pdb_id)
        continue


np.save("error_pdbs.npy", error_list)






