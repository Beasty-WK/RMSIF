import sys
import time
import os
import numpy as np
from IPython.core.debugger import set_trace

from default_config.rna_opts import rna_opts
import warnings
warnings.filterwarnings("ignore")


# load training data (From many files)
from rna_modules.read_data_from_surface import read_data_from_surface

from sklearn import metrics


params = rna_opts['site']
root_dir = "source/data_preparations/"
input_ply = " " # Ply files dir
decoy_data = " " # pdb files dir

rna_pdb_dir = " " # pdb files dir


RNA_pdb_names = os.listdir(rna_pdb_dir)
pdb_ids = []
for i, d in enumerate(RNA_pdb_names):
    d = d.split(".")
    d = d[0]
    pdb_ids.append(d)

faile_RNA = []
number = 0
for pdb_id in pdb_ids:
    number = number + 1
    if number >= 1:
        print(pdb_id)
        # print(pdb_id)
        all_list_desc = []
        all_list_coords = []
        all_list_shape_idx = []
        all_list_name = []
        idx_positives = []

        my_precomp_dir = 'data_preparation/data/' + decoy_data + pdb_id + '/'
        if not os.path.exists(my_precomp_dir):
            os.makedirs(my_precomp_dir)

        # Read directly from the ply file
        fields = pdb_id
        ply_file = {}
        in_ply_file = input_ply + pdb_id + "/" + pdb_id + ".ply"
        ply_file['p1'] = in_ply_file.format(fields[0:])


        pids = ['p1']

        # Compute shape
        rho = {}
        neigh_indices = {}
        mask = {}
        input_feat = {}
        theta = {}
        iface_lables = {}
        verts = {}
        # Compute the angular and radial coordinates--------rho, theta, neigh_indices, mask---
        try:
            for pid in pids:
                input_feat[pid], rho[pid], theta[pid], mask[pid], neigh_indices[pid], \
                    iface_lables[pid], verts[pid] = read_data_from_surface(root_dir + ply_file[pid], params)
             # Save data only if everything went well
            for pid in pids:
                np.save(my_precomp_dir + pid + '_rho_wrt_center', rho[pid])
                np.save(my_precomp_dir + pid + '_theta_wrt_center', theta[pid])
                np.save(my_precomp_dir + pid + '_input_feat', input_feat[pid])
                np.save(my_precomp_dir + pid + '_mask', mask[pid])
                np.save(my_precomp_dir + pid + '_list_indices', neigh_indices[pid])
                np.save(my_precomp_dir + pid + '_iface_labels', iface_lables[pid])
        except:
            faile_RNA.append(pdb_id)
            continue
np.save("MGbind_neg_1_RNA.npy", faile_RNA)

