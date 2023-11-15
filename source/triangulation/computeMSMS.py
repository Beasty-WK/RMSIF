import os
import sys
from subprocess import Popen, PIPE
from input_output.read_msms import read_msms
from triangulation.xyzrn_new import output_pdb_as_xyzrn
from default_config.rna_opts import rna_opts
import random

# Calls MSMS and returns the vertices.
# Special atoms are atoms with a reduced radius.

def computeMSMS(pdb_file,  protonate=True, Mg = None, pdb_id=None):
    randnum = random.randint(1, 10000000)
    file_base = rna_opts['temp_mid']+"/msms_"+ pdb_id #str(randnum)
    out_xyzrn = file_base+".xyzrn"

    if protonate:
        Mg_coords = output_pdb_as_xyzrn(pdb_file, out_xyzrn, Mg)
    else:
        print("Error - pdb2xyzrn is deprecated")
        sys.exit(1)

    # now run MSMS on xyzrn file
    cmd = "msms -density 3.0 -hdensity 3.0 -probe 1.5 -if " + out_xyzrn + " -of " + file_base + " -af " + file_base
    os.system(cmd)

    vertics, faces, normals, names = read_msms(file_base)
    areas = {}
    ses_file = open(file_base+".area")
    next(ses_file)  # ignore header line
    for line in ses_file:
        fields = line.split()
        areas[fields[3]] = fields[1]

    # Remove temporary files.
    os.remove(file_base+'.area')
    os.remove(file_base + '.xyzrn')
    os.remove(file_base + '.vert')
    os.remove(file_base + '.face')
    return vertics, faces, normals, names, areas, Mg_coords




