import meshio
import h5py
import numpy as np

#mesh = meshio.read('/home/fatima-zahranoussi/data/meshio/assets/input_code_aster.med')
# Inspecter le fichier brut
with h5py.File('/home/fatima-zahranoussi/data/meshio/assets/input_code_aster.med', "r") as f:
    def print_tree(name, obj):
        print(name)
    f.visititems(print_tree)