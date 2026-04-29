import meshio
import numpy as np

mesh = meshio.read('/home/fatima-zahranoussi/data/meshio/assets/input_code_aster.med')

# Inspecter la structure des polygones
for cells in mesh.cells:
    if cells.type == 'polygon':
        print("Tailles des polygones:")
        sizes = [len(c) for c in cells.data]
        print(f"  Min sommets: {min(sizes)}")
        print(f"  Max sommets: {max(sizes)}")
        print(f"  Distribution: {np.unique(sizes, return_counts=True)}")