
# benchmark_all_formats.py
import meshio
import time
import os
import numpy as np

def generer_mesh_test(n_points=10_000):
    """Mesh réaliste pour benchmark"""
    np.random.seed(42)
    points = np.random.rand(n_points, 3)

    # Triangles
    n_tri = n_points - 2
    triangles = np.array([
        [i, i+1, i+2] for i in range(n_tri)
    ])

    cells = [("triangle", triangles)]
    return meshio.Mesh(points=points, cells=cells)

FORMATS = ["vtk", "vtu", "stl", "msh", "xdmf", "inp"]
TAILLES = [100, 1_000, 10_000, 100_000]

for n in TAILLES:
    mesh = generer_mesh_test(n)
    print(f"\n{'='*65}")
    print(f"  MESH : {n:,} points")
    print(f"{'='*65}")
    print(f"{'Format':<10} {'Écriture':>12} {'Lecture':>12} {'Taille':>10}")
    print(f"{'-'*65}")

    for fmt in FORMATS:
        fichier = f"/tmp/test_{n}.{fmt}"
        try:
            t0 = time.perf_counter()
            meshio.write(fichier, mesh)
            t1 = time.perf_counter()
            temps_ecriture = (t1 - t0) * 1000

            t2 = time.perf_counter()
            meshio.read(fichier)
            t3 = time.perf_counter()
            temps_lecture = (t3 - t2) * 1000

            taille = os.path.getsize(fichier) / 1024

            print(f"{fmt:<10} {temps_ecriture:>10.2f}ms "
                  f"{temps_lecture:>10.2f}ms "
                  f"{taille:>8.1f}KB")

        except Exception as e:
            print(f"{fmt:<10} {'ERREUR':>12} : {e}")















'''

import meshio
import h5py
import numpy as np

#mesh = meshio.read('/home/fatima-zahranoussi/data/meshio/assets/input_code_aster.med')
# Inspecter le fichier brut
with h5py.File('/home/fatima-zahranoussi/data/meshio/assets/input_code_aster.med', "r") as f:
    def print_tree(name, obj):
        print(name)
    f.visititems(print_tree)
    '''