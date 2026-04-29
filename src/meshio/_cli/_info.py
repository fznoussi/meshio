import numpy as np

from .._common import warn
from .._helpers import read, reader_map


def add_args(parser):
    parser.add_argument("infile", type=str, help="mesh file to be read from")
    parser.add_argument(
        "--input-format",
        "-i",
        type=str,
        choices=sorted(list(reader_map.keys())),
        help="input file format",
        default=None,
    )


def info(args):
    # read mesh data
    mesh = read(args.infile, file_format=args.input_format)
    print(mesh)

    # check if the cell arrays are consistent with the points
    # Vérifier les points NaN/Inf
    if np.any(np.isnan(mesh.points)):
        print("WARNING: Maillage contient des points NaN")
    if np.any(np.isinf(mesh.points)):
        print("WARNING: Maillage contient des points Inf")

    num_points = mesh.points.shape[0]
    for cells in mesh.cells:
        has_invalid = False
        try:
            data = np.asarray(cells.data)
            has_invalid = np.any(data >= num_points)
        except Exception:
            has_invalid = np.any(
                np.any(np.asarray(cells.data) >= num_points)
                for cells in cells.data
            )
        if has_invalid:
            print(
                f"WARNING: '{cells.type}' contient des indices "
                f">= nombre de points ({num_points})"
            )

    # Afficher un résumé du maillage
    print("\n── Résumé ──────────────────────────────")
    print(f"  Dimension     : {mesh.points.shape[1]}D")
    print(f"  Nb points     : {num_points}")
    for cells in mesh.cells:
        try:
            sizes = [len(c) for c in cells.data]
            print(
                f"  {cells.type:<12}: {len(cells.data)} cellules | "
                f"sommets/cellule : min={min(sizes)} max={max(sizes)}"
            )
        except TypeError:
            print(f"  {cells.type:<12}: {len(cells.data)} cellules")
    print("────────────────────────────────────────")
