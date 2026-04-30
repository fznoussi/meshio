
import numpy as np
from collections import Counter
from .._common import num_nodes_per_cell
from .._exceptions import WriteError, ReadError
from .._mesh import Mesh
from ._med41 import FieldBitmaskWriter
from ._med import (
    meshio_to_med_type,
    med_to_geo_type,
    med_to_meshio_type,
    numpy_to_med_type,
    numpy_void_str,
    _write_families,
    _write_data,
    _read_data,
    _read_families,
)

def _resolve_mesh_names(meshes, mesh_names=None):
    "Recoit une liste de Mesh et une liste de noms, et retourne une liste de noms pour chaque Mesh."

    if mesh_names is None:
        mesh_names = [f"mesh_{i}" for i in range(len(meshes))] 
    if len (mesh_names) <= len(meshes): 
        " vérification des doublons "
        seen = {}
        resolved = []
        for name in mesh_names:
            if name in seen:
                seen[name] += 1
                name = f"{name}_{seen[name]}"
                resolved.append(name)
                
            else:
                seen[name] = 0
                resolved.append(name)
                seen[name] = 1
        mesh_names = resolved
        " on prend les noms qu'on a et remplit le reste avec des noms par défaut"
        if len(mesh_names) < len(meshes):
           mesh_names = mesh_names + [f"mesh_{i}" for i in range(len(mesh_names), len(meshes))]
    elif len(mesh_names) > len(meshes):
        raise WriteError(f"Plus de noms de maillage ({len(mesh_names)}) que de maillages ({len(meshes)})")
    return mesh_names


def _find_field_collisions(meshes):
    " Détecter les noms de champs qui apparaissent dans plusieurs maillages "
    all_names = []
    for mesh in meshes:
        for name in mesh.point_data.keys():
            if name != "point_tags":
                all_names.append(name)
        for name in mesh.cell_data.keys():
            if name not in {"cell_tags","gmsh:physical"}:
             all_names.append(name)

    counts = Counter(all_names)
    collisions = {name for name, count in counts.items() if count > 1}
    return collisions

def write_med_multi(filename, meshes, mesh_names=None, med_version ="4.1.0",  **kwargs):
    import h5py
    "On commence par la validation de entrées"
    if meshes is None or len(meshes) == 0:
        raise WriteError("Aucun maillage à écrire.")
    
    if not isinstance(meshes, list):
        raise WriteError("Les maillages doivent être fournis sous forme de liste.")
    
    " résoudre les noms avec _resolve_mesh_names, détecter les collisions avec _find_field_collisions et ouvrir le fichier HDF5 avec h5py"
    try:
        version_parts = [int(x) for x in med_version.split(".")]
        maj = version_parts[0]
        minor = version_parts[1] if len(version_parts) > 1 else 0
        rel = version_parts[2] if len(version_parts) > 2 else 0
    except ValueError:
        maj, minor, rel = 4, 1, 0

    f = h5py.File(filename, "w")

    info = f.create_group("INFOS_GENERALES")
    info.attrs.create("MAJ", maj)
    info.attrs.create("MIN", minor)
    info.attrs.create("REL", rel)

    # Meshes
    mesh_names = _resolve_mesh_names(meshes, mesh_names)
    mesh_ensemble = f.create_group("ENS_MAA")  # Ensemble de maillages, chaque maillage est un groupe dans ce groupe
    for mesh, name in zip(meshes, mesh_names):
            med_mesh = mesh_ensemble.create_group(name)
            med_mesh.attrs.create("DIM", mesh.points.shape[1])  # mesh dimension
            med_mesh.attrs.create("ESP", mesh.points.shape[1])  # spatial dimension
            med_mesh.attrs.create("REP", 0)  # cartesian coordinate system (repère in French)
            med_mesh.attrs.create("UNT", numpy_void_str)  # time unit
            med_mesh.attrs.create("UNI", numpy_void_str)  # spatial unit
            med_mesh.attrs.create("SRT", 1)  # sorting type MED_SORT_ITDT
            # component names:
            axis_names = ["X", "Y", "Z"][: mesh.points.shape[1]]
            med_mesh.attrs.create("NOM", np.bytes_("".join(f"{name:<16}" for name in axis_names)))
            med_mesh.attrs.create("DES", np.bytes_("Mesh created with meshio"))
            med_mesh.attrs.create("TYP", 0)  # mesh type (MED_NON_STRUCTURE)

            # Time-step
            step = "-0000000000000000001-0000000000000000001"  # NDT NOR
            time_step = med_mesh.create_group(step)
            time_step.attrs.create("CGT", 1)
            time_step.attrs.create("NDT", -1)  # no time step (-1)
            time_step.attrs.create("NOR", -1)  # no iteration step (-1)
            time_step.attrs.create("PDT", -1.0)  # current time

            # Points
            nodes_group = time_step.create_group("NOE")
            nodes_group.attrs.create("CGT", 1)
            nodes_group.attrs.create("CGS", 1)
            profile = "MED_NO_PROFILE_INTERNAL"
            nodes_group.attrs.create("PFL", np.bytes_(profile))
            coo = nodes_group.create_dataset("COO", data=mesh.points.flatten(order="F"))
            coo.attrs.create("CGT", 1)
            coo.attrs.create("NBR", len(mesh.points))

            # Point tags
            if "point_tags" in mesh.point_data:  # only works for med -> med
                family = nodes_group.create_dataset("FAM", data=mesh.point_data["point_tags"])
                family.attrs.create("CGT", 1)
                family.attrs.create("NBR", len(mesh.points))

            # Cells (mailles in French)
            cells_by_type = {}
            cell_tags_by_type = {}

            for k, cell_block in enumerate(mesh.cells):
                cell_type = cell_block.type
                if cell_type not in cells_by_type:
                    cells_by_type[cell_type] = []
                    cell_tags_by_type[cell_type] = []
                cells_by_type[cell_type].append(cell_block.data)
                if "cell_tags" in mesh.cell_data:
                    cell_tags_by_type[cell_type].append(mesh.cell_data["cell_tags"][k])
            cells_group = time_step.create_group("MAI")
            cells_group.attrs.create("CGT", 1)
            for cell_type, cells_list in cells_by_type.items():
            # fusion des cellules
                merged_cells = np.concatenate(cells_list, axis=0)
                med_type = meshio_to_med_type[cell_type]
                med_cells = cells_group.create_group(med_type)
                med_cells.attrs.create("CGT", 1)
                med_cells.attrs.create("CGS", 1)
                med_cells.attrs.create("PFL", np.bytes_(profile))
                nod = med_cells.create_dataset("NOD", data=merged_cells.flatten(order="F") + 1)
                nod.attrs.create("CGT", 1)
                nod.attrs.create("NBR", len(merged_cells))


            # Cell tags
                if "cell_tags" in mesh.cell_data and cell_tags_by_type[cell_type]:
                    merged_tags = np.concatenate(cell_tags_by_type[cell_type], axis=0)
                    family = med_cells.create_dataset("FAM", data=merged_tags)
                    family.attrs.create("CGT", 1)
                    family.attrs.create("NBR", len(merged_cells))

    # Information about point and cell sets (familles in French)
    fas = f.create_group("FAS")
    for mesh, name in zip(meshes, mesh_names):
        families = fas.create_group(name)
        family_zero = families.create_group("FAMILLE_ZERO")  # must be defined in any case
        family_zero.attrs.create("NUM", 0)

        # For point tags
        try:
            if len(mesh.point_tags) > 0:
                node = families.create_group("NOEUD")
                _write_families(node, mesh.point_tags)
        except AttributeError:
            pass

                # For cell tags
        try:
            if len(mesh.cell_tags) > 0:
                element = families.create_group("ELEME")
                _write_families(element, mesh.cell_tags)
        except AttributeError:
            pass
        
    # Champs
    fields = f.create_group("CHA")
    collisions = _find_field_collisions(meshes)
    profile = "MED_NO_PROFILE_INTERNAL"

    for mesh, name in zip(meshes, mesh_names):
        tracker = FieldBitmaskWriter()

        # Champs nodaux
        for field_name, data in mesh.point_data.items():
            if field_name == "point_tags":  
                continue
            print(f"field: {field_name}, dtype: {data.dtype}, shape: {data.shape}")
            hdf5_name = f"{field_name}@{name}" if field_name in collisions else field_name
            tracker.notify(
                "MED_NODE",
                "MED_POINT1",
                "0000000000000000000100000000000000000001"
            )
            _write_data(
                fields, name, None, profile,
                hdf5_name, "NOEU", data,
                tracker=tracker
            )

        # Champs cellulaires
        for field_name, data_list in mesh.cell_data.items():
            if field_name in ("cell_tags", "gmsh:physical"):
                continue
            hdf5_name = f"{field_name}@{name}" if field_name in collisions else field_name

            # Regroupement par type de cellule
            data_by_type = {}
            for cell, data in zip(mesh.cells, data_list):
                cell_type = cell.type
                if cell_type not in data_by_type:
                    data_by_type[cell_type] = []
                data_by_type[cell_type].append(data)

            for cell_type, blocks in data_by_type.items():
                merged_data = np.concatenate(blocks, axis=0)
                med_type = meshio_to_med_type[cell_type]
                # ignorer les données invalides (NaN placeholders ou dtype object)
                if merged_data.dtype == object:
                    continue
                if merged_data.ndim <= 2:
                    supp = "ELEM"
                elif merged_data.shape[1] == num_nodes_per_cell[cell_type]:
                    supp = "ELNO"
                else:
                    supp = "ELGA"

                _write_data(
                    fields, name, None, profile,
                    hdf5_name, supp, merged_data,
                    med_type, tracker=tracker
                )

        # Flush tracker après chaque maillage
        for fn in fields.keys():
            tracker.flush(fields[fn])

def read_med_multi(filename, **kwargs):
    "Lecture d'un fichier MED multi-maillage. Retourne une liste de Mesh et une liste de noms de maillage."
    import h5py
    f = h5py.File(filename, "r")

    # Ensemble de maillages 
    " On lit les nom des maillagges dans ENS_MAA et pour chaque maillage on stoque le nom dans meshes_names "
    meshes_name = list(f["ENS_MAA"].keys())
    meshes = []
    for name in meshes_name:
        mesh = _read_single_mesh(f, name)
        meshes.append(mesh)
    return meshes, meshes_name

def _read_single_mesh(f, name):
    "Lit un maillage individuel dans le groupe ENS_MAA et retourne un objet Mesh."
    meshes_grp = f["ENS_MAA"][name]
    dim = meshes_grp.attrs["ESP"]

    if "NOE" not in meshes_grp:
        time_step = list(meshes_grp.keys())
        meshes_grp = meshes_grp[time_step[0]]

    # Initialize data
    point_data = {}
    cell_data = {}
    field_data = {}

    # Points
    pts_dataset = meshes_grp["NOE"]["COO"]
    n_points = pts_dataset.attrs["NBR"]
    points = pts_dataset[()].reshape((n_points, dim), order="F")

    # Point tags
    if "FAM" in meshes_grp["NOE"]:
        tags = meshes_grp["NOE"]["FAM"][()]
        point_data["point_tags"] = tags

    # Information for point tags
    point_tags = {}
    if "FAS" in meshes_grp:
        fas = meshes_grp["FAS"]
    elif "FAS" in f and name in f["FAS"]:
        fas = f["FAS"][name]
    else:
        fas = None

    if fas is not None and "NOEUD" in fas:
        point_tags = _read_families(fas["NOEUD"])

    # CellBlock
    cells = []
    cell_types = []
    med_cells = meshes_grp["MAI"]
    for med_cell_type, med_cell_type_group in med_cells.items():
        cell_type = med_to_meshio_type[med_cell_type]
        cell_types.append(cell_type)
        if med_cell_type in ["POG", "POG2"]:
            nod = med_cell_type_group["NOD"][()] - 1
            inn = med_cell_type_group["INN"][()]
            polygons = [
                nod[inn[i] - 1: inn[i + 1] - 1] for i in range(len(inn) - 1)
            ]
            cells.append((cell_type, polygons))
        else:
            nod = med_cell_type_group["NOD"]
            n_cells = nod.attrs["NBR"]
            cells += [(cell_type, nod[()].reshape(n_cells, -1, order="F") - 1)]

        if "FAM" in med_cell_type_group:
            tags = med_cell_type_group["FAM"][()]
            if "cell_tags" not in cell_data:
                cell_data["cell_tags"] = []
            cell_data["cell_tags"].append(tags)

    # Information for cell tags
    cell_tags = {}
    if fas is not None and "ELEME" in fas:
        cell_tags = _read_families(fas["ELEME"])

    # Read fields if they exist
    if "CHA" in f:
        profiles = f["PROFILS"] if "PROFILS" in f else None
        for field_name, field_grp in f["CHA"].items():
            # filtrage par maillage
            if "@" in field_name:
                logical_name, owner = field_name.rsplit("@", 1)
                if owner != name:
                    continue
            else:
                mai = field_grp.attrs.get("MAI", b"").decode().strip("\x00")
                if mai and mai != name:
                    continue
                logical_name = field_name

            # lecture du champ via _read_data sur ce champ uniquement
            _read_data(
                {logical_name: field_grp},
                profiles,
                cell_types,
                point_data,
                cell_data,
                field_data,
            )

    # Construct the mesh object
    result = Mesh(
        points, cells,
        point_data=point_data,
        cell_data=cell_data,
        field_data=field_data,
    )
    result.point_tags = point_tags
    result.cell_tags = cell_tags
    return result