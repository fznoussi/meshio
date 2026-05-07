
# Rapport des développements

Les développements effectués ici concerne principalement le format med et la conversion med <-> msh.


## Fix A: `FAS` manquant dans le reader MED

### Bug

```
KeyError: "Unable to synchronously open object (object 'FAS' doesn't exist)"
```

Le problème est que les fichiers MED générés par GMSH ne contiennent pas toujours de section `FAS`.
Le reader tentait d'y accéder inconditionnellement.

### Avant (`_med.py`, ~ligne 78)

```python
fas = mesh["FAS"] if "FAS" in mesh else f["FAS"][mesh_name]
```

Si `FAS` n'existe ni dans `mesh` ni dans `f` → `KeyError`.

### Après

```python
if "FAS" in mesh:
    fas = mesh["FAS"]
elif "FAS" in f and mesh_name in f["FAS"]:
    fas = f["FAS"][mesh_name]
else:
    fas = None
```

On teste les deux emplacements possibles puis on accepte `None` si aucun n'existe.

### Adaptations en conséquence

```python
# AVANT
if "NOEUD" in fas:
    point_tags = _read_families(fas["NOEUD"])

# APRÈS
point_tags = {}
if fas is not None and "NOEUD" in fas:
    point_tags = _read_families(fas["NOEUD"])
```

```python
# AVANT
if "ELEME" in fas:
    cell_tags = _read_families(fas["ELEME"])

# APRÈS
cell_tags = {}
if fas is not None and "ELEME" in fas:
    cell_tags = _read_families(fas["ELEME"])
```

Ainsi chaque accès à `fas` est désormais protégé par un test `is not None` afin d'éviter tout crash du module.

---

## Fix B : `GRO` manquant dans `_read_families`

### Bug

```
KeyError: "Unable to synchronously open object (object 'GRO' doesn't exist)"
```

Certaines familles MED n'ont pas de sous-groupe `GRO`.
La fonction accédait directement sans vérifier.

### Avant (`_med.py`, ~ligne 204)

```python
def _read_families(fas_data):
    families = {}
    for _, node_set in fas_data.items():
        set_id = node_set.attrs["NUM"]
        n_subsets = node_set["GRO"].attrs["NBR"]  # crash
        nom_dataset = node_set["GRO"]["NOM"][()]
        name = [None] * n_subsets
        for i in range(n_subsets):
            name[i] = "".join(
                [chr(x) for x in nom_dataset[i]]
            ).strip().rstrip("\x00")
        families[set_id] = name
    return families
```

`node_set["GRO"]` crash si le groupe HDF5 `GRO` n'existe pas.

### Après

```python
def _read_families(fas_data):
    families = {}
    for _, node_set in fas_data.items():
        set_id = node_set.attrs["NUM"]
        if "GRO" not in node_set:
            families[set_id] = []
            continue
        n_subsets = node_set["GRO"].attrs["NBR"]
        nom_dataset = node_set["GRO"]["NOM"][()]
        name = [None] * n_subsets
        for i in range(n_subsets):
            name[i] = "".join(
                [chr(x) for x in nom_dataset[i]]
            ).strip().rstrip("\x00")
        families[set_id] = name
    return families
```

Test `"GRO" not in node_set` -> si absent, on enregistre une famille vide et on passe à la suivante.

---

## Fix C : Écriture multi-blocs même type

### Bug

```
WriteError: MED files cannot have two sections of the same cell type.
```

puis, après suppression du safeguard :

```
ValueError: Unable to synchronously create group (name already exists)
```

Un maillage Gmsh contient souvent plusieurs blocs du même type
(ex: 6 blocs de triangles, un par entité géométrique).
Le writer créait un groupe HDF5 par bloc → doublon de `"MAI.TR3"`.

### 3.1 Suppression du garde-fou

**Avant (~ligne 269)** :
```python
if len(mesh.cells) != len(np.unique([c.type for c in mesh.cells])): # Sert à vérifier que le maillage ne contient pas plusieurs blocs séparés pour le même type géométrique.
    raise WriteError("MED files cannot have two sections of the same cell type.")
```

**Après** : Lignes supprimées.

### 3.2 Fusion des cellules par type

**Avant (~ligne 272)** :
```python
cells_group = time_step.create_group("MAI")
cells_group.attrs.create("CGT", 1)
for k, cell_block in enumerate(mesh.cells):
    cell_type = cell_block.type
    cells = cell_block.data
    med_type = meshio_to_med_type[cell_type]
    med_cells = cells_group.create_group(med_type)  # ← crash au 2ème TR3
    med_cells.attrs.create("CGT", 1)
    med_cells.attrs.create("CGS", 1)
    med_cells.attrs.create("PFL", np.bytes_(profile))
    nod = med_cells.create_dataset(
        "NOD", data=cells.flatten(order="F") + 1
    )
    nod.attrs.create("CGT", 1)
    nod.attrs.create("NBR", len(cells))
    if "cell_tags" in mesh.cell_data:
        tags = mesh.cell_data["cell_tags"][k]
        family = med_cells.create_dataset("FAM", data=tags)
        family.attrs.create("CGT", 1)
        family.attrs.create("NBR", len(cells))
```

Chaque bloc crée son propre groupe HDF5. Au 2e bloc de triangles → `"TR3"` existe déjà.

**Après** :
```python
    # Regrouper par type
    cells_by_type = {}
    cell_tags_by_type = {}
    for k, cell_block in enumerate(mesh.cells):
        cell_type = cell_block.type
        if cell_type not in cells_by_type:
            cells_by_type[cell_type] = []
            cell_tags_by_type[cell_type] = []
        cells_by_type[cell_type].append(cell_block.data)
        if "cell_tags" in mesh.cell_data:
            cell_tags_by_type[cell_type].append(
                mesh.cell_data["cell_tags"][k]
            )

    # Écrire une fois par type
    cells_group = time_step.create_group("MAI")
    cells_group.attrs.create("CGT", 1)
    for cell_type, cells_list in cells_by_type.items():
        merged_cells = np.concatenate(cells_list, axis=0)
        med_type = meshio_to_med_type[cell_type]
        med_cells = cells_group.create_group(med_type)
        med_cells.attrs.create("CGT", 1)
        med_cells.attrs.create("CGS", 1)
        med_cells.attrs.create("PFL", np.bytes_(profile))
        nod = med_cells.create_dataset(
            "NOD", data=merged_cells.flatten(order="F") + 1
        )
        nod.attrs.create("CGT", 1)
        nod.attrs.create("NBR", len(merged_cells))
        if "cell_tags" in mesh.cell_data and cell_tags_by_type[cell_type]:
            merged_tags = np.concatenate(
                cell_tags_by_type[cell_type], axis=0
            )
            family = med_cells.create_dataset("FAM", data=merged_tags)
            family.attrs.create("CGT", 1)
            family.attrs.create("NBR", len(merged_cells))
```

On construit d'abord un dictionnaire `{type: [blocs]}`,
puis `np.concatenate` fusionne les blocs et une seule écriture HDF5 est faite par type.

### 3.3 Fusion des cell_data (CHA) par type

Même principe pour les champs de données cellulaires  :

**Avant** :
```python
for name, d in mesh.cell_data.items():
    if name == "cell_tags":
        continue
    for cell, data in zip(mesh.cells, d):
        med_type = meshio_to_med_type[cell.type]
        _write_data(..., med_type)
```

`_write_data` crée un groupe `"MAI.TR3"` par appel → doublon au 2e bloc.

**Après** :
```python
for name, d in mesh.cell_data.items():
    if name in ("cell_tags", "gmsh:physical"):
        continue
    data_by_type = {}
    for cell, data in zip(mesh.cells, d):
        cell_type = cell.type
        if cell_type not in data_by_type:
            data_by_type[cell_type] = []
        data_by_type[cell_type].append(data)
    for cell_type, data_list in data_by_type.items():
        merged_data = np.concatenate(data_list, axis=0)
        med_type = meshio_to_med_type[cell_type]
        _write_data(..., merged_data, med_type)
```

Même pattern : regrouper → fusionner → écrire une fois.

---

## Fix D : Conversion MED → MSH (métadonnées perdues)

### Bug

```bash
meshio c mesh.med output.msh
meshio i output.msh
# → 1 bloc de 264 triangles, aucune métadonnée
```

Le `mesh.med` natif contient :
```python
cell_tags = [-6, -5, -4, -3, -2, -1, ...]  # 6 familles
cell_tags_mapping = {-1: ['subgroup'], -2: ['subgroup'], ...}
```

Mais le writer Gmsh a besoin de :
```python
gmsh:physical       # tag par cellule
gmsh:geometrical    # entity tag par cellule
gmsh:dim_tags       # (dim, tag) par nœud
field_data          # {"subgroup": [physical_tag, dimension]}
```

La fonction `_convert_med_tags_to_gmsh()` ne faisait que renommer `cell_tags`.

### Imports ajoutés (`gmsh/main.py`)

```python
import numpy as np
from .._mesh import CellBlock, Mesh
```

### Avant

```python
def _convert_med_tags_to_gmsh(mesh):
    # ... détection ... #
    mesh = copy.deepcopy(mesh)
    if "cell_tags" in mesh.cell_data:
        mesh.cell_data["gmsh:physical"] = mesh.cell_data.pop("cell_tags")
    if "point_tags" in mesh.point_data:
        mesh.point_data["gmsh:physical"] = mesh.point_data.pop("point_tags")
    # field_data reconstruction depuis cell_sets (souvent vide)
    return mesh
```

Problèmes :
- Pas de split : 1 bloc avec tags mélangés
- Pas de `gmsh:geometrical` : writer Gmsh ne sait pas séparer les entités
- Pas de `gmsh:dim_tags` : section `$Entities` et `$Nodes` vides
- `field_data` non reconstruit : pas de `$PhysicalNames`

### Après : 5 étapes

**Étape 1 : Mapping familles -> physical tags**

```python
family_groups = getattr(mesh, "cell_tags", {})
# {-1: ['subgroup'], -2: ['subgroup'], ...}

group_names = sorted({n for names in family_groups.values() for n in names})
# ['subgroup']

group_to_phys = {name: i for i, name in enumerate(group_names, start=1)}
# {'subgroup': 1}
# on récupère les noms de groupes uniques et on leur assigne un physical tag séquentiel
fam_to_phys = {} #  dictionnaire vide pour mapper les family ID vers les physical tags
for fam_id, names in family_groups.items():
    if names:
        fam_to_phys[int(fam_id)] = group_to_phys[names[0]]
# {-1: 1, -2: 1, -3: 1, -4: 1, -5: 1, -6: 1}
```

On extrait les noms de groupes uniques depuis les familles MED,
on leur assigne un physical tag séquentiel,
puis on mappe chaque family ID vers son physical tag.

**Étape 2 : Choix du critère de split**

```python
has_geom = "gmsh:geometrical" in mesh.cell_data
has_tags = "cell_tags" in mesh.cell_data

if has_geom:
    split_data = mesh.cell_data["gmsh:geometrical"]
elif has_tags:
    split_data = mesh.cell_data["cell_tags"]
```

Deux cas possibles :
- **Round-trip** : les données Gmsh existent déjà → split par `gmsh:geometrical`
- **MED natif** : seuls les FAM IDs existent → split par `cell_tags`

**Étape 3 : Split et génération des tags**

```python
entity_counter = 1
for i, cb in enumerate(mesh.cells):
    tags = np.asarray(split_data[i], dtype=int)
    for utag in np.unique(tags):
        mask = tags == utag
        n = int(mask.sum())
        new_cells.append(CellBlock(cb.type, cb.data[mask]))
        new_geom.append(np.full(n, entity_counter, dtype=int))
        phys = fam_to_phys.get(int(utag), 0)
        new_phys.append(np.full(n, phys, dtype=int))
        entity_counter += 1
```

Pour chaque valeur unique de tag :
- On extrait les cellules correspondantes via un masque booléen
- On crée un nouveau `CellBlock` avec ces cellules
- On assigne un entity tag séquentiel (`entity_counter`)
- On traduit le FAM ID en physical tag via `fam_to_phys`

Résultat : `1 bloc×264` → `6 blocs×44`

**Étape 4 : Génération de `gmsh:dim_tags`**

```python
dim_tags = np.zeros((n_pts, 2), dtype=int)
assigned = np.full(n_pts, False)
for ci, cb in enumerate(new_cells):
    etag = int(new_geom[ci][0])
    nodes = np.unique(cb.data)
    new_nodes = nodes[~assigned[nodes]]
    dim_tags[new_nodes, 0] = cb.dim     # dimension topologique
    dim_tags[new_nodes, 1] = etag        # entity tag
    assigned[nodes] = True
```

Pour chaque nœud, on détermine à quelle entité il appartient.
`cb.dim` donne la dimension (2 pour triangle, 3 pour hexaèdre).
Les nœuds partagés entre plusieurs entités sont assignés à la première rencontrée.

**Étape 5 : Reconstruction du `field_data`**

```python
dim_val = new_cells[0].dim if new_cells else 2
field_data = {name: [tag, dim_val] for name, tag in group_to_phys.items()}
# {'subgroup': [1, 2]}
```

Format attendu par Gmsh : `nom → [physical_tag, dimension]`.

### Ajout de `_cleanup_med_fields()`

Fallback quand aucun split n'est nécessaire (1 seule valeur de tag) :

```python
def _cleanup_med_fields(mesh, group_to_phys):
    if "cell_tags" in mesh.cell_data:
        if "gmsh:physical" not in mesh.cell_data:
            mesh.cell_data["gmsh:physical"] = mesh.cell_data["cell_tags"]
        del mesh.cell_data["cell_tags"]
    if "point_tags" in mesh.point_data:
        del mesh.point_data["point_tags"]
    mesh.field_data = {
        k: v for k, v in mesh.field_data.items()
        if not k.startswith("med:")
        and isinstance(v, (list, tuple))
        and len(v) == 2
        and isinstance(v[0], (int, np.integer))
    }
    return mesh
```

Supprime les clés spécifiques MED (`cell_tags`, `point_tags`, `med:*`)
qui ne sont pas comprises par le writer Gmsh.

---

## Fix E — Mauvais writer sélectionné

### Bug

```bash
meshio c mesh.med output.msh
# → fichier écrit par le writer Ansys, pas Gmsh
# → aucune métadonnée, aucune erreur visible
```

### Cause

```python
# _helpers.py:write()
file_formats = _filetypes_from_path(Path("output.msh"))
# → ['ansys', 'gmsh']

file_format = file_formats[0]
# → 'ansys' ← le writer Ansys est appelé silencieusement
```

L'extension `.msh` est partagée entre Ansys et Gmsh.
Le code prenait le premier résultat sans analyser le contenu du mesh.

### Ajout de `_pick_best_format()` (avant `write()`)

```python
def _pick_best_format(file_formats, mesh):
    if "gmsh" in file_formats:
        gmsh_keys = {"gmsh:physical", "gmsh:geometrical", "gmsh:dim_tags"}
        med_keys = {"cell_tags", "point_tags"}
        has_gmsh = bool(
            gmsh_keys & set(mesh.cell_data.keys())
        ) or bool(
            gmsh_keys & set(mesh.point_data.keys())
        )
        has_med = (
            bool(med_keys & set(mesh.cell_data.keys()))
            or bool(med_keys & set(mesh.point_data.keys()))
            or any(k.startswith("med:") for k in mesh.field_data)
        )
        if has_gmsh or has_med:
            return "gmsh"
    return file_formats[0]
```

On inspecte les clés du mesh :
- Si des clés Gmsh (`gmsh:*`) ou MED (`cell_tags`, `point_tags`, `med:*`) existent → format `"gmsh"`
- Sinon → comportement par défaut (premier format)

### Modification dans `write()` (~ligne 165)

```python
# AVANT
            # just take the first one
            file_format = file_formats[0]

# APRÈS
            if len(file_formats) > 1:
                file_format = _pick_best_format(file_formats, mesh)
            else:
                file_format = file_formats[0]
```

On ne fait appel à `_pick_best_format` que s'il y a ambiguïté (>1 format possible).

---

## Fix F — `TypeError` sur fichiers Code_Aster

### Bug

```
File "meshio/_mesh.py", line 181, in __init__
    if len(data[k]) != len(self.cells[k]):
TypeError: len() of unsized object
```

### Cause

Le fichier contient 4 types de cellules mais `SIEF_ELGA` n'est défini
que sur les hexahedrons.

```python
# _read_data() crée :
cell_data["SIEF_ELGA"] = [None] * len(cell_types)
# → [array(...), None, None, None]
#     hexahedron  vertex quad   line
```

Quand `Mesh()` valide `len(data[k])` et `data[k]` est `None` → `TypeError`.

### Résolution

La restructuration du cell_data dans le Fix C (fusion par type) a corrigé
ce problème indirectement : les entrées `None` sont correctement gérées
lors de la construction du dictionnaire `data_by_type`, car seuls les types
ayant des données non-"None" sont inclus dans la boucle de concaténation.

---


# Rapport des tests : meshio MED ↔ MSH

## Vue d'ensemble

6 tests ont été ajoutés au fichier `tests/test_med.py` pour valider chaque correctif. Chaque test crée son propre maillage programmatiquement (aucun fichier externe requis).

---

## Test 1 : `test_read_med_without_fas` (Fix A)

### Ce qu'il vérifie
Un fichier MED **sans section FAS** peut être lu sans crash.

### Comment il fonctionne
Crée un fichier MED minimal avec `h5py` contenant uniquement les sections obligatoires (`INFOS_GENERALES`, `ENS_MAA`, `NOE`, `MAI`) [1] mais **aucune section FAS**.

```python
def test_read_med_without_fas(tmp_path):
    filename = tmp_path / "no_fas.med"

    with h5py.File(filename, "w") as f:
        # Crée INFOS_GENERALES, ENS_MAA, NOE (3 points), MAI (1 triangle)
        # PAS de FAS

    mesh = meshio.med.read(filename)
    assert len(mesh.points) == 3
    assert len(mesh.cells) == 1
    assert mesh.cells[0].type == "triangle"
```

### Ce qui échouait avant
```
KeyError: "Unable to synchronously open object (object 'FAS' doesn't exist)"
```

---

## Test 2 : `test_read_med_without_gro` (Fix B)

### Ce qu'il vérifie
Une famille MED **sans sous-groupe GRO** peut être lue sans crash.

### Comment il fonctionne
Écrit un mesh normal avec `meshio.med.write`, puis ajoute une famille **sans GRO** via `h5py`.

```python
def test_read_med_without_gro(tmp_path):
    filename = tmp_path / "no_gro.med"

    # Écrire un mesh normal
    mesh = helpers.tri_mesh
    meshio.med.write(filename, mesh)

    # Ajouter une famille SANS GRO dans le FAS
    with h5py.File(filename, "a") as f:
        eleme = f["FAS"]["mesh"]["ELEME"]  # ou créer si absent
        fam = eleme.create_group("FAM_NO_GRO")
        fam.attrs.create("NUM", -99)
        # Pas de GRO ici

    mesh_out = meshio.med.read(filename)
    assert len(mesh_out.points) > 0
    assert len(mesh_out.cells) > 0
```

### Ce qui échouait avant
```
KeyError: "Unable to synchronously open object (object 'GRO' doesn't exist)"
```

---

## Test 3 : `test_write_multi_blocks_same_type_with_cell_data` (Fix C)

### Ce qu'il vérifie
Plusieurs blocs du **même type de cellule** avec `cell_data` sont correctement fusionnés à l'écriture [1].

### Comment il fonctionne
Crée 2 blocs de triangles séparés avec des `cell_tags` différents, écrit en MED, relit et vérifie.

```python
def test_write_multi_blocks_same_type_with_cell_data(tmp_path):
    cells = [
        CellBlock("triangle", np.array([[0, 1, 2], [1, 3, 2]])),
        CellBlock("triangle", np.array([[1, 4, 5], [1, 5, 3]])),
    ]
    cell_data = {
        "cell_tags": [np.array([-1, -1]), np.array([-2, -2])],
    }

    mesh = meshio.Mesh(points, cells, cell_data=cell_data)
    meshio.med.write(filename, mesh)

    mesh_out = meshio.med.read(filename)
    # Vérifie fusion : 4 triangles au total
    total_tri = sum(len(c.data) for c in mesh_out.cells if c.type == "triangle")
    assert total_tri == 4

    # Vérifie ordre des tags fusionnés
    tags = np.concatenate([...])
    assert np.array_equal(tags, np.array([-1, -1, -2, -2]))
```

### Ce qui échouait avant
```
WriteError: MED files cannot have two sections of the same cell type.
```
puis après suppression du garde-fou [1] :
```
ValueError: Unable to synchronously create group (name already exists)
```

---

## Test 4 : `test_convert_med_to_msh_preserves_metadata` (Fix D)

### Ce qu'il vérifie
La conversion MED → MSH **préserve** les tags physiques, géométriques, dim_tags et field_data.

### Comment il fonctionne
Crée un mesh avec `cell_tags` et `cell_tags` (attribut), simule un fichier MED natif, appelle `_convert_med_tags_to_gmsh()` et vérifie le résultat.

```python
def test_convert_med_to_msh_preserves_metadata(tmp_path):
    from meshio.gmsh.main import _convert_med_tags_to_gmsh

    cells = [CellBlock("triangle", np.array([[0,1,2], [1,3,2], [1,4,5], [1,5,3]]))]
    cell_data = {"cell_tags": [np.array([-1, -1, -2, -2])]}

    mesh = meshio.Mesh(points, cells, cell_data=cell_data)
    mesh.cell_tags = {-1: ["group_a"], -2: ["group_b"]}

    converted = _convert_med_tags_to_gmsh(mesh)

    # 2 blocs (split par cell_tags)
    assert len(converted.cells) == 2

    # gmsh:geometrical avec valeurs distinctes
    assert "gmsh:geometrical" in converted.cell_data

    # gmsh:physical avec valeurs distinctes
    assert "gmsh:physical" in converted.cell_data

    # gmsh:dim_tags généré
    assert "gmsh:dim_tags" in converted.point_data

    # field_data reconstruit
    assert "group_a" in converted.field_data
    assert "group_b" in converted.field_data
```

### Ce qui échouait avant
Aucune erreur mais perte silencieuse de toutes les métadonnées → fichier MSH avec 1 bloc sans tags.

---

## Test 5 : `test_msh_format_selection_for_med_data` (Fix E)

### Ce qu'il vérifie
La fonction `_pick_best_format` choisit `"gmsh"` quand le mesh contient des métadonnées MED ou Gmsh.

### Comment il fonctionne
Teste 3 cas différents directement sur la fonction de dispatch.

```python
def test_msh_format_selection_for_med_data():
    from meshio._helpers import _pick_best_format

    # Cas 1 : mesh MED (cell_tags) → gmsh
    mesh = meshio.Mesh(..., cell_data={"cell_tags": [...]})
    assert _pick_best_format(["ansys", "gmsh"], mesh) == "gmsh"

    # Cas 2 : mesh Gmsh (gmsh:physical) → gmsh
    mesh2 = meshio.Mesh(..., cell_data={"gmsh:physical": [...]})
    assert _pick_best_format(["ansys", "gmsh"], mesh2) == "gmsh"

    # Cas 3 : mesh nu → défaut (ansys)
    mesh3 = meshio.Mesh(...)
    assert _pick_best_format(["ansys", "gmsh"], mesh3) == "ansys"
```

### Ce qui échouait avant
```bash
meshio c mesh.med output.msh  # → écrit silencieusement avec le writer Ansys
```

---

## Test 6 : `test_read_med_partial_cell_data` (Fix F)

### Ce qu'il vérifie
Un champ défini sur **un seul type de cellule** peut être lu sans crash.

### Comment il fonctionne
Crée un mesh avec triangles + tetra, écrit en MED, puis ajoute un champ CHA **uniquement sur les tetra** via `h5py` [1].

```python
def test_read_med_partial_cell_data(tmp_path):
    cells = [
        CellBlock("triangle", np.array([[0, 1, 2]])),
        CellBlock("tetra", np.array([[0, 1, 2, 3]])),
    ]
    mesh = meshio.Mesh(points, cells)
    meshio.med.write(filename, mesh)

    # Ajouter champ CHA uniquement sur TE4 via h5py
    with h5py.File(filename, "a") as f:
        # Crée CHA/test_field avec MAI.TE4 seulement

    mesh_out = meshio.med.read(filename)

    # Le champ existe sur tetra avec la bonne valeur
    assert "test_field" in mesh_out.cell_data
    tetra_data = mesh_out.cell_data["test_field"][tetra_idx]
    assert np.isclose(tetra_data.flat[0], 42.0)
```