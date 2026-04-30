_med.py (existant, on ne touche pas)        _medmulti.py (nouveau)
────────────────────────────────────        ──────────────────────

read()                                      write_med_multi()   ← Couche 1
write()                                     read_med_multi()    ← Couche 1
_write_data()                                    
_read_families()                            _write_mesh()       ← Couche 2
_read_nodal_data()                          _write_families()   ← Couche 2
_read_cell_data()                           _write_fields()     ← Couche 2
_write_families()                           _read_single_mesh() ← Couche 2
                                            
                                            _write_nodes()         ← Couche 3
                                            _write_cells()         ← Couche 3
                                            _write_field_dataset() ← Couche 3
                                            _read_nodal_data()     ← Couche 3
                                            _read_cell_data()      ← Couche 3
                                            _read_fields()         ← Couche 3

                                            _resolve_mesh_names()    ← Couche 4
                                            _find_field_collisions() ← Couche 4
                                            _hdf5_field_name()       ← Couche 4
                                            _get_fas()               ← Couche 4
                                            _read_family_entries()   ← Couche 4
                                            _write_family_entries()  ← Couche 4


Étape 4 : `write_med_multi`

C'est la fonction principale d'écriture. Son rôle est :

```
1. Valider les entrées
2. Résoudre les noms de maillages
3. Détecter les collisions de champs
4. Ouvrir le fichier HDF5 une seule fois
5. Créer les groupes racines ENS_MAA, FAS, CHA
6. Pour chaque maillage : appeler les fonctions d'écriture
```

## Sa signature

```python
def write_med_multi(filename, meshes, mesh_names=None, med_version="4.1.0"):
```

## Ta mission

Écris uniquement la **validation des entrées** et **l'ouverture du fichier HDF5**. C'est à dire :

```
- vérifier que meshes n'est pas vide → raise WriteError
- vérifier que meshes est bien une liste → raise WriteError
- résoudre les noms avec _resolve_mesh_names
- détecter les collisions avec _find_field_collisions
- ouvrir le fichier HDF5 avec h5py
- créer les trois groupes racines ENS_MAA, FAS, CHA
```

Pas encore les boucles d'écriture, juste cette structure de base. Montre moi ce que tu obtiens.

Copyright (c) Simvia SAS 