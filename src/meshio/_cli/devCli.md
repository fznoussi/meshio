Rapport des développements apportés au module _cli de meshio:
1. Modification dans le fichier _gmsh41.py:
   - Correction de la ligne 228 pour utiliser la méthode .get() afin de gérer les cas où tous les entités n'ont pas de tag physique, en enregistrant cela comme None au lieu d'une liste vide.
2. _infor.py est responsable de la CLI d'information sur les maillages. Il vérifiait le nombre de cellules et de sommets, en créant une matrice 2D  pour les sommets et les cellules. Cependant, les polygones n'ont pas de nombre fixe de sommets, ce qui a conduit à des erreurs : 
   - Les éléments de type "point" n'ont pas de sommets, ce qui a causé une erreur lors de la création de la matrice 2D.
   - On impose la création d'un tableau au moins 1D pour ce élements, pour éviter que ca crash lors de la création de la matrice 2D.
    