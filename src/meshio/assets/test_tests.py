import meshio 

mesh =meshio.read("testinp.inp", file_format="abaqus")

#print (mesh)
print(mesh.cell_data)
print(mesh.point_data)
print(mesh.field_data)
print(mesh.cells)
#print(mesh.points)
