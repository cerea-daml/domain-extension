'''Collection of tools for treating vtk data. Based on quickview.fonctions'''

import vtk
from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk # type: ignore

import numpy as np
from matplotlib import tri


########################## DATA READING #########################
def read_vtk(filename):
    reader=vtk.vtkDataSetReader()
    reader.SetFileName(filename)
    reader.Update()
    return reader.GetOutput()


def get_coords(data, loc:str = 'cells'):
    """
       Return coordinates,  either at cells (loc='cells') or at point (loc='points')
       data: vtkUnstructuredGrid
       loc: either 'cells' or 'points
    """

    if loc == 'cells':
        centres=vtk.vtkCellCenters()
        centres.SetInputData(data)
        centres.Update()
        data = centres.GetOutput()

    coords=vtk_to_numpy(data.GetPoints().GetData())

    return coords

def get_field(data,nom,loc="cells"):
    """get a field from the data, either at cells (loc='cells') or at point (loc='points')
    """

    #Verification
    assert loc in ("cells","points"),   "L'argument loc doit valoir 'cells' (champs aux cellules) ou 'points' (champs aux points) - STOP"

    Getdata={"cells":"GetCellData","points":"GetPointData"}

    field=getattr(data,Getdata[loc])().GetArray(nom)
    assert field is not None, f'{nom} not in data'
    field = vtk_to_numpy(field)
    return field


#################################### PLOTTING ################################
def triangulate_slice(slicexy):
    '''extract the triangulation of a horizontal slice of data'''
    triangles = slicexy.GetPolys().GetData()
    npts = slicexy.GetPoints().GetNumberOfPoints()
    ntri = int(triangles.GetNumberOfTuples()/4)
    coords = slicexy.GetPoints().GetData()
    x,y,_ = vtk_to_numpy(coords).T
    
    #Get the triangulation of the mesh
    triang = np.zeros((ntri, 3))
    for i in range(0, ntri):
        triang[i, 0] = triangles.GetTuple(4*i + 1)[0]
        triang[i, 1] = triangles.GetTuple(4*i + 2)[0]
        triang[i, 2] = triangles.GetTuple(4*i + 3)[0]
    
    triang = tri.Triangulation(x,y,triang)
    
    return triang

def get_slice(mesh, height):

    #add an ID field
    id_field = np.arange(mesh.GetNumberOfCells())
    id_field = numpy_to_vtk(id_field)
    id_field.SetName('IDs')
    mesh.GetCellData().AddArray(id_field)

    #define slice plave
    plane = vtk.vtkPlane()
    plane.SetOrigin(0,0,height)
    plane.SetNormal(0,0,1)

    # Extraction de la coupe
    filtre = vtk.vtkCutter()
    filtre.SetInputData(mesh)
    filtre.SetCutFunction(plane)
    filtre.Update()
    coupe=filtre.GetOutput()

    coupe_ids = get_field(coupe,'IDs').astype(int)
    triang = triangulate_slice(coupe)

    return triang, coupe_ids