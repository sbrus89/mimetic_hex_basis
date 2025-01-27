import netCDF4 as nc4
import numpy as np

class Mesh:

    def __init__(self, mesh_filename):

        nc_mesh = nc4.Dataset(mesh_filename, 'r+')

        self.lonVertex = nc_mesh.variables['lonVertex'][:]
        self.latVertex = nc_mesh.variables['latVertex'][:]
        self.lonEdge = nc_mesh.variables['lonEdge'][:]
        self.latEdge = nc_mesh.variables['latEdge'][:]
        self.lonCell = nc_mesh.variables['lonCell'][:]
        self.latCell = nc_mesh.variables['latCell'][:]

        self.lonVertex[self.lonVertex > np.pi] = self.lonVertex[self.lonVertex > np.pi] - 2.0*np.pi
        self.lonCell[self.lonCell > np.pi] = self.lonCell[self.lonCell > np.pi] - 2.0*np.pi

        self.cellsOnEdge = nc_mesh.variables['cellsOnEdge'][:]
        self.edgesOnCell = nc_mesh.variables['edgesOnCell'][:]
        self.verticesOnCell = nc_mesh.variables['verticesOnCell'][:]
        self.nEdgesOnCell = nc_mesh.variables['nEdgesOnCell'][:]
        self.verticesOnEdge = nc_mesh.variables['verticesOnEdge'][:]
        self.dvEdge = nc_mesh.variables['dvEdge'][:]
        self.angleEdge = nc_mesh.variables['angleEdge'][:]

        self.nEdges = self.lonEdge.size
        self.nCells = self.lonCell.size

        try:
           self.edgeSignOnCell = nc_mesh.variables['edgeSignOnCell'][:]
        except:
           self.edgeSignOnCell = self.compute_edgeSignOnCell()

        nc_mesh.close()

    def compute_edgeSignOnCell(self):

        edgeSignOnCell = np.zeros_like(self.edgesOnCell)
        for iCell in range(self.nCells):
            for i in range(self.nEdgesOnCell[iCell]):
                iEdge = self.edgesOnCell[iCell, i] - 1
                if iCell + 1 == self.cellsOnEdge[iEdge, 0]:
                    edgeSignOnCell[iCell, i] = -1.0
                else:
                    edgeSignOnCell[iCell, i] = 1.0
        return edgeSignOnCell

class Mapping:

    def __init__(self, edge_information_filename):

        edge_info = nc4.Dataset(edge_information_filename, 'r+')

        self.nb_sub_edges = edge_info.variables['nb_sub_edge'][:]
        self.cells_assoc = edge_info.variables['cells_assoc'][:]
        self.lat_sub_edge = edge_info.variables['lat_sub_edge'][:]
        self.lon_sub_edge = edge_info.variables['lon_sub_edge'][:] 

        edge_info.close()

class Field:

    def __init__(self, field_filename):

        nc_file = nc4.Dataset(field_filename, 'r+')

        self.edge = np.squeeze(nc_file.variables['barotropicThicknessFlux'][:])
        self.zonal = np.squeeze(nc_file.variables['barotropicThicknessFluxZonal'][:])
        self.meridional = np.squeeze(nc_file.variables['barotropicThicknessFluxMeridional'][:])

        nc_file.close()

    def set_edge_field(self, function, mesh):

        flon_edge, flat_edge = function(mesh.lonEdge, mesh.latEdge)
        flon_cell, flat_cell = function(mesh.lonCell, mesh.latCell)

        self.zonal = flon_cell
        self.meridional = flat_cell
        self.edge = np.cos(mesh.angleEdge)*flon_edge + np.sin(mesh.angleEdge)*flat_edge

    def average_to_edges(self, mesh):

        for i in range(mesh.nEdges):
            cell1 = mesh.cellsOnEdge[i,0] - 1
            cell2 = mesh.cellsOnEdge[i,1] - 1

            zonalEdge = 0.5*(self.zonal[cell1] + self.zonal[cell2])
            meridionalEdge = 0.5*(self.meridional[cell1] + self.meridional[cell2])

            self.edge[i] = np.cos(mesh.angleEdge[i])*zonalEdge + np.sin(mesh.angleEdge[i])*meridionalEdge

def function(lon, lat):

    flon = 2.0*np.cos(24.0*lon)
    flat = 2.0*np.cos(24.0*lat)

    return flon, flat

