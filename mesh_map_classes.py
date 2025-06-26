import netCDF4 as nc4
import numpy as np

R = 6371220.0

class Mesh:

    def __init__(self, mesh_filename):

        nc_mesh = nc4.Dataset(mesh_filename, 'r')

        self.mesh_filename = mesh_filename

        if mesh_filename.find('km') == -1:
            print('mesh_filename does not contain km')
        self.resolution = [s for s in mesh_filename.split('_') if "km" in s][0]

        self.lonVertex = nc_mesh.variables['lonVertex'][:]
        self.latVertex = nc_mesh.variables['latVertex'][:]
        self.lonEdge = nc_mesh.variables['lonEdge'][:]
        self.latEdge = nc_mesh.variables['latEdge'][:]
        self.lonCell = nc_mesh.variables['lonCell'][:]
        self.latCell = nc_mesh.variables['latCell'][:]

        if np.max(self.lonCell) > np.pi:
            self.lonVertex[self.lonVertex > np.pi] = self.lonVertex[self.lonVertex > np.pi] - 2.0*np.pi
            self.lonCell[self.lonCell > np.pi] = self.lonCell[self.lonCell > np.pi] - 2.0*np.pi
            self.lonEdge[self.lonEdge > np.pi] = self.lonEdge[self.lonEdge > np.pi] - 2.0*np.pi

        self.cellsOnEdge = nc_mesh.variables['cellsOnEdge'][:]
        self.edgesOnCell = nc_mesh.variables['edgesOnCell'][:]
        self.verticesOnCell = nc_mesh.variables['verticesOnCell'][:]
        self.nEdgesOnCell = nc_mesh.variables['nEdgesOnCell'][:]
        self.verticesOnEdge = nc_mesh.variables['verticesOnEdge'][:]
        self.dvEdge = nc_mesh.variables['dvEdge'][:]
        self.angleEdge = nc_mesh.variables['angleEdge'][:]

        # Off center point to evaluate reconstruction
        #self.lonCell = 0.5*(self.lonCell + self.lonEdge[self.edgesOnCell[:,0]-1])
        #self.latCell = 0.5*(self.latCell + self.latEdge[self.edgesOnCell[:,0]-1])

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

        edge_info = nc4.Dataset(edge_information_filename, 'r')

        self.nb_sub_edges = edge_info.variables['nb_sub_edge'][:]
        self.cells_assoc = edge_info.variables['cells_assoc'][:]
        self.lat_sub_edge = edge_info.variables['lat_sub_edge'][:]
        self.lon_sub_edge = edge_info.variables['lon_sub_edge'][:] 

        edge_info.close()

class Field:

    def __init__(self, field_filename, mesh):

        nc_file = nc4.Dataset(field_filename, 'r')

        self.nCells = mesh.nCells
        self.nEdges = mesh.nEdges
        self.field_filename = field_filename

        try: 
            self.edge = np.squeeze(nc_file.variables['barotropicThicknessFlux'][:])
            print(f"edge field read {self.edge.shape}")
        except:
            print(f"edge field not read from {field_filename}") 
            self.edge = np.zeros((mesh.nEdges))

        try:
            self.zonal = np.squeeze(nc_file.variables['barotropicThicknessFluxZonal'][:])
            print(f"zonal field read {self.zonal.shape}")
        except:
            print(f"zonal field not read from {field_filename}") 
            self.zonal = np.zeros((mesh.nCells))

        try:
            self.meridional = np.squeeze(nc_file.variables['barotropicThicknessFluxMeridional'][:])
            print(f"meridional field read {self.meridional.shape}")
        except:
            print(f"meridional field not read from {field_filename}") 
            self.meridional = np.zeros((mesh.nCells))

        nc_file.close()

    def set_edge_field(self, function, mesh):

        flon_edge, flat_edge = function(mesh.lonEdge, mesh.latEdge)

        self.edge = np.cos(mesh.angleEdge)*flon_edge + np.sin(mesh.angleEdge)*flat_edge
 
    def set_cell_field(self, function, mesh):

        flon_cell, flat_cell = function(mesh.lonCell, mesh.latCell)

        self.zonal = flon_cell
        self.meridional = flat_cell

    def average_to_edges(self, mesh):

        for i in range(mesh.nEdges):
            cell1 = mesh.cellsOnEdge[i,0] - 1
            cell2 = mesh.cellsOnEdge[i,1] - 1

            zonalEdge = 0.5*(self.zonal[cell1] + self.zonal[cell2])
            meridionalEdge = 0.5*(self.meridional[cell1] + self.meridional[cell2])

            self.edge[i] = np.cos(mesh.angleEdge[i])*zonalEdge + np.sin(mesh.angleEdge[i])*meridionalEdge

    def write_field(self, varname, values):

        ds = nc4.Dataset(self.field_filename, "r+")
        nc_vars = ds.variables.keys()

        if values.size == self.nCells:
            dimid = "nCells"
        elif values.size == self.nEdges:
            dimid = "nEdges"
        else:
            print("error finding dimension")

        if varname not in nc_vars:
            if "Time" not in ds.dimensions:
                ds.createDimension("Time", None)
            var = ds.createVariable(varname, np.float64, ("Time",dimid))
            var[0,:] = values
        else:
            var = ds.variables[varname]
            var[0,:] = values

        ds.close()

def function(lon, lat):

    flon = 2.0*np.cos(24.0*lon)
    flat = 2.0*np.cos(24.0*lat)

    return flon, flat

