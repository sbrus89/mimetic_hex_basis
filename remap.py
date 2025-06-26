import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import coo_array
import time
import xarray as xr
import netCDF4 as nc4
from progressbar import ProgressBar, Percentage, Bar, ETA, Timer

from basis import wachpress_vec, vector_basis, vector_basis_test, vector_basis_mimetic
from coordinates import edge_normal, transform_coordinates_forward, transform_coordinates_inverse, parameterize_integration, transform_vector_components_latlon_uv, transform_vector_components_uv_latlon

from plotting import plotReconstruct, plotRemap, plotInterp

color_list = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink']


def interp_edges(function, target, target_field, gnomonic=True):
    print("")
    print("Intergrate function along edges")
    print(f'   number of edges: {target.nEdges}')

    widgets = ['Integrating: ', Percentage(), ' ', Bar(), ' ', Timer(), ' ', ETA()]
    progress_bar = ProgressBar(widgets=widgets,
                               maxval=target.nEdges).start()
    
    t, w_gp = np.polynomial.legendre.leggauss(25)
    t = np.expand_dims(t, axis=1)

    target_field.edge = np.zeros((target.nEdges))
    #cell = 14508
    cell = -1
    plot = plotInterp(cell, target.nEdges)
    for edge in range(target.nEdges): 
    
        # Find local edge number for global edge on cell 0 
        cell = target.cellsOnEdge[edge,0] - 1
        iEdge = np.where(target.edgesOnCell[cell,:] == edge + 1)[0][0]
        n = target.nEdgesOnCell[cell]
        iEdgep1 = (iEdge+1) % n

        # projection center
        lon0 = target.lonEdge[edge]
        lat0 = target.latEdge[edge]
        #lon0 = target.lonCell[cell]
        #lat0 = target.latCell[cell]

        # Get normal vector for target edge 
        vertices = target.verticesOnCell[cell, 0:n] - 1
        vertices = np.roll(vertices, 1) # this is important to account for how mpas defines vertices on an edge
        uVertex, vVertex, wVertex = transform_coordinates_forward(target.lonVertex[vertices], target.latVertex[vertices], lon0, lat0, gnomonic)
        nu, nv = edge_normal(uVertex[iEdge], vVertex[iEdge], uVertex[iEdgep1], vVertex[iEdgep1])

        # Evaluate function at edge quadrature points
        lon1 = np.expand_dims(target.lonVertex[vertices[iEdge]], axis=0)
        lat1 = np.expand_dims(target.latVertex[vertices[iEdge]], axis=0)
        lon2 = np.expand_dims(target.lonVertex[vertices[iEdgep1]], axis=0)
        lat2 = np.expand_dims(target.latVertex[vertices[iEdgep1]], axis=0)
        ds, u, v, w = parameterize_integration(lon0, lat0, lon1, lat1, lon2, lat2, t, gnomonic)
        lon, lat = transform_coordinates_inverse(u, v, w, lon0, lat0, gnomonic)
        flon, flat = function(lon, lat) 
        fu, fv = transform_vector_components_latlon_uv(lon0, lat0, lon, lat, flon, flat, gnomonic)

        plot.plot_interp_edge(cell, edge, n, iEdge, uVertex, vVertex, u, v, nu, nv, fu, fv, flon, flat, function, lon0, lat0, gnomonic)

        # compute integral over edge
        L = np.sum(w_gp*ds)
        target.dvEdge[edge] = L
        integral = np.sum(w_gp*(fu*nu + fv*nv)*ds)
        target_field.edge[edge] = integral/L

        progress_bar.update(edge)
     
    progress_bar.finish()

    target_field.write_field('barotropicThicknessFlux', target_field.edge)

def remap_edges(source, target, edge_mapping, source_field, target_field, gnomonic=True):
    print("")
    print("Remap edges")

    print(f'   target edges: {target.nEdges}')
    print(f'   source edges: {source.nEdges}')

    widgets = ['Remapping: ', Percentage(), ' ', Bar(), ' ', Timer(), ' ',ETA()]
    progress_bar = ProgressBar(widgets=widgets,
                               maxval=target.nEdges).start()

    t, w_gp = np.polynomial.legendre.leggauss(15)
    t = np.expand_dims(t, axis=1)

    target_field.edge = np.zeros((target.nEdges))
    max_sub_edges = edge_mapping.nb_sub_edges.shape[1]
    max_source_edges = source.edgesOnCell.shape[1]

    data = np.zeros((4*target.nEdges*max_sub_edges*max_source_edges))
    row = np.zeros_like(data, dtype=np.int64)
    col = np.zeros_like(data, dtype=np.int64)
    m = 0

    #cell_target = 1919
    cell_target = -1
    plot = plotRemap(cell_target)
    for edge in range(target.nEdges):

        #print(edge)

        # Find local edge number for global edge on cell 0 
        cell_target = target.cellsOnEdge[edge,0] - 1
        iEdge = np.where(target.edgesOnCell[cell_target,:] == edge + 1)[0][0]
        n_target = target.nEdgesOnCell[cell_target]
        iEdgep1 = (iEdge+1) % n_target

        # projection center
        lon0 = target.lonEdge[edge]
        lat0 = target.latEdge[edge]
        #lon0 = target.lonCell[cell_target]
        #lat0 = target.latCell[cell_target]

        # Get normal vector for target edge 
        vertices = target.verticesOnCell[cell_target, 0:n_target] - 1
        vertices = np.roll(vertices, 1) # this is important to account for how mpas defines vertices on an edge
        uVertex_target, vVertex_target, wVertex_target = transform_coordinates_forward(target.lonVertex[vertices], target.latVertex[vertices], lon0, lat0, gnomonic)
        nu_target, nv_target = edge_normal(uVertex_target[iEdge], vVertex_target[iEdge], uVertex_target[iEdgep1], vVertex_target[iEdgep1])

        # Target edge length
        lon1 = target.lonVertex[target.verticesOnEdge[edge,0]-1]
        lat1 = target.latVertex[target.verticesOnEdge[edge,0]-1]
        lon2 = target.lonVertex[target.verticesOnEdge[edge,1]-1]
        lat2 = target.latVertex[target.verticesOnEdge[edge,1]-1]
        ds_quad, u_quad, v_quad, w_quad  = parameterize_integration(lon0, lat0, lon1, lat1, lon2, lat2, t, gnomonic)
        L_target = np.sum(w_gp*ds_quad.T)
        #L_target = target.dvEdge[edge]

        jEdge = (iEdge-1) % n_target # this is important fot getting edge right in cells_assoc, lon/lat_sub_edge
        n_sub_edges = edge_mapping.nb_sub_edges[cell_target, jEdge]

        for sub_edge in range(n_sub_edges):
            #print(f'   {sub_edge}')

            # Get vertices for sub edge source cell 
            sub_edge_cell = edge_mapping.cells_assoc[cell_target, jEdge, sub_edge] - 1
            n = source.nEdgesOnCell[sub_edge_cell]
            vertices = source.verticesOnCell[sub_edge_cell, 0:n] - 1
            vertices = np.roll(vertices, 1) # this is important to account for how mpas defines vertices on an edge
            vertices_p1 = np.roll(vertices, -1)

            # Cell vertex coordinates and edge normals in u, v
            uVertex, vVertex, wVertex = transform_coordinates_forward(source.lonVertex[vertices], source.latVertex[vertices], lon0, lat0, gnomonic)
            uv = np.vstack((uVertex, vVertex)).T # package for call to watchpress, vector_basis etc
            i = np.arange(n)
            ip1 = (i+1) % n
            nu, nv = edge_normal(uv[i,0] ,uv[i,1], uv[ip1,0], uv[ip1,1])

            # Evaluate watchpress functions at edge quadrature points (for normalization)
            lon1 = np.expand_dims(source.lonVertex[vertices], axis=1)
            lat1 = np.expand_dims(source.latVertex[vertices], axis=1)
            lon2 = np.expand_dims(source.lonVertex[vertices_p1], axis=1)
            lat2 = np.expand_dims(source.latVertex[vertices_p1], axis=1)
            ds, u, v, w =  parameterize_integration(lon0, lat0, lon1, lat1, lon2, lat2, t, gnomonic)
            phi = wachpress_vec(n, uv, u, v)

            # Evaluate watchpress functions at sub-edge quadrature points from target edge in u,v
            lat1_sub_edge = edge_mapping.lat_sub_edge[cell_target, jEdge, sub_edge]
            lon1_sub_edge = edge_mapping.lon_sub_edge[cell_target, jEdge, sub_edge]
            lat2_sub_edge = edge_mapping.lat_sub_edge[cell_target, jEdge, sub_edge+1]
            lon2_sub_edge = edge_mapping.lon_sub_edge[cell_target, jEdge, sub_edge+1]
            ds_quad, u_quad, v_quad, w_quad  = parameterize_integration(lon0, lat0, lon1_sub_edge, lat1_sub_edge, lon2_sub_edge, lat2_sub_edge, t, gnomonic)
            phi_quad = wachpress_vec(n, uv, u_quad.T, v_quad.T)
            ds_quad = np.squeeze(ds_quad)

            plot.plot_remap(cell_target, sub_edge, n_sub_edges, n_target, iEdge, uVertex_target, vVertex_target, nu_target, nv_target, n, uVertex, vVertex, uv, u, v, u_quad, v_quad, nu, nv)

            for i in range(n):
                edge_source = source.edgesOnCell[sub_edge_cell,i] - 1

                # evaluate basis functions at quadrature points on edge
                #Phiu, Phiv = vector_basis(n, i, uv, np.expand_dims(phi[:,i,:], -1), norm_factor=1.0)
                Phiu, Phiv = vector_basis_test(n, i, uv, np.expand_dims(phi[:,i,:], axis=-1), t, w_gp, ds, nu, nv)
                #Phiu, Phiv = vector_basis_mimetic(n, i, uv, np.expand_dims(phi[:,i,:], -1))
                Phiu = np.squeeze(Phiu)
                Phiv = np.squeeze(Phiv)

                # compute integral over edge for basis function normalization factor      
                norm_integral = np.sum(w_gp*(Phiu*nu[i] + Phiv*nv[i])*ds[i,:])

                # compute normalized basis functions at cell centers 
                #Phiu, Phiv = vector_basis(n, i, uv, phi_quad, norm_factor=norm_integral)
                #Phiu, Phiv = vector_basis(n, i, uv, phi_quad)
                Phiu, Phiv = vector_basis_test(n, i, uv, phi_quad, t, w_gp, ds, nu, nv, norm_integral)
                Phiu = np.squeeze(Phiu)
                Phiv = np.squeeze(Phiv)

                # compute reconstruction 
                L_source = np.sum(w_gp*ds[i,:])
                #L_source = source.dvEdge[edge_source]
                integral = np.sum(w_gp*(Phiu*nu_target + Phiv*nv_target)*ds_quad)
                coef = -source.edgeSignOnCell[sub_edge_cell, i]*L_source*integral/L_target
                target_field.edge[edge] = target_field.edge[edge] + coef*source_field.edge[edge_source]

                row[m] = edge
                col[m] = edge_source
                data[m] = coef
                m = m + 1

        progress_bar.update(edge)
     
    progress_bar.finish()

    #M = coo_array((data, (row, col)), shape=(target.nEdges, source.nEdges)).toarray()
    #btf_target_mv = M.dot(source_field.edge)
    #print(np.max(np.abs(target_field.edge - btf_target_mv)))

    ds = nc4.Dataset(f"weight_file_{source.resolution}_to_{target.resolution}.nc", "w")
    n_s = ds.createDimension("n_s", m)
    n_a = ds.createDimension("n_a", source.nEdges)
    n_b = ds.createDimension("n_b", target.nEdges)

    row_nc = ds.createVariable("row", int, ("n_s",))
    col_nc = ds.createVariable("col", int, ("n_s",))
    S_nc = ds.createVariable("S", np.float64, ("n_s",))

    row_nc[:] = row[0:m] + 1
    col_nc[:] = col[0:m] + 1
    S_nc[:] = data[0:m]

    ds.close()

    target_field.write_field('barotropicThicknessFluxRemapped', target_field.edge)



def reconstruct_edges_to_centers(mesh, field_source, field_target, gnomonic):

    from mesh_map_classes import function

    print("")
    print("Reconstruct edges to centers")
    print(f'   number of cells: {mesh.nCells}')
    print(f'   number of edge: {mesh.nEdges}')

    widgets = ['Reconstructing: ', Percentage(), ' ', Bar(), ' ', Timer(), ' ', ETA()]
    progress_bar = ProgressBar(widgets=widgets,
                               maxval=mesh.nCells).start()

    # quadrature points for computing the edge integral for the basis function normalization
    t, w_gp = np.polynomial.legendre.leggauss(5)
    t = np.expand_dims(t, axis=1)

    field_target.zonal = np.zeros((mesh.nCells))
    field_target.meridional = np.zeros((mesh.nCells))

    #cell = 1919
    #cell = 14508
    #cell = 2425
    #cell = 24657
    cell = 2254
    plot = plotReconstruct(cell)
    plot2 = plotReconstruct(835)
    plot3 = plotReconstruct(1894)
    plot4 = plotReconstruct(14360)
    for cell in range(mesh.nCells):

        # projection center
        lon0 = mesh.lonCell[cell]
        lat0 = mesh.latCell[cell]

        n = mesh.nEdgesOnCell[cell]
        vertices = mesh.verticesOnCell[cell, 0:n] - 1
        vertices = np.roll(vertices, 1) # this is important to account for how mpas defines vertices on an edge
        vertices_p1 = np.roll(vertices, -1)

        # Cell vertex coordinates and edge normals in u, v
        uVertex, vVertex, wVertex = transform_coordinates_forward(mesh.lonVertex[vertices], mesh.latVertex[vertices], lon0, lat0, gnomonic)
        uv = np.vstack((uVertex, vVertex)).T # package for call to watchpress, vector_basis etc
        i = np.arange(n)
        ip1 = (i+1) % n
        nu, nv = edge_normal(uv[i,0] ,uv[i,1], uv[ip1,0], uv[ip1,1])

        # Evaluate watchpress functions at cell center coordinates in u,v
        uCell, vCell, wCell = transform_coordinates_forward(mesh.lonCell[cell], mesh.latCell[cell], lon0, lat0, gnomonic)
        uCell = np.expand_dims(np.asarray([uCell]), axis=0) # put single value in an array for call to watchpress
        vCell = np.expand_dims(np.asarray([vCell]), axis=0)
        phi_cell = wachpress_vec(n, uv, uCell, vCell)

        # Evaluate watchpress functions at quadrature points along edges (for normalization)
        lon1 = np.expand_dims(mesh.lonVertex[vertices], axis=1)
        lat1 = np.expand_dims(mesh.latVertex[vertices], axis=1)
        lon2 = np.expand_dims(mesh.lonVertex[vertices_p1], axis=1)
        lat2 = np.expand_dims(mesh.latVertex[vertices_p1], axis=1)
        ds, u, v, w =  parameterize_integration(lon0, lat0, lon1, lat1, lon2, lat2, t, gnomonic)
        phi = wachpress_vec(n, uv, u, v)


        fu = np.zeros(uCell.shape)
        fv = np.zeros(vCell.shape)
        for i in range(n):
            edge = mesh.edgesOnCell[cell,i] - 1

            # evaluate basis functions at edge quadrature points
            #Phiu, Phiv = vector_basis(n, i, uv, np.expand_dims(phi[:,i,:], axis=-1), norm_factor=1.0)
            Phiu, Phiv = vector_basis_test(n, i, uv, np.expand_dims(phi[:,i,:], axis=-1), t, w_gp, ds, nu, nv)
            #Phiu, Phiv = vector_basis_mimetic(n, i, uv, np.expand_dims(phi[:,i,:], axis=-1))
            Phiu = np.squeeze(Phiu)
            Phiv = np.squeeze(Phiv)

            # compute integral over edge for basis function normalization factor      
            integral = np.sum(w_gp*(Phiu*nu[i] + Phiv*nv[i])*ds[i,:])

            #Phiu, Phiv, a , b = vector_basis_test(n, i, uv, phi_cell, norm_factor=integral)
            #ip1 = (i+1) % n
            #ip2 = (i+2) % n
            ##vi_mag =   np.sqrt((uv[i  ,0]-uv[i-1,0])**2 + (uv[i  ,1]-uv[i-1,1])**2)
            ##vip1_mag = np.sqrt((uv[ip1,0]-uv[ip2,0])**2 + (uv[ip1,1]-uv[ip2,1])**2)
            #vi_mag = 1.0
            #vip1_mag = 1.0 
            #ffu = 0.5*(1.0-t)*(uv[i,0]-uv[i-1,0])/vi_mag + 0.5*(1.0+t)*(uv[ip1,0]-uv[ip2,0])/vip1_mag
            #ffv = 0.5*(1.0-t)*(uv[i,1]-uv[i-1,1])/vi_mag + 0.5*(1.0+t)*(uv[ip1,1]-uv[ip2,1])/vip1_mag
            ##integral2 = np.sum(w_gp*(ffu.T*nu[i] + ffv.T*nv[i])*ds[i,:])
            #integral2 = ((a*(uv[i,0]-uv[i-1,0])/vi_mag + b*(uv[ip1,0]-uv[ip2,0])/vip1_mag)*nu[i] + (a*(uv[i,1]-uv[i-1,1])/vi_mag + b*(uv[ip1,1]-uv[ip2,1])/vip1_mag)*nv[i])*0.5*np.sum(w_gp*ds[i,:])
            #print(integral, integral2)

            # compute normalized basis functions at cell centers 
            #Phiu, Phiv = vector_basis(n, i, uv, phi_cell, norm_factor=integral)
            Phiu, Phiv = vector_basis_test(n, i, uv, phi_cell, t, w_gp, ds, nu, nv, integral)
            #Phiu, Phiv = vector_basis_test(n, i, uv, phi_cell, t, w_gp, ds, nu, nv)
            #Phiu, Phiv = vector_basis_mimetic(n, i, uv, phi_cell)

            # compute reconstruction at cell center
            L = np.sum(w_gp*ds[i,:])
            coef = -mesh.edgeSignOnCell[cell, i]*L*field_source.edge[edge]
            fu = fu + coef*Phiu
            fv = fv + coef*Phiv

            #plot.plot_cell_reconstruct(cell, n, i, uCell, vCell, uVertex, vVertex, uv, u, v, nu, nv, integral, coef, function, lon0, lat0, gnomonic, ds, t, w_gp)
            #plot2.plot_cell_reconstruct(cell, n, i, uCell, vCell, uVertex, vVertex, uv, u, v, nu, nv, integral, coef, function, lon0, lat0, gnomonic, ds, t, w_gp)
            #plot3.plot_cell_reconstruct(cell, n, i, uCell, vCell, uVertex, vVertex, uv, u, v, nu, nv, integral, coef, function, lon0, lat0, gnomonic, ds, t, w_gp)
            #plot4.plot_cell_reconstruct(cell, n, i, uCell, vCell, uVertex, vVertex, uv, u, v, nu, nv, integral, coef, function, lon0, lat0, gnomonic, ds, t, w_gp)

        # compute lon lat vector components
        field_target.zonal[cell], field_target.meridional[cell] = transform_vector_components_uv_latlon(lon0, lat0, mesh.lonCell[cell], mesh.latCell[cell], fu[0,0], fv[0,0], gnomonic)

        progress_bar.update(cell)

    progress_bar.finish()

    field_target.write_field('barotropicThicknessFluxZonalRemapped', field_target.zonal)
    field_target.write_field('barotropicThicknessFluxMeridionalRemapped', field_target.meridional)
