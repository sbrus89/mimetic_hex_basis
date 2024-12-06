import time
import numpy as np
import matplotlib.pyplot as plt
import random
from shapely.geometry.polygon import Polygon
from shapely.geometry import Point
import netCDF4 as nc4
from scipy.sparse import coo_array

np.seterr(divide='ignore', invalid='ignore')
 
#R = 6356.0*1000.0
R= 6371220.0

def area(x, y, x1, y1, x2, y2):

    xr = x.ravel()
    yr = y.ravel()
    M = np.ones((xr.size, 3, 3))
    M[:, 1, 0] = xr
    M[:, 2, 0] = yr

    M[:, 1, 1] = x1
    M[:, 2, 1] = y1

    M[:, 1, 2] = x2
    M[:, 2, 2] = y2

    A = 0.5*np.linalg.det(M)
    A = A.reshape(x.shape)
    return A

def area_vec(x, y, x1, y1, x2, y2):

    x_shape = x.shape
    xr = x.ravel()
    yr = y.ravel()
    M = np.ones((xr.size, 3, 3, x1.size))

    npts = xr.size
    nv = x1.size

    xr = np.repeat(xr[:,np.newaxis], nv, axis=1)
    yr = np.repeat(yr[:,np.newaxis], nv, axis=1)

    M[:, 1, 0, :] = xr
    M[:, 2, 0, :] = yr

    M[:, 1, 1, :] = x1
    M[:, 2, 1, :] = y1

    M[:, 1, 2, :] = x2
    M[:, 2, 2, :] = y2

    M = np.transpose(M, (0, 3, 1, 2)).reshape((npts, nv, 3, 3))

    A = 0.5*np.linalg.det(M)
    out_shape = np.append(x.shape,(nv))
    A = A.reshape(out_shape)
    A = np.squeeze(A)

    return A

def area_c(xim1, yim1, xi, yi, xip1, yip1):

    M = np.ones((3, 3, xi.size))


    M[1, 0, :] = xim1
    M[2, 0, :] = yim1

    M[1, 1, :] = xi
    M[2, 1, :] = yi

    M[1, 2, :] = xip1
    M[2, 2, :] = yip1

    M = np.transpose(M, (2, 0, 1))

    A = 0.5*np.linalg.det(M)

    return A

def vector_basis(n, i, v, phi, norm_factor=1.0):

    ip1 = (i+1) % n
    ip2 = (i+2) % n

    Phix = (v[i,0] - v[i-1,0])*phi[i,:,:] + (v[ip1,0] - v[ip2,0])*phi[ip1,:,:]
    Phiy = (v[i,1] - v[i-1,1])*phi[i,:,:] + (v[ip1,1] - v[ip2,1])*phi[ip1,:,:]

    return Phix/norm_factor, Phiy/norm_factor

def wachpress_vec(n, v, xx, yy):

    npts = xx.size

    i = np.arange(0, n)
    ip1 = (i+1) % n

    C = area_c(v[i-1,0], v[i-1,1], v[i,0], v[i,1], v[ip1,0], v[ip1,1])
    A = area_vec(xx, yy, v[i,0], v[i,1], v[ip1,0], v[ip1,1])
    A = A.reshape((npts,n))
    A = np.repeat(A[:,:, np.newaxis], n, axis=2)

    mask = np.zeros((n,n), dtype=np.int32)
    j = np.arange(-1,n-1)
    mask[j,j] = 1
    mask[j,j-1] = 1
    mask=mask.T
    mask = np.repeat(mask[np.newaxis,:,:], npts, axis=0)
    A = np.ma.masked_array(A,mask=mask)
    A = np.prod(A,axis=1)
    A = A.filled()

    w = C*A
    sum_w = np.sum(w, axis=-1)

    phi = w/sum_w[:,np.newaxis]

    out_shape = np.append(xx.shape,(n))
    phi = np.reshape(phi,(out_shape))
    #axes = np.arange(-1,phi.ndim-1)
    #phi = np.transpose(phi, axes)
    phi = np.transpose(phi, (2,0,1))

    return phi

def wachpress(n,v,xx,yy,method='area'):

    if len(xx.shape) == 2:
      dims = (n,xx.shape[0],xx.shape[1])
    else:
      dims = (n, xx.shape[0],1)

    phi = np.zeros((dims))
    rx = np.zeros((dims))
    ry = np.zeros((dims))
    for i in range(n):
    
        ws = np.zeros_like(xx)
        for j in range(n):

            jp1 = (j+1) % n
            jm1 = (j-1) % n
    
            if method == 'distance':
                hjm1 = (v[j-1,0]-xx)*norm[j-1,0] + (v[j-1,1]-yy)*norm[j-1,1]
                hj = (v[j,0]-xx)*norm[j,0] + (v[j,1]-yy)*norm[j,1]
    
                cross = norm[j-1,0]*norm[j,1] - norm[j-1,1]*norm[j,0]
                wj = cross/np.multiply(hj,hjm1)
    
                if j == i:
                    wi = np.copy(wj)
                    rx[i,:,:] = norm[j-1,0]/hjm1 + norm[j,0]/hj
                    ry[i,:,:] = norm[j-1,1]/hjm1 + norm[j,1]/hj
    
            elif method == 'area':
                Ap = np.ones_like(xx)
                for k in range(n):
        
                    if (k != jm1) and (k != j):
                        kp1 = (k+1) % n
                        A = area(xx, yy, v[k,0], v[k,1], v[kp1,0], v[kp1,1])
                        Ap = np.multiply(Ap,A) 
        
                C = area(np.array([v[jm1,0]]), np.array([v[jm1,1]]), v[j,0], v[j,1], v[jp1,0], v[jp1,1])
                wj = C*Ap
        
                if j == i:
                    wi = np.copy(wj)

            ws = ws+wj
    
        phi[i,:,:] = np.divide(wi,ws)

    if method == 'distance':
        return phi, rx, ry
    elif method == 'area':
        return phi

def edge_normal(xi, yi, xip1, yip1):

    nx = yip1 - yi
    ny = -(xip1 - xi)
    mag = np.sqrt(nx**2 + ny**2)
    nx = nx/mag
    ny = ny/mag

    return nx, ny

def parameterize_line(t, xi, yi, xip1, yip1):

    tmid = (t[1:] + t[:-1]) / 2.0
    dt = t[1]-t[0] 

    x = (1.0-tmid)*xi + tmid*xip1
    y = (1.0-tmid)*yi + tmid*yip1
    
    nx, ny = edge_normal(xi, yi, xip1, yip1)
    
    ds = np.sqrt(np.square(xip1-xi) + np.square(yip1-yi))*dt

    return x, y, nx, ny, ds

def gnomonic_forward(lon, lat, lon0, lat0):

    cos_alpha = np.sin(lat0)*np.sin(lat) + np.cos(lat0)*np.cos(lat)*np.cos(lon-lon0)

    u = R*np.cos(lat)*np.sin(lon-lon0)/cos_alpha
    v = R*(np.cos(lat0)*np.sin(lat) - np.sin(lat0)*np.cos(lat)*np.cos(lon-lon0))/cos_alpha

    return u, v

def gnomonic_inverse(u, v, lon0, lat0):

    rho = np.sqrt(u**2 + v**2)
    alpha = np.arctan2(rho,R)

    lat = np.arcsin(np.cos(alpha)*np.sin(lat0) + v*np.sin(alpha)*np.cos(lat0)/rho)
    lon = lon0 + np.arctan2(u*np.sin(alpha), rho*np.cos(lat0)*np.cos(alpha) - v*np.sin(lat0)*np.sin(alpha)) 

    return lon, lat

def latlon_uv_jacobian(u, v, lon0, lat0):

    k = 1.0 + (u**2 + v**2)/R**2

    den = u**2 + (R*np.cos(lat0) - v*np.sin(lat0))**2
    dlondu = (R*np.cos(lat0) - v*np.sin(lat0))/den 
    dlondv = u*np.sin(lat0)/den

    den = R**3*k**(3.0/2.0)*np.sqrt(1.0 - (v*np.cos(lat0) + R*np.sin(lat0))**2/(R**2*k)) 
    dlatdu = -u*(v*np.cos(lat0) + R*np.sin(lat0))/den
    dlatdv = ((R**2 + u**2)*np.cos(lat0) - R*v*np.sin(lat0))/den

    return dlondu, dlondv, dlatdu, dlatdv

def gnomonic_integration(lon0, lat0, lon1, lat1, lon2, lat2, t):

    u1, v1 = gnomonic_forward(lon1, lat1, lon0, lat0)
    u2, v2 = gnomonic_forward(lon2, lat2, lon0, lat0)

    if isinstance(u1,np.ndarray):
        u = 0.5*(np.matmul(u2, (1.0+t).T) + np.matmul(u1, (1.0-t).T))
        v = 0.5*(np.matmul(v2, (1.0+t).T) + np.matmul(v1, (1.0-t).T))
    else:
        u = 0.5*((1.0+t)*u2 + (1.0-t)*u1)
        v = 0.5*((1.0+t)*v2 + (1.0-t)*v1)

    lon, lat = gnomonic_inverse(u, v, lon0, lat0)

    dudt = 0.5*(u2 - u1)
    dvdt = 0.5*(v2 - v1)

    dxdlat = -R*np.sin(lat)*np.cos(lon)
    dxdlon = -R*np.cos(lat)*np.sin(lon)

    dydlat = -R*np.sin(lat)*np.sin(lon)
    dydlon =  R*np.cos(lat)*np.cos(lon)

    dzdlat = R*np.cos(lat)
    #dzdlon = 0.0

    dlondu, dlondv, dlatdu, dlatdv = latlon_uv_jacobian(u, v, lon0, lat0)

    dlatdt = dlatdu*dudt + dlatdv*dvdt 
    dlondt = dlondu*dudt + dlondv*dvdt

    dxdt = dxdlat*dlatdt + dxdlon*dlondt
    dydt = dydlat*dlatdt + dydlon*dlondt
    dzdt = dzdlat*dlatdt #+ dzdlon*dlondt

    ds = np.sqrt(np.square(dxdt) + np.square(dydt) + np.square(dzdt))

    return ds, u, v

def transform_vector_components_uv_latlon(lon0, lat0, lon, lat, fu, fv):

    u, v = gnomonic_forward(lon, lat, lon0, lat0)

    dlondu, dlondv, dlatdu, dlatdv = latlon_uv_jacobian(u, v, lon0, lat0)

    beta  = np.sqrt(dlondu**2 + dlatdu**2)
    gamma = np.sqrt(dlondv**2 + dlatdv**2)

    flon = dlondu/beta*fu + dlondv/gamma*fv
    flat = dlatdu/beta*fu + dlatdv/gamma*fv    
    #flon = dlondu*fu + dlondv*fv
    #flat = dlatdu*fu + dlatdv*fv    

    return flon, flat

def transform_vector_components_latlon_uv(lon0, lat0, lon, lat, flon, flat):

    u, v = gnomonic_forward(lon, lat, lon0, lat0)

    dlondu, dlondv, dlatdu, dlatdv = latlon_uv_jacobian(u, v, lon0, lat0)

    beta  = np.sqrt(dlondu**2 + dlatdu**2)
    gamma = np.sqrt(dlondv**2 + dlatdv**2)

    fu = dlondu/beta*flon + dlatdu/beta*flat
    fv = dlondv/gamma*flon + dlatdv/gamma*flat    
    #fu = dlondu*flon + dlatdu*flat
    #fv = dlondv*flon + dlatdv*flat    

    return fu, fv


def interp_edges(function, target, target_field):

    lon0 = 0.5*(np.max(target.lonVertex) + np.min(target.lonVertex))
    lat0 = 0.5*(np.max(target.latVertex) + np.min(target.latVertex))
    
    # get number of edges
    print(target.nEdges)
    
    t, w = np.polynomial.legendre.leggauss(5)
    t = np.expand_dims(t, axis=1)
    
    t_start = time.time()

    #plot_edge = True
    plot_edge = False

    target_field.edge = np.zeros((target.nEdges))
    for edge in range(target.nEdges): 
    
        print(edge)
   
        # Get normal vector for target edge 
        vertices = target.verticesOnEdge[edge, 0:2] - 1 
        vertices = np.roll(vertices, 1) # this is important to account for how mpas defines vertices on an edge
        uVertex, vVertex = gnomonic_forward(target.lonVertex[vertices], target.latVertex[vertices], lon0, lat0)
        nu, nv = edge_normal(uVertex[0], vVertex[0], uVertex[1], vVertex[1])

        # Evaluate function at edge quadrature points
        lon1 = np.expand_dims(target.lonVertex[vertices[0]], axis=0)
        lat1 = np.expand_dims(target.latVertex[vertices[0]], axis=0)
        lon2 = np.expand_dims(target.lonVertex[vertices[1]], axis=0)
        lat2 = np.expand_dims(target.latVertex[vertices[1]], axis=0)
        ds, u, v =  gnomonic_integration(lon0, lat0, lon1, lat1, lon2, lat2, t)
        lon, lat = gnomonic_inverse(u, v, lon0, lat0)
        flon, flat = function(lon, lat)
        fu, fv = transform_vector_components_latlon_uv(lon0, lat0, lon, lat, flon, flat)

        if plot_edge:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.scatter(uVertex, vVertex, marker='o', color='k', alpha=0.5)
            ax.plot([uVertex[0], uVertex[1]], [vVertex[0], vVertex[1]],color='k', alpha=0.5)
            ax.quiver(0.5*(uVertex[0]+uVertex[1]), 0.5*(vVertex[0]+vVertex[1]), nu, nv)
            ax.scatter(u, v, marker='.', color='r')
            ax.quiver(u, v, fu, fv, color='b')
            ax.quiver(u, v, flon, flat, color='m')
            ax.axis('equal')
            plt.savefig('test_edge.png',dpi=500)
            plt.close()
            raise SystemExit(0)

        # compute integral over edge
        integral = -np.sum(w*(fu*nu + fv*nv)*ds)
        #integral = -np.sum(w*(flon*nu + flat*nv)*ds)
        target_field.edge[edge] = integral/target.dvEdge[edge]
      
    print(np.round(time.time() - t_start, 3))


def remap_edges(source, target, edge_mapping, source_field, target_field):

    color_list = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown']

    lon0 = 0.5*(np.max(source.lonVertex) + np.min(source.lonVertex))
    lat0 = 0.5*(np.max(source.latVertex) + np.min(source.latVertex))
    
    # get number of edges
    print(target.nEdges)
    print(source.nEdges)
    
    t, w = np.polynomial.legendre.leggauss(5)
    t = np.expand_dims(t, axis=1)
    
    t_start = time.time()

    target_field.edge = np.zeros((target.nEdges))
    max_sub_edges = edge_mapping.nb_sub_edges.shape[1]
    max_source_edges = source.edgesOnCell.shape[1]

    data = np.zeros((target.nEdges*max_sub_edges*max_source_edges))
    row = np.zeros_like(data, dtype=np.int64)
    col = np.zeros_like(data, dtype=np.int64)
    m = 0
    for edge in range(target.nEdges): 
    
        print(edge)
   
        # Find local edge number for global edge on cell 0 
        cell_target = target.cellsOnEdge[edge,0] - 1
        iEdge = np.where(target.edgesOnCell[cell_target,:] == edge + 1)[0][0] 
        n = target.nEdgesOnCell[cell_target]
        iEdgep1 = (iEdge+1) % n
   
        # Get normal vector for target edge 
        vertices = target.verticesOnCell[cell_target, 0:n] - 1 
        vertices = np.roll(vertices, 1) # this is important to account for how mpas defines vertices on an edge
        uVertex, vVertex = gnomonic_forward(target.lonVertex[vertices], target.latVertex[vertices], lon0, lat0)
        nu_target, nv_target = edge_normal(uVertex[iEdge], vVertex[iEdge], uVertex[iEdgep1], vVertex[iEdgep1])

        jEdge = iEdge-1 # this is important fot getting edge right in cells_assoc, lon/lat_sub_edge
        n_sub_edges = edge_mapping.nb_sub_edges[cell_target, jEdge] 

        plot_edge = False
        #if n_sub_edges < 5:
        #    plot_edge = False
        #else:
        #    plot_edge = True
        
        if plot_edge:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.scatter(uVertex, vVertex, marker='o', color='k', alpha=0.5)
            for i in range(n):
                ip1 = (i+1) % n
                ax.plot([uVertex[i], uVertex[ip1]], [vVertex[i], vVertex[ip1]],color='k', alpha=0.5)
            ax.quiver(0.5*(uVertex[iEdge]+uVertex[iEdgep1]), 0.5*(vVertex[iEdge]+vVertex[iEdgep1]), nu_target, nv_target)
        
        for sub_edge in range(n_sub_edges):
            print(f'   {sub_edge}')
   
            # Get vertices for sub edge source cell 
            sub_edge_cell = edge_mapping.cells_assoc[cell_target, jEdge, sub_edge] - 1
            n = source.nEdgesOnCell[sub_edge_cell]
            vertices = source.verticesOnCell[sub_edge_cell, 0:n] - 1
            vertices = np.roll(vertices, 1) # this is important to account for how mpas defines vertices on an edge
            vertices_p1 = np.roll(vertices, -1)
        
            # Cell vertex coordinates and edge normals in u, v
            uVertex, vVertex = gnomonic_forward(source.lonVertex[vertices], source.latVertex[vertices], lon0, lat0)
            uv = np.vstack((uVertex, vVertex)).T # package for call to watchpress, vector_basis etc
            i = np.arange(n)
            ip1 = (i+1) % n
            nu, nv = edge_normal(uv[i,0] ,uv[i,1], uv[ip1,0], uv[ip1,1])

            # Evaluate watchpress functions at edge quadrature points (for normalization)
            lon1 = np.expand_dims(source.lonVertex[vertices], axis=1)
            lat1 = np.expand_dims(source.latVertex[vertices], axis=1)
            lon2 = np.expand_dims(source.lonVertex[vertices_p1], axis=1)
            lat2 = np.expand_dims(source.latVertex[vertices_p1], axis=1)
            ds, u, v =  gnomonic_integration(lon0, lat0, lon1, lat1, lon2, lat2, t)
            phi = wachpress_vec(n, uv, u, v)
            
            # Evaluate watchpress functions at sub-edge quadrature points from target edge in u,v
            lat1_sub_edge = edge_mapping.lat_sub_edge[cell_target, jEdge, sub_edge]
            lon1_sub_edge = edge_mapping.lon_sub_edge[cell_target, jEdge, sub_edge]
            lat2_sub_edge = edge_mapping.lat_sub_edge[cell_target, jEdge, sub_edge+1]
            lon2_sub_edge = edge_mapping.lon_sub_edge[cell_target, jEdge, sub_edge+1]        
            ds_quad, u_quad, v_quad = gnomonic_integration(lon0, lat0, lon1_sub_edge, lat1_sub_edge, lon2_sub_edge, lat2_sub_edge, t)
            phi_quad = wachpress_vec(n, uv, u_quad.T, v_quad.T)
            ds_quad = np.squeeze(ds_quad)

            if plot_edge: 
                ax.plot([uVertex[i], uVertex[ip1]], [vVertex[i], vVertex[ip1]],color=color_list[sub_edge], alpha=0.5)
                ax.scatter(uVertex, vVertex, marker='o', color=color_list[sub_edge], alpha=0.5)
                ax.scatter(u_quad, v_quad, marker='x', color=color_list[sub_edge])
                ax.scatter(u, v, marker='.', color=color_list[sub_edge])
                ax.quiver(0.5*(uv[i,0]+uv[ip1,0]), 0.5*(uv[i,1]+uv[ip1,1]), nu[i], nv[i], color=color_list[sub_edge])
        
            for i in range(n):
                edge_source = source.edgesOnCell[sub_edge_cell,i] - 1
        
                # evaluate basis functions at quadrature points on edge
                Phiu, Phiv = vector_basis(n, i, uv, np.expand_dims(phi[:,i,:], -1), norm_factor=1.0)
                Phiu = np.squeeze(Phiu)
                Phiv = np.squeeze(Phiv)
        
                # compute integral over edge for basis function normalization factor      
                norm_integral = np.sum(w*(Phiu*nu[i] + Phiv*nv[i])*ds[i,:])
        
                # compute normalized basis functions at cell centers 
                Phiu, Phiv = vector_basis(n, i, uv, phi_quad, norm_factor=norm_integral)
                Phiu = np.squeeze(Phiu)
                Phiv = np.squeeze(Phiv)
        
                # compute reconstruction 
                integral = np.sum(w*(Phiu*nu_target + Phiv*nv_target)*ds_quad)
                coef = -source.edgeSignOnCell[sub_edge_cell, i]*source.dvEdge[edge_source]*integral/target.dvEdge[edge]
                target_field.edge[edge] = target_field.edge[edge] + coef*source_field.edge[edge_source]

                row[m] = edge
                col[m] = edge_source
                data[m] = coef 
                m = m + 1

        if plot_edge:
            ax.axis('equal')
            plt.savefig('test_cell.png',dpi=500)
            plt.close()
            raise SystemExit(0)

    M = coo_array((data, (row, col)), shape=(target.nEdges, source.nEdges)).toarray()
    btf_target_mv = M.dot(source_field.edge)

    print(np.max(np.abs(target_field.edge - btf_target_mv)))

    print(np.round(time.time() - t_start, 3))


def reconstruct_edges_to_centers(mesh, field_source, field_target):

    t_start = time.time()
    
    # get number of cells and edges
    print(mesh.nCells)
    print(mesh.nEdges)
    
    # gnomonic projection center
    lon0 = 0.5*(np.max(mesh.lonCell) + np.min(mesh.lonCell))
    lat0 = 0.5*(np.max(mesh.latCell) + np.min(mesh.latCell))
    
    # quadrature points for computing the edge integral for the basis function normalization
    t, w = np.polynomial.legendre.leggauss(5)
    t = np.expand_dims(t, axis=1)
    
    field_target.zonal = np.zeros((mesh.nCells))
    field_target.meridional = np.zeros((mesh.nCells))
    #nCells = 0
    for cell in range(mesh.nCells):
        print(cell)
    
        n = mesh.nEdgesOnCell[cell]
        vertices = mesh.verticesOnCell[cell, 0:n] - 1
        vertices = np.roll(vertices, 1) # this is important to account for how mpas defines vertices on an edge
        vertices_p1 = np.roll(vertices, -1)
    
    
        # Cell vertex coordinates and edge normals in u, v
        uVertex, vVertex = gnomonic_forward(mesh.lonVertex[vertices], mesh.latVertex[vertices], lon0, lat0)
        uv = np.vstack((uVertex, vVertex)).T # package for call to watchpress, vector_basis etc
        i = np.arange(n)
        ip1 = (i+1) % n
        nu, nv = edge_normal(uv[i,0] ,uv[i,1], uv[ip1,0], uv[ip1,1])
    
        # Evaluate watchpress functions at cell center coordinates in u,v
        uCell, vCell = gnomonic_forward(mesh.lonCell[cell], mesh.latCell[cell], lon0, lat0)
        uCell = np.expand_dims(np.asarray([uCell]), axis=0) # put single value in an array for call to watchpress
        vCell = np.expand_dims(np.asarray([vCell]), axis=0)
        phi_cell = wachpress_vec(n, uv, uCell, vCell)
    
        # Evaluate watchpress functions at quadrature points 
        lon1 = np.expand_dims(mesh.lonVertex[vertices], axis=1)
        lat1 = np.expand_dims(mesh.latVertex[vertices], axis=1)
        lon2 = np.expand_dims(mesh.lonVertex[vertices_p1], axis=1)
        lat2 = np.expand_dims(mesh.latVertex[vertices_p1], axis=1)
        ds, u, v =  gnomonic_integration(lon0, lat0, lon1, lat1, lon2, lat2, t)
        phi = wachpress_vec(n, uv, u, v)
    
        fu = np.zeros(uCell.shape)
        fv = np.zeros(vCell.shape)
        for i in range(n):
            ip1 = (i+1) % n
            edge = mesh.edgesOnCell[cell,i] - 1
    
            # evaluate basis functions at quadrature points
            Phiu, Phiv = vector_basis(n, i, uv, np.expand_dims(phi[:,i,:], axis=-1), norm_factor=1.0)
            Phiu = np.squeeze(Phiu)
            Phiv = np.squeeze(Phiv)
    
            # compute integral over edge for basis function normalization factor      
            integral = np.sum(w*(Phiu*nu[i] + Phiv*nv[i])*ds[i,:], axis=0)
    
            # compute normalized basis functions at cell centers 
            Phiu, Phiv = vector_basis(n, i, uv, phi_cell , norm_factor=integral)
    
            # compute reconstruction at cell center
            coef = -mesh.edgeSignOnCell[cell, i]*mesh.dvEdge[edge]*field_source.edge[edge]
            fu = fu + coef*Phiu
            fv = fv + coef*Phiv
    
        # compute lon lat vector components
        field_target.zonal[cell], field_target.meridional[cell] = transform_vector_components_uv_latlon(lon0, lat0, mesh.lonCell[cell], mesh.latCell[cell], fu, fv)
    
    print(np.round(time.time() - t_start, 3))


def plot_fields(mesh1, field1, mesh2, field2, fig_name):

    same_mesh = False
    if mesh1.nCells == mesh2.nCells:
        same_mesh = True

    if same_mesh:
        rows = 3
        plot_diff = True
    else:
        rows = 2
        plot_diff = False

    # Set up figure
    cols = 3
    fig = plt.figure(figsize=(18, 4.5*rows))
    k = 1
    vmin = -2.5
    vmax = 3.5
    levels = np.linspace(vmin, vmax, 10)
    mlevels = np.linspace(0, vmax, 10)
    
    # Plot original RBF field
    ax = fig.add_subplot(rows, cols, k)
    cf = ax.tricontourf(mesh1.lonCell, mesh1.latCell, field1.zonal, levels=levels, extend='both')
    ax.set_title('Zonal component')
    ax.set_ylabel('RFB')
    fig.colorbar(cf, ax=ax)
    k = k + 1
    
    ax = fig.add_subplot(rows, cols, k)
    cf = ax.tricontourf(mesh1.lonCell, mesh1.latCell, field1.meridional, levels=levels, extend='both')
    ax.set_title('Meridonal component')
    fig.colorbar(cf, ax=ax)
    k = k + 1
    
    ax = fig.add_subplot(rows, cols, k)
    mag1 = np.sqrt(field1.zonal**2 + field1.meridional**2)
    cf = ax.tricontourf(mesh1.lonCell, mesh1.latCell, mag1, levels=mlevels, extend='max')
    ax.set_title('Magnitude')
    fig.colorbar(cf, ax=ax)
    k = k + 1
    
    # Plot lat lon components of reconstructed field
    ax = fig.add_subplot(rows, cols, k)
    cf = ax.tricontourf(mesh2.lonCell, mesh2.latCell, field2.zonal, levels=levels, extend='both')
    fig.colorbar(cf, ax=ax)
    ax.set_ylabel('Mimetic interpolation')
    k = k + 1
    
    ax = fig.add_subplot(rows, cols, k)
    cf = ax.tricontourf(mesh2.lonCell, mesh2.latCell, field2.meridional, levels=levels, extend='both')
    fig.colorbar(cf, ax=ax)
    k = k + 1
    
    ax = fig.add_subplot(rows, cols, k)
    mag2 = np.sqrt(field2.zonal**2 + field2.meridional**2)
    cf = ax.tricontourf(mesh2.lonCell, mesh2.latCell, mag2, levels=mlevels, extend='max')
    fig.colorbar(cf, ax=ax)
    k = k + 1
    
    # Plot differences 
    if plot_diff:
        ax = fig.add_subplot(rows, cols, k)
        diff = field2.zonal-field1.zonal
        vrange = np.max(np.abs(diff)) 
        levels = np.linspace(-vrange, vrange, 10)
        cf = ax.tricontourf(mesh1.lonCell, mesh1.latCell, diff, cmap='RdBu', levels=levels) 
        fig.colorbar(cf, ax=ax)
        ax.set_ylabel('Mimetic-RBF')
        k = k + 1
        
        ax = fig.add_subplot(rows, cols, k)
        diff = field2.meridional-field1.meridional
        vrange = np.max(np.abs(diff)) 
        levels = np.linspace(-vrange, vrange, 10)
        cf = ax.tricontourf(mesh1.lonCell, mesh1.latCell, diff, cmap='RdBu', levels=levels)
        fig.colorbar(cf, ax=ax)
        k = k + 1
        
        ax = fig.add_subplot(rows, cols, k)
        diff = mag2-mag1
        vrange = np.max(np.abs(diff)) 
        levels = np.linspace(-vrange, vrange, 10)
        cf = ax.tricontourf(mesh1.lonCell, mesh1.latCell, diff, cmap='RdBu', levels=levels)
        fig.colorbar(cf, ax=ax)
        k = k + 1
    
    plt.savefig(fig_name)
    plt.close()

#skip_test = True
skip_test = False
if not skip_test:
    #######################################################################
    #######################################################################
    
    n = 6      # Number of polygon sides
    eps = 0.0  # Factor to perturb polygon vertices
    Nx = 300   # Number of points in x direction 
    Ny = 300   # Number of points in y direction
    nt = 101   # Number of quadrature points along an edge
    N = int(Nx/20) # Quiver plot subsample factor
    ncols = 3
    nrows = int(np.ceil(n/ncols))
    figsize = (16,4*nrows)
    
    ############################################
    # Define polygon
    ############################################
    
    # Create vertices
    drad = 2.0*np.pi/float(n)
    v = []
    for i in range(n):
        rad = i*drad
        dx = random.uniform(0.0,1.0)
        dy = random.uniform(0.0,1.0)
        v.append([np.cos(rad) + eps*dx, np.sin(rad) + eps*dy])
    v = np.asarray(v)
    
    # Generate x,y coordinates
    x = np.linspace(-1.0-eps, 1.0+eps, Nx)
    y = np.linspace(-1.0-eps, 1.0+eps, Ny)
    dx = x[1]-x[0]
    dy = y[1]-y[0]
    xx, yy = np.meshgrid(x, y)
    xxm = 0.25*(xx[:-1,1:]+xx[:-1,:-1]+xx[1:,1:]+xx[1:,:-1])
    yym = 0.25*(yy[:-1,1:]+yy[:-1,:-1]+yy[1:,1:]+yy[1:,:-1])
    xy = np.vstack((xx.ravel(), yy.ravel())).T
    mask = np.array(xx.ravel(),dtype='bool')
    
    # Find coordinates inside polygon
    polygon = Polygon(v)
    for pt in range(Nx*Ny):
        p = Point(xy[pt])
        mask[pt] = polygon.contains(p)
    mask = mask.reshape(xx.shape)
    
    # Compute normals
    norm = np.zeros_like(v)
    for j in range(n):
      jp1 = (j+1) % n
      norm[j,0] = v[jp1,1] - v[j,1]
      norm[j,1] = -(v[jp1,0] - v[j,0])
      norm[j,:] = norm[j,:] / np.sqrt(np.square(norm[j,0]) + np.square(norm[j,1]))
    
    ############################################
    # Wachpress functions
    ############################################

    # Compute Wachpress coordinates
    phi_vec = wachpress_vec(n, v, xx, yy)
    phi, rx, ry = wachpress(n, v, xx, yy, method='distance')
    phi_area = wachpress(n, v, xx, yy, method='area')
    x = xx.ravel()[:,np.newaxis]
    y = yy.ravel()[:,np.newaxis]
    phi_area2 = wachpress(n, v, x, y, method='area')
    phi_area2 = np.reshape(phi_area2, phi_area.shape)
    phi_vec2 = wachpress_vec(n, v, x, y)
    phi_vec2 = np.reshape(phi_vec2, phi_area.shape)
    phi[:,~mask] = np.nan
    phi_area[:,~mask] = np.nan
    phi_area2[:,~mask] = np.nan
    phi_vec[:,~mask] = np.nan
    phi_vec2[:,~mask] = np.nan
    
    # Ensure both distance and area methods are equivalent
    diff = np.nanmax(np.abs(phi-phi_area))
    if diff > 1e-15:
        print("Distance and area methods do not agree")
        print(diff)
        raise SystemExit(0)

    diff = np.nanmax(np.abs(phi-phi_area2))
    if diff > 1e-15:
        print("Distance and area methods do not agree")
        print(diff)
        raise SystemExit(0)

    # Ensure both vectorized and standard functions are equivalent
    diff = np.nanmax(np.abs(phi-phi_vec))
    if diff > 1e-15:
        print("Distance and vectorized area methods do not agree")
        print(diff)
        raise SystemExit(0)

    # Compute gradients
    phix = np.zeros((n,xx.shape[0],xx.shape[1]))
    phiy = np.zeros((n,xx.shape[0],xx.shape[1]))
    for i in range(n):
    
        sx = np.zeros_like(xx)
        sy = np.zeros_like(xx)
        for j in range(n):
            sx = sx + phi[j,:,:]*rx[j,:,:]
            sy = sy + phi[j,:,:]*ry[j,:,:]
    
        phix[i,:,:] = phi[i,:,:]*(rx[i,:,:] - sx)
        phiy[i,:,:] = phi[i,:,:]*(ry[i,:,:] - sy)
    
    # Plot Wachpress coordinates
    fig,ax = plt.subplots(nrows, ncols, figsize=figsize)
    ax = ax.reshape(-1)
    for i in range(n):
    
        ip1 = (i+1) % n
    
        cr = np.linspace(0,1.0,10)
        ax[i].scatter(v[i,0], v[i,1])
        c = ax[i].contourf(xx, yy, phi[i,:,:], cr)
        cbar = fig.colorbar(c)
        ax[i].quiver(xx[::N,::N],yy[::N,::N],phix[i,::N,::N],phiy[i,::N,::N])
        ax[i].quiver(0.5*(v[i,0]+v[ip1,0]), 0.5*(v[i,1]+v[ip1,1]), norm[i,0], norm[i,1])
        ax[i].axis('equal')
    
    fig.tight_layout()
    plt.savefig('hex.png',bbox_inches='tight')
    
    ############################################
    # Vector basis functions
    ############################################
    
    # Plot vector basis functions
    fig,ax = plt.subplots(nrows, ncols, figsize=figsize)
    ax = ax.reshape(-1)
    
    for i in range(n):
    
        ip1 = (i+1) % n
    
        Phix, Phiy = vector_basis(n, i, v, phi)
    
        # plot magnitude
        cr = np.linspace(0,1.0,10)
        c = ax[i].contourf(xx, yy, np.sqrt(np.square(Phix)+np.square(Phiy)), cr)
        cbar = fig.colorbar(c)
    
        # plot vectors
        ax[i].quiver(xx[::N,::N],yy[::N,::N],Phix[::N,::N],Phiy[::N,::N])
    
        ## plot edge normal
        #ax[i].quiver(0.5*(v[i,0]+v[ip1,0]), 0.5*(v[i,1]+v[ip1,1]), norm[i,0], norm[i,1])
    
        # plot edge lines
        for j in range(n):
            jp1 = (j+1) % n
            ax[i].plot([v[j,0], v[jp1,0]], [v[j,1], v[jp1,1]],color='k')
    
         # plot vertices
        for j in range(n):
            ax[i].scatter(v[j,0], v[j,1],color='k')
    
        ax[i].axis('equal')
    
    fig.tight_layout()
    plt.savefig('edge.png',bbox_inches='tight')
    
    # Compute line integral quadrature points
    x = np.zeros((nt-1,n))
    y = np.zeros((nt-1,n))
    nx = np.zeros((nt-1,n))
    ny = np.zeros((nt-1,n))
    ds = np.zeros((nt-1,n))
    
    t = np.linspace(0.0,1.0,nt)
    tmid = (t[1:] + t[:-1]) / 2.0
    dt = t[1]-t[0] 
    for i in range(n):
        ip1 = (i+1) % n
    
        x[:,i], y[:,i], nx[:,i], ny[:,i], ds[:,i] = parameterize_line(t, v[i,0] ,v[i,1], v[ip1,0], v[ip1,1])
    
    # Compute line integrals
    phi = wachpress(n, v, x, y, method='area')
    norm_factors = np.zeros(n)
    print("Line integrals along polygon edges")
    for i in range(n):
    
        ip1 = (i+1) % n
        ip2 = (i+2) % n
    
        # Numerical integral along edge
        Phix, Phiy = vector_basis(n, i, v, phi)
        integral = np.sum((Phix*nx + Phiy*ny)*ds, axis=0)
        integral[integral < 1e-15] = 0.0
        #print(integral[i])
    
        # Analytical integral along edge
        xip1 = v[ip1,0]
        xi = v[i,0]
        xip2 = v[ip2,0]
        xim1 = v[i-1,0]
    
        yip1 = v[ip1,1]
        yi = v[i,1]
        yip2 = v[ip2,1]
        yim1 = v[i-1,1]
    
        norm_factors[i]= 0.5*(yip1-yi)*((xi-xim1)+(xip1-xip2)) - 0.5*(xip1-xi)*((yi-yim1)+(yip1-yip2))
        #print(norm_factors[i])
       
        # Check between numerical and analytical integrals 
        if abs(integral[i] - norm_factors[i]) > 1e-12:
            print("Edge integration issue")
        
        # Integration of normalized functions
        Phix, Phiy = vector_basis(n, i, v, phi, norm_factors[i])
        integral = np.sum((Phix*nx + Phiy*ny)*ds, axis=0)
        integral[abs(integral) < 1e-15] = 0.0
        print(integral)
    
        if abs(integral[i] - 1.0) > 1e-12:
            print("Normalized edge integration problem")
    
    # Compute area integral of divergence
    print("Area integral of divergence over polygon")
    fig,ax = plt.subplots(nrows, ncols, figsize=figsize)
    ax = ax.reshape(-1)
    
    phi, rx, ry = wachpress(n, v, xx, yy, method='distance')
    phi[:,~mask] = np.nan
    integral = np.zeros((Nx-1,Ny-1))
    for i in range(n):
    
        ip1 = (i+1) % n
    
        for j in range(n):
            jp1 = (j+1) % n
            ax[i].scatter(v[j,0], v[j,1])
            ax[i].plot([v[j,0], v[jp1,0]], [v[j,1], v[jp1,1]])
    
        Phix, Phiy = vector_basis(n, i, v, phi, norm_factors[i])
    
        dPhix = Phix[:-1,1:]-Phix[:-1,:-1]+Phix[1:,1:]-Phix[1:,:-1]
        dPhiy = Phiy[1:,:-1]-Phiy[:-1,:-1]+Phiy[1:,1:]-Phiy[:-1,1:]
    
        ax[i].quiver(xxm[::N,::N],yym[::N,::N],dPhix[::N,::N],dPhiy[::N,::N])
        ax[i].axis('equal')
    
        integral = integral + 0.5*dx*dPhix+0.5*dy*dPhiy
    
    print(np.nansum(integral))    
    fig.tight_layout()
    plt.savefig('div.png',bbox_inches='tight')
    
    ############################################
    # Sub-polygon integals
    ############################################
    print("")
    
    # Define sub-polygon coordinates
    s = 0.15
    
    v1 = np.zeros((n-1,2))
    v2 = np.zeros((n-1,2))
    
    v1[0,:] = v[0,:]
    v1[1,:] = (1.0-s)*v[0,:] + s*v[1,:]
    v1[2,:] = (1.0-s)*v[3,:] + s*v[4,:]
    v1[3,:] = v[4,:]
    v1[4,:] = v[5,:]
    
    v2[0,:] = (1.0-s)*v[0,:] + s*v[1,:]
    v2[1,:] = v[1,:]
    v2[2,:] = v[2,:]
    v2[3,:] = v[3,:]
    v2[4,:] = (1.0-s)*v[3,:] + s*v[4,:]
    
    fig,ax = plt.subplots(nrows, ncols, figsize=figsize)
    ax = ax.reshape(-1)
    
    line_integral = 0.0
    area_integral = 0.0
    for k in range(2):
    
        if k == 0:
          vs = v1
          color = 'r'
        elif k == 1:
          vs = v2
          color = 'b'
        
        # Plot vector basis functions and sub-polygons
        for i in range(n):
        
            ip1 = (i+1) % n
        
            for j in range(n):
                jp1 = (j+1) % n
                ax[i].scatter(v[j,0], v[j,1])
                ax[i].plot([v[j,0], v[jp1,0]], [v[j,1], v[jp1,1]])
        
            Phix, Phiy = vector_basis(n, i, v, phi)
        
            ax[i].quiver(xx[::N,::N],yy[::N,::N],Phix[::N,::N],Phiy[::N,::N])
            ax[i].quiver(0.5*(v[i,0]+v[ip1,0]), 0.5*(v[i,1]+v[ip1,1]), norm[i,0], norm[i,1])
            ax[i].axis('equal')
        
            for j in range(n-1):
                jp1 = (j+1) % (n-1)
                ax[i].plot([vs[j,0], vs[jp1,0]], [vs[j,1], vs[jp1,1]], color=color, alpha=0.2, linewidth=5)
        
        # Compute sub-polygon line integrals
        x = np.zeros((nt-1,n-1))
        y = np.zeros((nt-1,n-1))
        nx = np.zeros((nt-1,n-1))
        ny = np.zeros((nt-1,n-1))
        ds = np.zeros((nt-1,n-1))
        for i in range(n-1):
            ip1 = (i+1) % (n-1)
        
            x[:,i], y[:,i], nx[:,i], ny[:,i], ds[:,i] = parameterize_line(t, vs[i,0] ,vs[i,1], vs[ip1,0], vs[ip1,1])
        
        phi = wachpress(n, v, x, y, method='area')
        tot_integral = np.zeros(n-1)
        print("Sub-polygon line integrals")
        for i in range(n):
        
            # Integration of normalized functions
            Phix, Phiy = vector_basis(n, i, v, phi, norm_factors[i])
            integral = np.sum((Phix*nx + Phiy*ny)*ds, axis=0)
            integral[abs(integral) < 1e-15] = 0.0
            print(integral)
            tot_integral = tot_integral + integral
       
            Nl = 10 
            ax[i].quiver(x[::Nl,:],y[::Nl,:],nx[::Nl,:],ny[::Nl,:], color=color)
        
        fig.tight_layout()
        plt.savefig('sub.png',bbox_inches='tight')
    
        print("Sub-polygon total line integral")
        print(np.sum(tot_integral))
        line_integral = line_integral + np.sum(tot_integral)
        
        # Find coordinates inside sub polygon
        submask = np.array(xx.ravel(),dtype='bool')
        polygon = Polygon(vs)
        for pt in range(Nx*Ny):
            p = Point(xy[pt])
            submask[pt] = polygon.contains(p)
        submask = submask.reshape(xx.shape)
        
        # Compute sub-polygon divergence area integral 
        phi, rx, ry = wachpress(n, v, xx, yy, method='distance')
        phi[:,~submask] = np.nan
        integral = np.zeros((Nx-1,Ny-1))
        for i in range(n):
        
            ip1 = (i+1) % n
        
            for j in range(n):
                jp1 = (j+1) % n
                ax[i].scatter(v[j,0], v[j,1])
                ax[i].plot([v[j,0], v[jp1,0]], [v[j,1], v[jp1,1]])
        
            Phix, Phiy = vector_basis(n, i, v, phi, norm_factors[i])
        
            dPhix = Phix[:-1,1:]-Phix[:-1,:-1]+Phix[1:,1:]-Phix[1:,:-1]
            dPhiy = Phiy[1:,:-1]-Phiy[:-1,:-1]+Phiy[1:,1:]-Phiy[:-1,1:]
        
            integral = integral + 0.5*dx*dPhix+0.5*dy*dPhiy
       
        print("Sub-polygon area integral") 
        print(np.nansum(integral))    
        area_integral = area_integral + np.nansum(integral)
    
    print("")
    print("Total line integral")
    print(line_integral)
    print("Total area integral")
    print(area_integral)
    
    
    ############################################
    # Spherical line integration
    ############################################
    
    # New York
    lon1 = -74.006 
    lat1 = 40.7128
    
    # London
    lon2 = -0.1276
    lat2 = 51.5072
    
    ## Washington DC
    #lon2 = -77.0369
    #lat2 = 38.9072
    
    lon1 = np.radians(lon1)
    lat1 = np.radians(lat1)
    
    lon2 = np.radians(lon2)
    lat2 = np.radians(lat2)
    
    lon0 = 0.5*(lon1 + lon2)
    lat0 = 0.5*(lat1 + lat2)
    t, w = np.polynomial.legendre.leggauss(50)
    
    ds, u, v =  gnomonic_integration(lon0, lat0, lon1, lat1, lon2, lat2, t)
    L = np.sum(ds*w)
    print(L/1000.0)
    
    hav = np.sin(0.5*(lat2-lat1))**2 + np.cos(lat1)*np.cos(lat2)*np.sin(0.5*(lon2-lon1))**2
    d = 2.0*R*np.arcsin(np.sqrt(hav))
    print(d/1000.0)

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
        self.edgeSignOnCell = nc_mesh.variables['edgeSignOnCell'][:]
        self.verticesOnEdge = nc_mesh.variables['verticesOnEdge'][:]
        self.dvEdge = nc_mesh.variables['dvEdge'][:]
        self.angleEdge = nc_mesh.variables['angleEdge'][:]

        self.nEdges = self.lonEdge.size
        self.nCells = self.lonCell.size

        nc_mesh.close()

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

############################################
# Remap MPAS edge field from 16km to 32km 
############################################
use_exact_field = True
#use_exact_field = False

#skip_remap = True
skip_remap = False
if not skip_remap:

    source_mesh_filename = 'soma_16km_mpas_mesh_with_rbf_weights.nc'
    target_mesh_filename = 'soma_32km_mpas_mesh_with_rbf_weights.nc'
    edge_information_filename = 'target_edge.nc'
   
    # Create mesh, mapping, and field objects 
    source = Mesh(source_mesh_filename)
    target = Mesh(target_mesh_filename)
    edge_mapping = Mapping(edge_information_filename)
    source_field = Field(source_mesh_filename)
    target_field = Field(target_mesh_filename) 
   
    if use_exact_field: 
        #source_field.set_edge_field(function, source)
        interp_edges(function, source, source_field)
        target_field.set_edge_field(function, target)
        target_exact = Field(target_mesh_filename)
        target_exact.set_edge_field(function, target)

    plot_fields(source, source_field, target, target_field, 'field_target_source_rbf.png')

    remap_edges(source, target, edge_mapping, source_field, target_field)

    if use_exact_field:
        rmse = np.sqrt(np.mean(np.square(target_exact.edge - target_field.edge)))
        print(rmse) 

############################################
# Reconstruct MPAS edge field at cell centers 
############################################
mesh_filename = 'soma_32km_mpas_mesh_with_rbf_weights.nc'
mesh = Mesh(mesh_filename)
field_s = Field(mesh_filename)
field_t = Field(mesh_filename)

# read in fields
if not skip_remap:
    field_s.edge = target_field.edge

reconstruct_edges_to_centers(mesh, field_s, field_t)

if use_exact_field:
    plot_fields(mesh, target_exact, mesh, field_t, 'field.png')
    rmse = np.sqrt(np.mean(np.square(target_exact.zonal - field_t.zonal)))
    print(rmse)
    rmse = np.sqrt(np.mean(np.square(target_exact.meridional - field_t.meridional)))
    print(rmse)
    max_err = np.max(np.abs(target_exact.zonal - field_t.zonal))
    print(max_err)
    max_err = np.max(np.abs(target_exact.meridional - field_t.meridional))
    print(max_err)
else:
    plot_fields(mesh, field_s, mesh, field_t, 'field.png')
