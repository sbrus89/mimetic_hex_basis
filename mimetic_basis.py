import time
import numpy as np
import matplotlib.pyplot as plt
import random
from shapely.geometry.polygon import Polygon
from shapely.geometry import Point
from scipy.sparse import coo_array
import netCDF4 as nc4

from basis import wachpress, wachpress_vec, vector_basis, grad_wachpress, integrate_cell, vector_basis_mimetic
from coordinates import edge_normal, parameterize_line, transform_coordinates_forward, transform_coordinates_inverse, parameterize_integration, transform_vector_components_latlon_uv, transform_vector_components_uv_latlon, R
from remap import interp_edges, remap_edges, reconstruct_edges_to_centers
from mesh_map_classes import Mesh, Mapping, Field, function
from plotting import plot_cell_vector_fields, plot_edge_fields

np.seterr(divide='ignore', invalid='ignore')
 
#R = 6356.0*1000.0
#R= 6371220.0

gnomonic = False
#gnomonic = True



skip_test = True
#skip_test = False
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

    phix2, phiy2 = grad_wachpress(phi,rx,ry)
    diffx = np.nanmax(np.abs(phix-phix2))
    diffy = np.nanmax(np.abs(phiy-phiy2))
    if diffx > 1e-15 or diffy > 1e-15:
        print("Watchpress derivatives do not agree")
        print(diffx,diffy)
        raise SystemExit(0)
    
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

    #for j in range(n):
    #    print(j)
    #    jp1 = (j+1) % n
    #    jp2 = (j+2) % n

    #    intx_j, inty_j = integrate_cell(j,v)
    #    intx_jp1, inty_jp1 = integrate_cell(jp1,v)

    #    vjx = (v[j,0] - v[j-1,0])
    #    vjy = (v[j,1] - v[j-1,1])
    #    Lj = np.sqrt(vjx**2 + vjy**2)

    #    vjp1x = (v[jp1,0] - v[jp2,0])
    #    vjp1y = (v[jp1,1] - v[jp2,1])
    #    Ljp1 = np.sqrt(vjp1x**2 + vjp1y**2)

    #    phix, phiy  = vector_basis_mimetic(n, j, v, phi, coef=True)

    ############################################
    # Vector basis functions
    #############################################
    
    # Plot vector basis functions
    fig,ax = plt.subplots(nrows, ncols, figsize=figsize)
    ax = ax.reshape(-1)
   
    fx = np.zeros_like(xx)
    fy = np.zeros_like(xx) 
    for i in range(n):
    
        ip1 = (i+1) % n
        ip2 = (i+2) % n
        ip3 = (i+3) % n
    
        #Phix, Phiy = vector_basis(n, i, v, phi)
        Phix, Phiy = vector_basis_mimetic(n, i, v, phi)
    
        # plot magnitude
        mag = np.sqrt(np.square(Phix)+np.square(Phiy))
        cr = np.linspace(0,np.nanmax(mag),10)
        c = ax[i].contourf(xx, yy, mag, cr)
        cbar = fig.colorbar(c)
    
        # plot vectors
        ax[i].quiver(xx[::N,::N],yy[::N,::N],Phix[::N,::N],Phiy[::N,::N])
    
        ## plot edge normal
        ax[i].quiver(0.5*(v[i,0]+v[ip1,0]), 0.5*(v[i,1]+v[ip1,1]), norm[i,0], norm[i,1])

        # plot edge lines
        for j in range(n):
            jp1 = (j+1) % n
            ax[i].plot([v[j,0], v[jp1,0]], [v[j,1], v[jp1,1]],color='k')
    
         # plot vertices
        for j in range(n):
            ax[i].scatter(v[j,0], v[j,1],color='k')
    
        ax[i].axis('equal')

        fx = fx + Phix
        fy = fy + Phiy
    
    fig.tight_layout()
    plt.savefig('edge.png',bbox_inches='tight')

    fig = plt.figure(figsize=(12,4))
    ax = fig.add_subplot(131)
    c = ax.contourf(xx, yy, fx)
    cbar = fig.colorbar(c)
    ax.axis('equal')

    ax = fig.add_subplot(132)
    c = ax.contourf(xx, yy, fy)
    cbar = fig.colorbar(c)
    ax.axis('equal')

    ax = fig.add_subplot(133)
    mag = np.sqrt(np.square(fx)+np.square(fy))
    c = ax.contourf(xx, yy, mag)
    ax.quiver(xx[::N,::N],yy[::N,::N],fx[::N,::N],fy[::N,::N])
    cbar = fig.colorbar(c)
    ax.axis('equal')
    fig.tight_layout()
    plt.savefig('edge_sum.png',bbox_inches='tight')
    
    
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
        #Phix, Phiy = vector_basis(n, i, v, phi)
        Phix, Phiy = vector_basis_mimetic(n, i, v, phi)
        integral = np.sum((Phix*nx + Phiy*ny)*ds, axis=0)
        integral[integral < 1e-15] = 0.0
        print(integral[i])
    
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
        print(norm_factors[i])
       
        # Check between numerical and analytical integrals 
        if abs(integral[i] - norm_factors[i]) > 1e-12:
            print("Edge integration issue")
        norm_factors[i] = integral[i]
        
        # Integration of normalized functions
        #Phix, Phiy = vector_basis(n, i, v, phi, norm_factors[i])
        Phix, Phiy = vector_basis_mimetic(n, i, v, phi)
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
    
        #Phix, Phiy = vector_basis(n, i, v, phi, norm_factors[i])
        Phix, Phiy = vector_basis_mimetic(n, i, v, phi)
    
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
        
            #Phix, Phiy = vector_basis(n, i, v, phi)
            Phix, Phiy = vector_basis_mimetic(n, i, v, phi)
        
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
        
            #Phix, Phiy = vector_basis(n, i, v, phi, norm_factors[i])
            Phix, Phiy = vector_basis_mimetic(n, i, v, phi)
        
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
    t, w_gp = np.polynomial.legendre.leggauss(50)
    
    ds, u, v, w =  parameterize_integration(lon0, lat0, lon1, lat1, lon2, lat2, t, gnomonic)
    L = np.sum(ds*w_gp)
    print("")
    print("gnomonic integration")
    print(L/1000.0)
    
    hav = np.sin(0.5*(lat2-lat1))**2 + np.cos(lat1)*np.cos(lat2)*np.sin(0.5*(lon2-lon1))**2
    d = 2.0*R*np.arcsin(np.sqrt(hav))
    print("Haversine distance")
    print(d/1000.0)
    print("Difference")
    diff = abs(L-d)
    print(diff)

    if diff > 1e-8 and gnomonic:
        print("Arc length integration issue")
        raise SystemExit(0)

#raise SystemExit(0)

############################################
# Remap MPAS edge field from 16km to 32km 
############################################
print("")
use_exact_field = True
#use_exact_field = False

#skip_remap = True
skip_remap = False
if not skip_remap:

    #source_mesh_filename = 'soma_16km_mpas_mesh_with_rbf_weights.nc'
    #target_mesh_filename = 'soma_32km_mpas_mesh_with_rbf_weights.nc'
    #edge_information_filename = 'target_edge_iulian.nc'

    ###################################################################
    #source_mesh_filename = '16km_mesh_culled.nc'
    #target_mesh_filename = '32km_mesh_culled_16to32.nc'
    #edge_information_filename = 'target_edge_16to32.nc'

    ###################################################################
    source_mesh_filename = '4km_mesh.nc'
    target_mesh_filename = '128km_mesh_culled.nc'
    edge_information_filename = 'target_edge_4to128.nc'

    #source_mesh_filename = '4km_mesh.nc'
    #target_mesh_filename = '64km_mesh_culled.nc'
    #edge_information_filename = 'target_edge_4to64.nc'

    #source_mesh_filename = '4km_mesh.nc'
    #target_mesh_filename = '32km_mesh_culled.nc'
    #edge_information_filename = 'target_edge_4to32.nc'

    #source_mesh_filename = '4km_mesh.nc'
    #target_mesh_filename = '16km_mesh_culled.nc'
    #edge_information_filename = 'target_edge_4to16_iulian.nc'

    ###################################################################
    #source_mesh_filename = '4km_mesh.nc'
    #target_mesh_filename = '128km_mesh_culled.nc'
    #edge_information_filename = 'target_edge_4to128.nc'

    #source_mesh_filename = '8km_mesh_culled.nc'
    #target_mesh_filename = '128km_mesh_culled.nc'
    #edge_information_filename = 'target_edge_8to128.nc'

    #source_mesh_filename = '16km_mesh_culled.nc'
    #target_mesh_filename = '128km_mesh_culled.nc'
    #edge_information_filename = 'target_edge_16to128.nc'

    #source_mesh_filename = '32km_mesh_culled.nc'
    #target_mesh_filename = '128km_mesh_culled.nc'
    #edge_information_filename = 'target_edge_32to128.nc'

    #source_mesh_filename = '64km_mesh_culled.nc'
    #target_mesh_filename = '128km_mesh_culled.nc'
    #edge_information_filename = 'target_edge_64to128.nc'
   
    # Create mesh, mapping, and field objects 
    source = Mesh(source_mesh_filename)
    target = Mesh(target_mesh_filename)
    edge_mapping = Mapping(edge_information_filename)
    print("Read source field")
    source_field = Field(source_mesh_filename, source)
    print("\nRead target field")
    target_field = Field(target_mesh_filename, target) 

    if use_exact_field: 
        print("\nInitialize exact target field")
        source_exact = Field(source_mesh_filename, source)
        #interp_edges(function, source, source_exact, gnomonic)
        source_exact.set_cell_field(function, source)

        print("\nInitialize exact source field")
        target_exact = Field(target_mesh_filename, target)
        interp_edges(function, target, target_exact, gnomonic)
        target_exact.set_cell_field(function, target)

        #reconstruct_edges_to_centers(source, source_field, source_field, gnomonic)

    plot_cell_vector_fields(source, source_field, 'interpolated', source, source_exact, 'exact', f'source_cell_field_{source.resolution}.png')
    plot_edge_fields(source, source_field, 'interpolated', source, source_exact, 'exact', f'source_edge_field_{source.resolution}.png')

    remap_edges(source, target, edge_mapping, source_field, target_field, gnomonic)

    plot_edge_fields(target, target_field, 'remapped', target, target_exact, 'exact', f'remapped_edge_field_{source.resolution}_to_{target.resolution}.png')

    if use_exact_field:
        rmse = np.sqrt(np.mean(np.square(target_exact.edge - target_field.edge)))
        #rmse = np.sqrt((np.square(target_exact.edge - target_field.edge)*target.dvEdge)/np.sum(target.dvEdge))
        print(f"edge rmse: {rmse}") 
        max_err = np.max(target_exact.edge - target_field.edge)
        print(f"edge max: {max_err}")

        diff = np.zeros((target.nCells))
        for cell in range(target.nCells):
            n = target.nEdgesOnCell[cell]
            sum_target = 0.0
            sum_exact = 0.0
            for j in range(n):
                edge = target.edgesOnCell[cell,j] - 1
                sum_exact  = sum_exact  - target.edgeSignOnCell[cell,j]*target_exact.edge[edge]*target.dvEdge[edge]
                sum_target = sum_target - target.edgeSignOnCell[cell,j]*target_field.edge[edge]*target.dvEdge[edge]
            diff[cell] = sum_exact - sum_target
            print(sum_exact, sum_target, diff[cell])
        print(f"max cell-sum error: {np.max(diff)}")

    # Write fields to file
    target_field.write_field('barotropicThicknessFluxDiff', target_field.edge - target_exact.edge)


############################################
# Reconstruct MPAS edge field at cell centers 
############################################
#mesh_filename = 'soma_32km_mpas_mesh_with_rbf_weights.nc'
mesh_filename = target_mesh_filename
mesh = Mesh(mesh_filename)
print("\nRead target field")
field_s = Field(mesh_filename, mesh)
print("\nRead target field")
field_t = Field(mesh_filename, mesh)


# read in fields
if not skip_remap:
    field_s.edge = target_field.edge

reconstruct_edges_to_centers(mesh, field_s, field_t, gnomonic)

# Write fields to file
field_t.write_field('barotropicThicknessFluxZonalDiff', field_t.zonal - target_exact.zonal)
field_t.write_field('barotropicThicknessFluxMeridionalDiff', field_t.meridional - target_exact.meridional)
field_t.write_field('barotropicThicknessFluxZonal', target_exact.zonal)
field_t.write_field('barotropicThicknessFluxMeridional', target_exact.meridional)


if use_exact_field:
    plot_cell_vector_fields(mesh, field_t, 'reconstruction', mesh, target_exact, 'exact' ,f'remapped_cell_field_{source.resolution}_to_{target.resolution}.png')
    rmse = np.sqrt(np.mean(np.square(target_exact.zonal - field_t.zonal)))
    print(f"zonal rmse: {rmse}")
    rmse = np.sqrt(np.mean(np.square(target_exact.meridional - field_t.meridional)))
    print(f"meridional rmse: {rmse}")
    max_err = np.max(np.abs(target_exact.zonal - field_t.zonal))
    print(f"zonal max: {max_err}")
    max_err = np.max(np.abs(target_exact.meridional - field_t.meridional))
    print(f"meridional max: {max_err}")
else:
    plot_cell_vector_fields(mesh, field_s, 'mimetic', mesh, field_t, 'RBF', 'field.png')
