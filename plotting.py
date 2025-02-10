import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry.polygon import Polygon
from shapely.geometry import Point
from basis import wachpress_vec, vector_basis

color_list = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink']

def plot_edge_fields(mesh1, field1, label1, mesh2, field2, label2, fig_name):

    same_mesh = False
    if mesh1.nEdges == mesh2.nEdges:
        same_mesh = True

    if same_mesh:
        cols = 3 
        plot_diff = True
    else:
        cols = 2 
        plot_diff = False

    # Set up figure
    rows = 1
    fig = plt.figure(figsize=(18, 4.5*rows))
    k = 1 
    #vmin = -2.5
    #vmax = 3.5 
    #levels = np.linspace(vmin, vmax, 10) 
    #mlevels = np.linspace(0, vmax, 10) 
    levels = 10
    mlevels = 10
    
    # Plot original RBF field
    ax = fig.add_subplot(rows, cols, k)
    cf = ax.tricontourf(mesh1.lonEdge, mesh1.latEdge, field1.edge, levels=levels, extend='both')
    ax.set_title(label1)
    fig.colorbar(cf, ax=ax)
    k = k + 1 
    
    # Plot lat lon components of reconstructed field
    ax = fig.add_subplot(rows, cols, k)
    cf = ax.tricontourf(mesh2.lonEdge, mesh2.latEdge, field2.edge, levels=levels, extend='both')
    fig.colorbar(cf, ax=ax)
    ax.set_title(label2)
    k = k + 1 
    
    # Plot differences 
    if plot_diff:
        ax = fig.add_subplot(rows, cols, k)
        diff = field2.edge - field1.edge
        vrange = np.max(np.abs(diff)) 
        if vrange < 1e-8:
            vrange = 1e-8
        #vrange = 0.3887
        levels = np.linspace(-vrange, vrange, 10)
        #levels = 10
        cf = ax.tricontourf(mesh1.lonEdge, mesh1.latEdge, diff, cmap='RdBu', levels=levels)
        fig.colorbar(cf, ax=ax)
        ax.set_title(f'{label2}-{label1}')

    plt.savefig(fig_name)
    plt.close()

def plot_cell_vector_fields(mesh1, field1, label1, mesh2, field2, label2, fig_name):

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
    vmin = -1.7
    vmax = 1.7
    levels = np.linspace(vmin, vmax, 10) 
    vmax = 2.7
    mlevels = np.linspace(0, vmax, 10) 
    #levels = 10
    #mlevels = 10
    
    # Plot original RBF field
    ax = fig.add_subplot(rows, cols, k)
    cf = ax.tricontourf(mesh1.lonCell, mesh1.latCell, field1.zonal, levels=levels, extend='both')
    ax.set_title('Zonal component')
    ax.set_ylabel(label1)
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
    ax.set_ylabel(label2)
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
        if vrange < 1e-8:
            vrange = 1e-8
        #vrange = 0.2785
        levels = np.linspace(-vrange, vrange, 10)
        #levels = 10
        cf = ax.tricontourf(mesh1.lonCell, mesh1.latCell, diff, cmap='RdBu', levels=levels)
        fig.colorbar(cf, ax=ax)
        ax.set_ylabel(f'{label2}-{label1}')
        k = k + 1

        ax = fig.add_subplot(rows, cols, k)
        diff = field2.meridional-field1.meridional
        vrange = np.max(np.abs(diff)) 
        if vrange < 1e-8:
            vrange = 1e-8
        #vrange = 0.3598
        levels = np.linspace(-vrange, vrange, 10)
        #levels = 10
        cf = ax.tricontourf(mesh1.lonCell, mesh1.latCell, diff, cmap='RdBu', levels=levels)
        fig.colorbar(cf, ax=ax)
        k = k + 1

        ax = fig.add_subplot(rows, cols, k)
        diff = mag2-mag1
        vrange = np.max(np.abs(diff)) 
        if vrange < 1e-8:
            vrange = 1e-8
        #vrange = 0.3887
        levels = np.linspace(-vrange, vrange, 10)
        #levels = 10
        cf = ax.tricontourf(mesh1.lonCell, mesh1.latCell, diff, cmap='RdBu', levels=levels)
        fig.colorbar(cf, ax=ax)
        k = k + 1

    plt.savefig(fig_name)
    plt.close()

def plot_interp_edge(uVertex, vVertex, u, v, nu, nv, fu, fv, flon, flat):

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

def plot_cell_reconstruct(fig, n, i, uCell, vCell, uVertex, vVertex, uv, u, v, nu, nv, norm_factor):

    ax = fig.add_subplot(3,3,i+1)
    ax.scatter(uCell, vCell, marker='o', color='k', alpha=0.5)
    
    Nx = 100 
    Ny = 100 
    N = 5 
    x = np.linspace(np.min(uVertex), np.max(uVertex), Nx) 
    y = np.linspace(np.min(vVertex), np.max(vVertex), Ny) 
    xx, yy = np.meshgrid(x, y)
    mask = np.array(xx.ravel(),dtype='bool')
    xy = np.vstack((xx.ravel(), yy.ravel())).T
    polygon = Polygon(uv)
    for pt in range(Nx*Ny):
        p = Point(xy[pt])
        mask[pt] = polygon.contains(p)
    mask = mask.reshape(xx.shape)
    phi_vec_grid = wachpress_vec(n, uv, xx, yy) 
    phi_vec_grid[:,~mask] = np.nan
    Phix, Phiy = vector_basis(n, i, uv, phi_vec_grid, norm_factor=norm_factor)
    cr = np.linspace(0,1.0,10)
    c = ax.contourf(xx, yy, np.sqrt(np.square(Phix)+np.square(Phiy)))
    cbar = fig.colorbar(c)
    ax.quiver(xx[::N,::N],yy[::N,::N],Phix[::N,::N],Phiy[::N,::N])
    for j in range(n):
        jp1 = (j+1) % n 
        ax.scatter(uVertex[j],vVertex[j],marker='o', color=color_list[j])
        ax.plot([uVertex[j], uVertex[jp1]], [vVertex[j], vVertex[jp1]], color=color_list[j], alpha=0.5)
        ax.scatter(u[j,:], v[j,:], marker='x', color=color_list[j])
        ax.quiver(0.5*(uv[j,0]+uv[jp1,0]), 0.5*(uv[j,1]+uv[jp1,1]), nu[j], nv[j], color=color_list[j])
    ax.axis('equal')

    if i == n-1:
        plt.savefig('test_cell_reconstruct.png',dpi=500)
        plt.close()
        raise SystemExit(0)

def plot_remap(sub_edge, ax, n_target, iEdge, uVertex_target, vVertex_target, nu_target, nv_target, n, uVertex, vVertex, uv, u, v, u_quad, v_quad, nu, nv):

    # Plot target cell
    if sub_edge == 0:
        iEdgep1 = (iEdge+1) % n_target
        ax.scatter(uVertex, vVertex, marker='o', color='k', alpha=0.5)
        for j in range(n):
            jp1 = (j+1) % n
            ax.plot([uVertex_target[j], uVertex_target[jp1]], [vVertex_target[j], vVertex_target[jp1]],color='k', alpha=0.5)
        ax.quiver(0.5*(uVertex_target[iEdge]+uVertex_target[iEdgep1]), 0.5*(vVertex_target[iEdge]+vVertex_target[iEdgep1]), nu_target, nv_target)

    i = np.arange(n)
    ip1 = (i+1) % n

    # Plot source cell edges
    ax.plot([uVertex[i], uVertex[ip1]], [vVertex[i], vVertex[ip1]],color=color_list[sub_edge], alpha=0.5)

    # Plot source cell vertices
    ax.scatter(uVertex, vVertex, marker='o', color=color_list[sub_edge], alpha=0.5)

    # Plot quadrature points along target edge
    ax.scatter(u_quad, v_quad, marker='x', color=color_list[sub_edge])

    # Plot edge normalization quadrature points
    ax.scatter(u, v, marker='.', color=color_list[sub_edge])

    # Plot source cell normals
    ax.quiver(0.5*(uv[i,0]+uv[ip1,0]), 0.5*(uv[i,1]+uv[ip1,1]), nu[i], nv[i], color=color_list[sub_edge])

    ax.axis('equal')
    plt.savefig('test_cell.png',dpi=500)
    plt.close()
    raise SystemExit(0)

