import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry.polygon import Polygon
from shapely.geometry import Point
from basis import wachpress_vec, vector_basis
from coordinates import transform_coordinates_inverse, transform_vector_components_uv_latlon
from scipy.spatial import KDTree

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
    
    dx = 0.02
    x = np.arange(np.min(mesh1.lonCell), np.max(mesh1.lonCell), dx)
    y = np.arange(np.min(mesh1.latCell), np.max(mesh1.latCell), dx)
    xx, yy = np.meshgrid(x,y)
    xy = np.vstack((xx.ravel(), yy.ravel())).T

    pts = np.vstack((mesh1.lonCell.ravel(), mesh1.latCell.ravel())).T
    tree = KDTree(pts)
    d, idx = tree.query(xy)

    ax = fig.add_subplot(rows, cols, k)
    mag1 = np.sqrt(field1.zonal**2 + field1.meridional**2)
    cf = ax.tricontourf(mesh1.lonCell, mesh1.latCell, mag1, levels=mlevels, extend='max')
    ax.quiver(mesh1.lonCell[idx], mesh1.latCell[idx], field1.zonal[idx], field1.meridional[idx], scale=35)
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
    
    pts = np.vstack((mesh2.lonCell.ravel(), mesh2.latCell.ravel())).T
    tree = KDTree(pts)
    d, idx = tree.query(xy)

    ax = fig.add_subplot(rows, cols, k)
    mag2 = np.sqrt(field2.zonal**2 + field2.meridional**2)
    cf = ax.tricontourf(mesh2.lonCell, mesh2.latCell, mag2, levels=mlevels, extend='max')
    ax.quiver(mesh2.lonCell[idx], mesh2.latCell[idx], field2.zonal[idx], field2.meridional[idx], scale=35)
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

class plotInterp:

    def __init__(self, cell, nEdges):
        fig = plt.figure()
        self.ax = fig.add_subplot(111)
        self.edge_list = []
        self.nEdges = nEdges
        self.cell = cell

    def plot_interp_edge(self, cell, edge, n, iEdge, uVertex, vVertex, u, v, nu, nv, fu, fv, flon, flat, function, lon0, lat0, gnomonic):
    
        if cell != self.cell:
            return

        Nx = 100
        Ny = 100 
        N = 5
        ulin = np.linspace(np.min(uVertex), np.max(uVertex), Nx) 
        vlin = np.linspace(np.min(vVertex), np.max(vVertex), Ny) 
        uu, vv = np.meshgrid(ulin, vlin)
        ww = uu*0.0

        lon, lat = transform_coordinates_inverse(uu, vv, ww, lon0, lat0, gnomonic)
        fflon, fflat = function(lon, lat)

        self.ax.quiver(uu[::N,::N],vv[::N,::N],fflon[::N,::N],fflat[::N,::N])

        self.ax.scatter(uVertex, vVertex, marker='o', color='k', alpha=0.5)
        for j in range(n):
            jp1 = (j+1) % n
            self.ax.plot([uVertex[j], uVertex[jp1]], [vVertex[j], vVertex[jp1]], color='k', alpha=0.5)
        iEdgep1 = (iEdge+1) % n
        self.ax.quiver(0.5*(uVertex[iEdge]+uVertex[iEdgep1]), 0.5*(vVertex[iEdge]+vVertex[iEdgep1]), nu, nv)

        self.ax.scatter(u, v, marker='x', color='r')

        self.ax.quiver(u, v, fu, fv, color='b')
        self.ax.quiver(u, v, flon, flat, color='m')

        self.edge_list.append(iEdge)

        if edge == self.nEdges:
            self.plot_finalize()

    def plot_finalize(self):
        self.ax.axis('equal')
        plt.savefig(f'test_interp_cell_{self.cell}.png',dpi=500)
        plt.close()
        raise SystemExit(0)

class plotReconstruct:

    def __init__(self, cell):

        self.fig =  plt.figure(figsize=(16,8))
        self.cell = cell

    def plot_cell_reconstruct(self, cell, n, i, uCell, vCell, uVertex, vVertex, uv, u, v, nu, nv, norm_factor, coef, function, lon0, lat0, gnomonic):

        if cell != self.cell:
            return

        rows = 4
        cols = 4 
        ax = self.fig.add_subplot(rows,cols,i+1)
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
        if i == 0:
            self.fu = np.zeros_like(xx)
            self.fv = np.zeros_like(xx)
            self.Phix = np.zeros_like(xx)
            self.Phiy = np.zeros_like(xx)


        phi_vec_grid = wachpress_vec(n, uv, xx, yy) 
        phi_vec_grid[:,~mask] = np.nan
        Phix, Phiy = vector_basis(n, i, uv, phi_vec_grid, norm_factor=norm_factor)
        cr = np.linspace(0,1.0,10)
        c = ax.contourf(xx, yy, np.sqrt(np.square(Phix)+np.square(Phiy)))
        cbar = self.fig.colorbar(c)
        ax.quiver(xx[::N,::N],yy[::N,::N],Phix[::N,::N],Phiy[::N,::N])
        for j in range(n):
            jp1 = (j+1) % n 
            ax.scatter(uVertex[j],vVertex[j],marker='o', color=color_list[j])
            ax.plot([uVertex[j], uVertex[jp1]], [vVertex[j], vVertex[jp1]], color=color_list[j], alpha=0.5)
            ax.scatter(u[j,:], v[j,:], marker='x', color=color_list[j])
            ax.quiver(0.5*(uv[j,0]+uv[jp1,0]), 0.5*(uv[j,1]+uv[jp1,1]), nu[j], nv[j], color=color_list[j])
        ax.axis('equal')

        self.fu = self.fu + coef*Phix
        self.fv = self.fv + coef*Phiy

        self.Phix = self.Phix + Phix
        self.Phiy = self.Phiy + Phiy
    
        if i == n-1:
            lon, lat = transform_coordinates_inverse(xx, yy, xx*0.0, lon0, lat0, gnomonic)
            shape = xx.shape
            flon, flat = transform_vector_components_uv_latlon(lon0, lat0, lon.ravel(), lat.ravel(), self.fu.ravel(), self.fv.ravel(), gnomonic)
            flon = flon.reshape(shape)
            flat = flat.reshape(shape)
             
            ax = self.fig.add_subplot(rows,cols,i+2)
            mag_reconstruct = np.sqrt(self.fu**2 + self.fv**2)
            c = ax.contourf(xx, yy, mag_reconstruct)
            cbar = self.fig.colorbar(c)
            ax.quiver(xx[::N,::N],yy[::N,::N],flon[::N,::N],flat[::N,::N])
            ax.axis('equal')

            fflon, fflat = function(lon, lat)
            fflon[~mask] = np.nan
            fflat[~mask] = np.nan
            ax = self.fig.add_subplot(rows,cols,i+3)
            mag_exact = np.sqrt(fflon**2 + fflat**2)
            c = ax.contourf(xx, yy, mag_exact)
            cbar = self.fig.colorbar(c)
            ax.quiver(xx[::N,::N],yy[::N,::N],fflon[::N,::N],fflat[::N,::N])
            ax.axis('equal')

            ax = self.fig.add_subplot(rows,cols,i+4)
            diff = mag_reconstruct - mag_exact
            vrange = np.nanmax(np.abs(diff)) 
            levels = np.linspace(-vrange, vrange, 10)
            c = ax.contourf(xx, yy, diff, cmap='RdBu', levels=levels)
            cbar = self.fig.colorbar(c)
            ax.quiver(xx[::N,::N],yy[::N,::N],flon[::N,::N] - fflon[::N,::N],flat[::N,::N] - fflat[::N,::N])
            ax.axis('equal')

            ax = self.fig.add_subplot(rows,cols,i+5)
            c = ax.contourf(xx, yy, self.Phix)
            cbar = self.fig.colorbar(c)
            ax.axis('equal')

            ax = self.fig.add_subplot(rows,cols,i+6)
            c = ax.contourf(xx, yy, self.Phiy)
            cbar = self.fig.colorbar(c)
            ax.axis('equal')

            ax = self.fig.add_subplot(rows,cols,i+7)
            mag = np.sqrt(self.Phix**2 + self.Phiy**2)
            c = ax.contourf(xx, yy, mag)
            ax.quiver(xx[::N,::N],yy[::N,::N],self.Phix[::N,::N],self.Phiy[::N,::N])
            cbar = self.fig.colorbar(c)
            ax.axis('equal')

            plt.savefig(f'test_reconstruct_cell_{cell}.png',dpi=500)
            plt.close()

class plotRemap:

    def __init__(self, cell):

        self.target_edge_list = []
        self.cell = cell
        

    def plot_remap(self, cell, sub_edge, n_sub_edges, n_target, iEdge, uVertex_target, vVertex_target, nu_target, nv_target, n, uVertex, vVertex, uv, u, v, u_quad, v_quad, nu, nv):

        if cell != self.cell: 
            return

        if sub_edge == 0:
           fig = plt.figure()
           self.ax = fig.add_subplot(111)
    
        # Plot target cell
        self.ax.scatter(uVertex_target, vVertex_target, marker='o', color='k', alpha=0.5)
        for j in range(n):
            jp1 = (j+1) % n
            self.ax.plot([uVertex_target[j], uVertex_target[jp1]], [vVertex_target[j], vVertex_target[jp1]],color='k', alpha=0.5)

        # Plot target edge normal
        iEdgep1 = (iEdge+1) % n_target
        self.ax.quiver(0.5*(uVertex_target[iEdge]+uVertex_target[iEdgep1]), 0.5*(vVertex_target[iEdge]+vVertex_target[iEdgep1]), nu_target, nv_target)
    
        i = np.arange(n)
        ip1 = (i+1) % n

        color = color_list[sub_edge % len(color_list)]
    
        # Plot source cell edges
        self.ax.plot([uVertex[i], uVertex[ip1]], [vVertex[i], vVertex[ip1]],color=color, alpha=0.5)
    
        # Plot source cell vertices
        #self.ax.scatter(uVertex, vVertex, marker='o', color=color)
    
        # Plot quadrature points along target edge
        self.ax.scatter(u_quad, v_quad, marker='x', color=color)
    
        # Plot edge normalization quadrature points
        #self.ax.scatter(u, v, marker='.', color=color)
    
        # Plot source cell normals
        #self.ax.quiver(0.5*(uv[i,0]+uv[ip1,0]), 0.5*(uv[i,1]+uv[ip1,1]), nu[i], nv[i], color=color)

        if iEdge not in self.target_edge_list:
            self.target_edge_list.append(iEdge)

        if sub_edge == n_sub_edges -1:
            self.ax.axis('equal')
            plt.savefig(f'test_cell_{iEdge}.png',dpi=500)
            plt.close()
    
        if len(self.target_edge_list) == n_target and sub_edge == n_sub_edges-1:
            raise SystemExit(0)
    
