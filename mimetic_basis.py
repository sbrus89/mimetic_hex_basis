import numpy as np
import matplotlib.pyplot as plt
import random
from shapely.geometry.polygon import Polygon
from shapely.geometry import Point

np.seterr(divide='ignore', invalid='ignore')

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

def vector_basis(n, i, v, phi, norm_factor=1.0):

    ip1 = (i+1) % n
    ip2 = (i+2) % n

    ## This seems to work for faces, but isn't symmetrical
    #J = v[i,0]*v[ip1,1] - v[ip1,0]*v[i,1]
    #Phix = 1.0/J*v[i,0]*phi[ip1,:,:]
    #Phiy = 1.0/J*v[i,1]*phi[ip1,:,:]

    ## Jacobian factors 
    #J = v[i,0]*v[ip1,1] - v[ip1,0]*v[i,1]
    #Phix = 1.0/J*v[i,0]*(0*phi[i,:,:] + 1)
    #Phiy = 1.0/J*v[i,1]*(0*phi[i,:,:] + 1)

    ## Edge tangents
    #Phix = (v[i,0] - v[i-1,0])*(0*phi[i,:,:] + 1)
    #Phiy = (v[i,1] - v[i-1,1])*(0*phi[i,:,:] + 1)

    ## This works for hex faces, but uses phi i+2
    #Ji   = v[i,0]*v[ip1,1] - v[ip1,0]*v[i,1]
    #Jip1 = v[ip1,0]*v[ip2,1] - v[ip2,0]*v[ip1,1]
    #Phix = 1.0/Ji*v[i,0]*phi[ip1,:,:] + 1.0/Jip1*v[ip1,0]*phi[i,:,:]
    #Phiy = 1.0/Ji*v[i,1]*phi[ip1,:,:] + 1.0/Jip1*v[ip1,1]*phi[i,:,:]

    ## This works for general cases
    Phix = (v[i,0] - v[i-1,0])*phi[i,:,:] + (v[ip1,0] - v[ip2,0])*phi[ip1,:,:]
    Phiy = (v[i,1] - v[i-1,1])*phi[i,:,:] + (v[ip1,1] - v[ip2,1])*phi[ip1,:,:]

    #Phix = phi[ip1,:,:] + phi[i,:,:]
    #Phiy = phi[ip1,:,:] + phi[i,:,:]

    # Edges ?
    # Phix = phix[ip1,:,:]*phi[i,:,:]
    # Phiy = phiy[ip1,:,:]*phi[i,:,:]

    return Phix/norm_factor, Phiy/norm_factor

def wachpress(n,v,xx,yy,method='area'):

    phi = np.zeros((n,xx.shape[0],xx.shape[1]))
    rx = np.zeros((n,xx.shape[0],xx.shape[1]))
    ry = np.zeros((n,xx.shape[0],xx.shape[1]))
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
                wj = C[0]*Ap
        
                if j == i:
                    wi = np.copy(wj)

            ws = ws+wj
    
        phi[i,:,:] = np.divide(wi,ws)

    if method == 'distance':
        return phi, rx, ry
    elif method == 'area':
        return phi

def parameterize_line(t, xi, yi, xip1, yip1):

    tmid = (t[1:] + t[:-1]) / 2.0
    dt = t[1]-t[0] 

    x = (1.0-tmid)*xi + tmid*xip1
    y = (1.0-tmid)*yi + tmid*yip1
    
    nx = yip1 - yi
    ny = -(xip1 - xi)
    mag = np.sqrt(nx**2 + ny**2)
    nx = nx/mag
    ny = ny/mag
    
    ds = np.sqrt(np.square(xip1-xi) + np.square(yip1-yi))*dt

    return x, y, nx, ny, ds

#######################################################################
#######################################################################

n = 6      # Number of polygon sides
eps = 0.0  # Factor to perturb polygon vertices
Nx = 400   # Number of points in x direction 
Ny = 400   # Number of points in y direction
nt = 101   # Number of quadrature points along an edge
N = 15     # Quiver plot subsample factor

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
x = np.linspace(-1.0+eps, 1.0+eps, Nx)
y = np.linspace(-1.0+eps, 1.0+eps, Ny)
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
phi, rx, ry = wachpress(n, v, xx, yy, method='distance')
phi_area = wachpress(n, v, xx, yy, method='area')
phi[:,~mask] = np.nan
phi_area[:,~mask] = np.nan

# Ensure both distance and area methods are equivalent
diff = np.nanmax(np.abs(phi-phi_area))
if diff > 1e-15:
    print("Distance and area methods do not agree")
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
fig,ax = plt.subplots(3,3, figsize=(16,12))
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
fig,ax = plt.subplots(3,3, figsize=(16,12))
ax = ax.reshape(-1)

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
fig,ax = plt.subplots(3,3, figsize=(16,12))
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

fig,ax = plt.subplots(3,3, figsize=(16,12))
ax = ax.reshape(-1)

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
    print(np.sum(tot_integral))
    
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
    
    print(np.nansum(integral))    

