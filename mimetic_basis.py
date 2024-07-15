import numpy as np
import matplotlib.pyplot as plt
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

def vector_basis(n, i, v, phi):

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

    # This works for faces, but uses phi i+2
    Ji   = v[i,0]*v[ip1,1] - v[ip1,0]*v[i,1]
    Jip1 = v[ip1,0]*v[ip2,1] - v[ip2,0]*v[ip1,1]
    Phix = 1.0/Ji*v[i,0]*phi[ip1,:,:] + 1.0/Jip1*v[ip1,0]*phi[i,:,:]
    Phiy = 1.0/Ji*v[i,1]*phi[ip1,:,:] + 1.0/Jip1*v[ip1,1]*phi[i,:,:]

    #Phix = phi[ip1,:,:] + phi[i,:,:]
    #Phiy = phi[ip1,:,:] + phi[i,:,:]

    # Edges ?
    # Phix = phix[ip1,:,:]*phi[i,:,:]
    # Phiy = phiy[ip1,:,:]*phi[i,:,:]

    return Phix, Phiy

def wachpress(n,xx,yy,method='area'):

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

#######################################################################
#######################################################################

n = 6

# Create vertices
drad = 2.0*np.pi/float(n)
v = []
for i in range(n):
    rad = i*drad
    v.append([np.cos(rad), np.sin(rad)])
v = np.asarray(v)

# Generate x,y coordinates
nx = 300 
ny = 300
x = np.linspace(-1.0, 1.0, nx)
y = np.linspace(-1.0, 1.0, ny)
xx, yy = np.meshgrid(x, y)
xy = np.vstack((xx.ravel(), yy.ravel())).T
mask = np.array(xx.ravel(),dtype='bool')

# Find coordinates inside polygon
polygon = Polygon(v)
for pt in range(nx*ny):
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

# Compute Wachpress coordinates
phi, rx, ry = wachpress(n, xx, yy, method='distance')
phi_area = wachpress(n, xx, yy, method='area')
phi[:,~mask] = np.nan
phi_area[:,~mask] = np.nan

# Ensure both distance and area methods are equivalent
print(np.nanmax(np.abs(phi-phi_area)))

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
N = 10
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

    N = 5 
    ax[i].quiver(xx[::N,::N],yy[::N,::N],Phix[::N,::N],Phiy[::N,::N])
    ax[i].quiver(0.5*(v[i,0]+v[ip1,0]), 0.5*(v[i,1]+v[ip1,1]), norm[i,0], norm[i,1])
    ax[i].axis('equal')

fig.tight_layout()
plt.savefig('edge.png',bbox_inches='tight')

# Compute line integral quadrature points
nt = 1000
x = np.zeros((nt-1,n))
y = np.zeros((nt-1,n))
dx = np.zeros((nt-1,n))
dy = np.zeros((nt-1,n))
nx = np.zeros((nt-1,n))
ny = np.zeros((nt-1,n))
ds = np.zeros((nt-1,n))

t = np.linspace(0.0,1.0,nt)
tmid = (t[1:] + t[:-1]) / 2.0
dt = t[1]-t[0] 
for i in range(n):
    ip1 = (i+1) % n

    x[:,i] = (1.0-tmid)*v[i,0] + tmid*v[ip1,0]
    y[:,i] = (1.0-tmid)*v[i,1] + tmid*v[ip1,1]

    xt = (1.0-t)*v[i,0] + t*v[ip1,0]
    yt = (1.0-t)*v[i,1] + t*v[ip1,1]
    
    dx[:,i] = xt[1:] - xt[:-1] 
    dy[:,i] = yt[1:] - yt[:-1] 

    nx[:,i] = norm[i,0]
    ny[:,i] = norm[i,1]

    #ds[:,i] = np.sqrt(np.square(dx[:,i]) + np.square(dy[:,i]))
    ds[:,i] = np.sqrt(np.square(v[i,0]) + np.square(v[i,1]))*dt

# Compute line integrals
phi = wachpress(n, x, y, method='area')
for i in range(n):

    ip1 = (i+1) % n
    Phix, Phiy = vector_basis(n, i, v, phi)

    integral = np.sum((Phix*nx + Phiy*ny)*ds, axis=0)
    integral[integral < 1e-15] = 0.0
    print(integral)
