import numpy as np
import matplotlib.pyplot as plt
import random
from shapely.geometry.polygon import Polygon
from shapely.geometry import Point
import netCDF4 as nc4

np.seterr(divide='ignore', invalid='ignore')
 
R = 6356.0*1000.0

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
                wj = C[0]*Ap
        
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
    dzdlon = 0.0

    dlondu, dlondv, dlatdu, dlatdv = latlon_uv_jacobian(u, v, lon0, lat0)

    dlatdt = dlatdu*dudt + dlatdv*dvdt 
    dlondt = dlondu*dudt + dlondv*dvdt

    dxdt = dxdlat*dlatdt + dxdlon*dlondt
    dydt = dydlat*dlatdt + dydlon*dlondt
    dzdt = dzdlat*dlatdt + dzdlon*dlondt

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
t, w = np.polynomial.legendre.leggauss(100)

ds, u, v =  gnomonic_integration(lon0, lat0, lon1, lat1, lon2, lat2, t)
L = np.sum(ds*w)
print(L/1000.0)

hav = np.sin(0.5*(lat2-lat1))**2 + np.cos(lat1)*np.cos(lat2)*np.sin(0.5*(lon2-lon1))**2
d = 2.0*R*np.arcsin(np.sqrt(hav))
print(d/1000.0)


############################################
# Remap MPAS field
############################################
mesh_filename = 'soma_32km_mpas_mesh_with_rbf_weights.nc'
nc_mesh = nc4.Dataset(mesh_filename, 'r+')

# read in mpas mesh coordinates
nc_vars = nc_mesh.variables.keys()
lonVertex = nc_mesh.variables['lonVertex'][:]
latVertex = nc_mesh.variables['latVertex'][:]
lonCell = nc_mesh.variables['lonCell'][:]
latCell = nc_mesh.variables['latCell'][:]
lonEdge = nc_mesh.variables['lonEdge'][:]
latEdge = nc_mesh.variables['latEdge'][:]

# 0,360 -> -180,180 adjustment
lonCell[lonCell > np.pi] = lonCell[lonCell > np.pi] - 2.0*np.pi
lonVertex[lonVertex > np.pi] = lonVertex[lonVertex > np.pi] - 2.0*np.pi

# read in mesh connectivity information
verticesOnEdge = nc_mesh.variables['verticesOnEdge'][:]
verticesOnCell = nc_mesh.variables['verticesOnCell'][:]
edgesOnCell = nc_mesh.variables['edgesOnCell'][:]
nEdgesOnCell = nc_mesh.variables['nEdgesOnCell'][:]
cellsOnEdge = nc_mesh.variables['cellsOnEdge'][:]
edgeSignOnCell = nc_mesh.variables['edgeSignOnCell'][:]
angleEdge = nc_mesh.variables['angleEdge'][:]
dvEdge = nc_mesh.variables['dvEdge'][:]

# read in fields
barotropicThicknessFlux = np.squeeze(nc_mesh.variables['barotropicThicknessFlux'][:])
barotropicThicknessFluxZonal = np.squeeze(nc_mesh.variables['barotropicThicknessFluxZonal'][:])
barotropicThicknessFluxMeridional = np.squeeze(nc_mesh.variables['barotropicThicknessFluxMeridional'][:])
nc_mesh.close()

# get number of cells and edges
nCells = lonCell.size
print(nCells)
nEdges = barotropicThicknessFlux.size
print(nEdges)


# Set up figure
rows = 3
cols = 3
fig = plt.figure(figsize=(18, 4.5*nrows))
k = 1
vmin = -2.5
vmax = 3.5
levels = np.linspace(vmin, vmax, 10)
mlevels = np.linspace(0, vmax, 10)

# Plot original RBF field
ax = fig.add_subplot(rows, cols, k)
cf = ax.tricontourf(lonCell, latCell, barotropicThicknessFluxZonal, levels=levels, extend='both')
ax.set_title('Zonal component')
ax.set_ylabel('RFB')
fig.colorbar(cf, ax=ax)
k = k + 1

ax = fig.add_subplot(rows, cols, k)
cf = ax.tricontourf(lonCell, latCell, barotropicThicknessFluxMeridional, levels=levels, extend='both')
ax.set_title('Meridonal component')
fig.colorbar(cf, ax=ax)
k = k + 1

ax = fig.add_subplot(rows, cols, k)
mag_rbf = np.sqrt(barotropicThicknessFluxZonal**2 + barotropicThicknessFluxMeridional**2)
cf = ax.tricontourf(lonCell, latCell, mag_rbf, levels=mlevels, extend='max')
ax.set_title('Magnitude')
fig.colorbar(cf, ax=ax)
k = k + 1

# Plot u, v component of RBF field
#bu, bv = transform_vector_components_latlon_uv(lon0, lat0, lonCell, latCell, barotropicThicknessFluxZonal, barotropicThicknessFluxMeridional)
# 
#ax = fig.add_subplot(rows, cols, k)
#cf = ax.tricontourf(lonCell, latCell, bu)
#fig.colorbar(cf, ax=ax)
#k = k + 1
#
#ax = fig.add_subplot(rows, cols, k)
#cf = ax.tricontourf(lonCell, latCell, bv)
#fig.colorbar(cf, ax=ax)
#k = k + 1
#
#ax = fig.add_subplot(rows, cols, k)
#mag2 = np.sqrt(bu**2 + bv**2)
#cf = ax.tricontourf(lonCell, latCell, mag2)
#fig.colorbar(cf, ax=ax)
#k = k + 1

#print(np.max(np.abs(mag1-mag2)))


# quadrature points for computing the edge integral for the basis function normalization
t, w = np.polynomial.legendre.leggauss(100)

flon = np.zeros((nCells))
flat = np.zeros((nCells))
fuu = np.zeros((nCells))
fvv = np.zeros((nCells))
#nCells = 0
for cell in range(nCells):
    print(cell)

    n = nEdgesOnCell[cell]
    vertices = verticesOnCell[cell, 0:n] - 1
    vertices = np.roll(vertices, 1) # this is important to account for how mpas defines vertices on an edge

    # gnomonic projection center
    #lon0 = lonCell[cell]
    #lat0 = latCell[cell]
    lon0 = 0.5*(np.max(lonCell) + np.min(lonCell))
    lat0 = 0.5*(np.max(latCell) + np.min(latCell))

    # Cell vertex coordinates in u, v
    uVertex, vVertex = gnomonic_forward(lonVertex[vertices], latVertex[vertices], lon0, lat0)
    uv = np.vstack((uVertex, vVertex)).T # package for call to watchpress, vector_basis etc

    # Cell center coordinates in u,v
    uCell, vCell = gnomonic_forward(lonCell[cell], latCell[cell], lon0, lat0)
    uCell = np.expand_dims(np.asarray([uCell]), axis=0) # put single value in an array for call to watchpress
    vCell = np.expand_dims(np.asarray([vCell]), axis=0)

    fu = np.zeros(uCell.shape)
    fv = np.zeros(vCell.shape)
    for i in range(n):
        ip1 = (i+1) % n
        edge = edgesOnCell[cell,i] - 1

        # lon, lat coordinates of edge vertices
        lon1 = lonVertex[vertices][i]
        lat1 = latVertex[vertices][i]
        lon2 = lonVertex[vertices][ip1]
        lat2 = latVertex[vertices][ip1]

        # u, v quadrature points and ds transformation factor for integral
        ds, u, v =  gnomonic_integration(lon0, lat0, lon1, lat1, lon2, lat2, t)
        u = np.expand_dims(u, axis=1)
        v = np.expand_dims(v, axis=1)

        # evaluate basis functions at quadrature points
        phi = wachpress(n, uv, u, v, method='area')
        Phiu, Phiv = vector_basis(n, i, uv, phi, norm_factor=1.0)
        Phiu = np.squeeze(Phiu)
        Phiv = np.squeeze(Phiv)

        # compute integral over edge for basis function normalization factor      
        nu, nv = edge_normal(uv[i,0] ,uv[i,1], uv[ip1,0], uv[ip1,1])
        integral = -np.sum(w*(Phiu*nu + Phiv*nv)*ds, axis=0)

        # compute normalized basis functions at cell centers 
        phi = wachpress(n, uv, uCell, vCell, method='area')
        Phiu, Phiv = vector_basis(n, i, uv, phi, norm_factor=integral)

        # compute reconstruction at cell center
        fu = fu + edgeSignOnCell[cell, i]*dvEdge[edge]*barotropicThicknessFlux[edge]*Phiu
        fv = fv + edgeSignOnCell[cell, i]*dvEdge[edge]*barotropicThicknessFlux[edge]*Phiv

    fuu[cell] = fu
    fvv[cell] = fv

    # compute lon lat vector components
    flon[cell], flat[cell] = transform_vector_components_uv_latlon(lon0, lat0, lonCell[cell], latCell[cell], fu, fv)

# Plot lat lon components of reconstructed field
ax = fig.add_subplot(rows, cols, k)
cf = ax.tricontourf(lonCell, latCell, flon, levels=levels, extend='both')
fig.colorbar(cf, ax=ax)
ax.set_ylabel('Mimetic interpolation')
k = k + 1

ax = fig.add_subplot(rows, cols, k)
cf = ax.tricontourf(lonCell, latCell, flat, levels=levels, extend='both')
fig.colorbar(cf, ax=ax)
k = k + 1

ax = fig.add_subplot(rows, cols, k)
mag = np.sqrt(flon**2 + flat**2)
cf = ax.tricontourf(lonCell, latCell, mag, levels=mlevels, extend='max')
fig.colorbar(cf, ax=ax)
k = k + 1

# Plot u, v components of reconstructed field
#ax = fig.add_subplot(rows, cols, k)
#cf = ax.tricontourf(lonCell, latCell, fuu)
#fig.colorbar(cf, ax=ax)
#k = k + 1
#
#ax = fig.add_subplot(rows, cols, k)
#cf = ax.tricontourf(lonCell, latCell, fvv)
#fig.colorbar(cf, ax=ax)
#k = k + 1
#
#ax = fig.add_subplot(rows, cols, k)
#mag = np.sqrt(fuu**2 + fvv**2)
#cf = ax.tricontourf(lonCell, latCell, mag)
#fig.colorbar(cf, ax=ax)
#k = k + 1


# Plot differences 
ax = fig.add_subplot(rows, cols, k)
diff = flon-barotropicThicknessFluxZonal
vrange = np.max(np.abs(diff)) 
levels = np.linspace(-vrange, vrange, 10)
cf = ax.tricontourf(lonCell, latCell, diff, cmap='RdBu', levels=levels) 
fig.colorbar(cf, ax=ax)
ax.set_ylabel('Mimetic-RBF')
k = k + 1

ax = fig.add_subplot(rows, cols, k)
diff = flat-barotropicThicknessFluxMeridional
vrange = np.max(np.abs(diff)) 
levels = np.linspace(-vrange, vrange, 10)
cf = ax.tricontourf(lonCell, latCell, diff, cmap='RdBu', levels=levels)
fig.colorbar(cf, ax=ax)
k = k + 1

ax = fig.add_subplot(rows, cols, k)
mag = np.sqrt(flon**2 + flat**2)
diff = mag-mag_rbf
vrange = np.max(np.abs(diff)) 
levels = np.linspace(-vrange, vrange, 10)
cf = ax.tricontourf(lonCell, latCell, diff, cmap='RdBu', levels=levels)
fig.colorbar(cf, ax=ax)
k = k + 1

plt.savefig('field.png')
plt.close()
