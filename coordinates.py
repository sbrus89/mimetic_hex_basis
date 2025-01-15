import numpy as np

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

def latlon_xyz(lon, lat):

    x = R*np.cos(lat)*np.cos(lon)
    y = R*np.cos(lat)*np.sin(lon)
    z = R*np.sin(lat)

    return x, y, z

def gnomonic_forward(lon, lat, lon0, lat0):

    if true_gnomonic:
        cos_alpha = np.sin(lat0)*np.sin(lat) + np.cos(lat0)*np.cos(lat)*np.cos(lon-lon0)

        u = R*np.cos(lat)*np.sin(lon-lon0)/cos_alpha
        v = R*(np.cos(lat0)*np.sin(lat) - np.sin(lat0)*np.cos(lat)*np.cos(lon-lon0))/cos_alpha
        w = 0.0*u + R

    else:
        # Local tangent plane projection
        x0, y0, z0 = latlon_xyz(lon0, lat0)
        x, y, z = latlon_xyz(lon, lat)

        u = -np.sin(lon0)*(x-x0)              + np.cos(lon0)*(y-y0)
        v = -np.sin(lat0)*np.cos(lon0)*(x-x0) - np.sin(lat0)*np.sin(lon0)*(y-y0) + np.cos(lat0)*(z-z0)
        w =  np.cos(lat0)*np.cos(lon0)*(x-x0) + np.cos(lat0)*np.sin(lon0)*(y-y0) + np.sin(lat0)*(z-z0)

    return u, v, w

def gnomonic_inverse(u, v, w, lon0, lat0):

    # True gnomonic
    if true_gnomonic:
        rho = np.sqrt(u**2 + v**2)
        alpha = np.arctan2(rho,R)

        lat = np.arcsin(np.cos(alpha)*np.sin(lat0) + v*np.sin(alpha)*np.cos(lat0)/rho)
        lon = lon0 + np.arctan2(u*np.sin(alpha), rho*np.cos(lat0)*np.cos(alpha) - v*np.sin(lat0)*np.sin(alpha))

    else:
        # Local tangent plane projection
        x0, y0, z0 = latlon_xyz(lon0, lat0)

        x = -np.sin(lon0)*u - np.sin(lat0)*np.cos(lon0)*v + np.cos(lat0)*np.cos(lon0)*w + x0
        y =  np.cos(lon0)*u - np.sin(lat0)*np.sin(lon0)*v + np.cos(lat0)*np.sin(lon0)*w + y0
        z =                   np.cos(lat0)*v              + np.sin(lat0)*w              + z0

        lat = np.arcsin(z/R)
        lon = np.arctan2(y,x)

    return lon, lat

def latlon_uv_jacobian(u, v, lon0, lat0):

    k = 1.0 + (u**2 + v**2)/R**2

    den = u**2 + (R*np.cos(lat0) - v*np.sin(lat0))**2
    dlondu = (R*np.cos(lat0) - v*np.sin(lat0))/den
    dlondv = u*np.sin(lat0)/den
    dlondw = -u*np.cos(lat0)/den

    den = R**3*k**(3.0/2.0)*np.sqrt(1.0 - (v*np.cos(lat0) + R*np.sin(lat0))**2/(R**2*k))
    dlatdu = -u*(v*np.cos(lat0) + R*np.sin(lat0))/den
    dlatdv = ((R**2 + u**2)*np.cos(lat0) - R*v*np.sin(lat0))/den
    dlatdw = ((u**2 + v**2)*np.sin(lat0) - R*v*np.cos(lat0))/den

    return dlondu, dlondv, dlondw, dlatdu, dlatdv, dlatdw

def xyz_uv_jaconian(lon0, lat0):

    dxdu = -np.sin(lon0)
    dxdv = -np.sin(lat0)*np.cos(lon0)
    dxdw =  np.cos(lat0)*np.cos(lon0)

    dydu =  np.cos(lon0)
    dydv = -np.sin(lat0)*np.sin(lon0)
    dydw =  np.cos(lat0)*np.sin(lon0)

    dzdu = 0.0
    dzdv = np.cos(lat0)
    dzdw = np.sin(lat0)

    return dxdu, dxdv, dydu, dydv, dzdu, dzdv

def gnomonic_integration(lon0, lat0, lon1, lat1, lon2, lat2, t):

    u1, v1, w1 = gnomonic_forward(lon1, lat1, lon0, lat0)
    u2, v2, w2 = gnomonic_forward(lon2, lat2, lon0, lat0)

    if isinstance(u1,np.ndarray):
        u = 0.5*(np.matmul(u2, (1.0+t).T) + np.matmul(u1, (1.0-t).T))
        v = 0.5*(np.matmul(v2, (1.0+t).T) + np.matmul(v1, (1.0-t).T))
        w = 0.5*(np.matmul(w2, (1.0+t).T) + np.matmul(w1, (1.0-t).T))
    else:
        u = 0.5*((1.0+t)*u2 + (1.0-t)*u1)
        v = 0.5*((1.0+t)*v2 + (1.0-t)*v1)
        w = 0.5*((1.0+t)*w2 + (1.0-t)*w1)


    dudt = 0.5*(u2 - u1)
    dvdt = 0.5*(v2 - v1)

    if true_gnomonic:
        lon, lat = gnomonic_inverse(u, v, w, lon0, lat0)
        dxdlat = -R*np.sin(lat)*np.cos(lon)
        dxdlon = -R*np.cos(lat)*np.sin(lon)

        dydlat = -R*np.sin(lat)*np.sin(lon)
        dydlon =  R*np.cos(lat)*np.cos(lon)

        dzdlat = R*np.cos(lat)
        #dzdlon = 0.0

        dlondu, dlondv, dlondw, dlatdu, dlatdv, dlatdw = latlon_uv_jacobian(u, v, lon0, lat0)

        dlatdt = dlatdu*dudt + dlatdv*dvdt
        dlondt = dlondu*dudt + dlondv*dvdt

        dxdt = dxdlat*dlatdt + dxdlon*dlondt
        dydt = dydlat*dlatdt + dydlon*dlondt
        dzdt = dzdlat*dlatdt #+ dzdlon*dlondt

    else:
        # Local tangent plane projection
        dxdu, dxdv, dydu, dydv, dzdu, dzdv = xyz_uv_jaconian(lon0, lat0)

        dxdt = dxdu*dudt + dxdv*dvdt
        dydt = dydu*dudt + dydv*dvdt
        dzdt = dzdu*dudt + dzdv*dvdt

    ds = np.sqrt(np.square(dxdt) + np.square(dydt) + np.square(dzdt))

    return ds, u, v, w

def transform_vector_components(lon0, lat0, lon, lat):

    n = lon.size

    dxdr = np.cos(lat)*np.cos(lon)
    dydr = np.cos(lat)*np.sin(lon)
    dzdr = np.sin(lat)

    dxdlon = -R*np.cos(lat)*np.sin(lon)
    dydlon = R*np.cos(lat)*np.cos(lon)
    dzdlon = 0.0

    dxdlat = -R*np.sin(lat)*np.cos(lon)
    dydlat = -R*np.sin(lat)*np.sin(lon)
    dzdlat = R*np.cos(lat)

    alpha = np.sqrt(dxdr**2 + dydr**2 + dzdr**2)
    beta  = np.sqrt(dxdlon**2 + dydlon**2 + dzdlon**2)
    gamma = np.sqrt(dxdlat**2 + dydlat**2 + dzdlat**2)

    A = np.zeros((3,3,n))
    A[0,0,:] = dxdr/alpha; A[0,1,:] = dxdlon/beta; A[0,2,:] = dxdlat/gamma;
    A[1,0,:] = dydr/alpha; A[1,1,:] = dydlon/beta; A[1,2,:] = dydlat/gamma;
    A[2,0,:] = dzdr/alpha; A[2,1,:] = dzdlon/beta; A[2,2,:] = dzdlat/gamma;

    if true_gnomonic:
        u, v, w = gnomonic_forward(lon, lat, lon0, lat0)
        dlondu, dlondv, dlondw, dlatdu, dlatdv, dlatdw = latlon_uv_jacobian(u, v, lon0, lat0)
        drdw = 1.0

        dxdw = dxdlon*dlondw + dxdlat*dlatdw + dxdr*drdw
        dydw = dydlon*dlondw + dydlat*dlatdw + dydr*drdw
        dzdw = dzdlon*dlondw + dzdlat*dlatdw + dzdr*drdw

        dxdu = dxdlon*dlondu + dxdlat*dlatdu
        dydu = dydlon*dlondu + dydlat*dlatdu
        dzdu = dzdlon*dlondu + dzdlat*dlatdu

        dxdv = dxdlon*dlondv + dxdlat*dlatdv
        dydv = dydlon*dlondv + dydlat*dlatdv
        dzdv = dzdlon*dlondv + dzdlat*dlatdv

    else:
        # Local tangent plane projection
        dxdu, dxdv, dydu, dydv, dzdu, dzdv = xyz_uv_jaconian(lon0, lat0)
        dxdw = 0.0
        dydw = np.cos(lat0)
        dzdw = np.sin(lat0)

    alpha = np.sqrt(dxdu**2 + dydu**2 + dzdu**2)
    beta  = np.sqrt(dxdv**2 + dydv**2 + dzdv**2)
    gamma = np.sqrt(dxdw**2 + dydw**2 + dzdw**2)

    B = np.zeros((3,3,n))
    B[0,0,:] = dxdu/alpha; B[0,1,:] = dxdv/beta; B[0,2,:] = dxdw/gamma;
    B[1,0,:] = dydu/alpha; B[1,1,:] = dydv/beta; B[1,2,:] = dydw/gamma;
    B[2,0,:] = dzdu/alpha; B[2,1,:] = dzdv/beta; B[2,2,:] = dzdw/gamma;

    return A, B

def transform_vector_components_uv_latlon(lon0, lat0, lon, lat, fu, fv):

    n = lon.size

    A, B = transform_vector_components(lon0, lat0, lon, lat)
    A = np.transpose(A, (2, 0, 1))
    B = np.transpose(B, (2, 0, 1))

    fuv = np.zeros((n,3))

    fuv[:,0] = fu
    fuv[:,1] = fv

    b = np.einsum('nij,nj->ni',B,fuv)
    b = np.expand_dims(b,axis=-1)
    f = np.linalg.solve(A,b)

    flon = f[:,1,0]
    flat = f[:,2,0]

    return flon, flat

def transform_vector_components_latlon_uv(lon0, lat0, lon, lat, flon, flat):

    n = lon.size

    A, B = transform_vector_components(lon0, lat0, lon, lat)
    B = np.transpose(B, (2, 0, 1))
    A = np.transpose(A, (2, 0, 1))

    fll = np.zeros((n,3))
    fll[:,1] = flon
    fll[:,2] = flat

    b = np.einsum('nij,nj->ni',A,fll)
    b = np.expand_dims(b,axis=-1)
    f = np.linalg.solve(B,b)

    # Local tangent plane projection
    fu = f[:,0,0]
    fv = f[:,1,0]

    return fu, fv

