import numpy as np

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

    vix = (v[i,0] - v[i-1,0])
    viy = (v[i,1] - v[i-1,1])
    vi_mag = np.sqrt(vix**2 + viy**1)
    #den = (v[i,0]-v[i-1,0])*(v[i,1]-v[ip1,1]) - (v[i,1]-v[i-1,1])*(v[i,0]-v[ip1,0])
    #a = ((v[i,1]-v[ip1,1]) + (v[ip1,0]-v[i,0]))/den

    vip1x = (v[ip1,0] - v[ip2,0])
    vip1y = (v[ip1,1] - v[ip2,1])
    vip1_mag = np.sqrt(vip1x**2 + vip1y**2)
    #den = (v[ip1,0]-v[i,0])*(v[ip1,1]-v[ip2,1]) - (v[ip1,1]-v[i,1])*(v[ip1,0]-v[ip2,0])
    #b = ((v[i,1]-v[ip1,1]) + (v[ip1,0]-v[i,0]))/den

    A = np.zeros((2,2))
    f = np.ones((2,1))

    #v1_mag = np.sqrt((v[i,0]-v[i-1,0])**2 + (v[i,1]-v[i-1,1])**2)
    #v2_mag = np.sqrt((v[i,0]-v[ip1,0])**2 + (v[i,1]-v[ip1,1])**2)
    v1_mag = 1.0
    v2_mag = 1.0
    A[0,0] = (v[i,0]-v[i-1,0])/v1_mag; A[0,1] = (v[i,0]-v[ip1,0])/v2_mag;
    A[1,0] = (v[i,1]-v[i-1,1])/v1_mag; A[1,1] = (v[i,1]-v[ip1,1])/v2_mag;

    x = np.linalg.solve(A,f)
    a = x[0]

    #v1_mag = np.sqrt((v[ip1,0]-v[i,0])**2 + (v[ip1,1]-v[i,1])**2)
    #v2_mag = np.sqrt((v[ip1,0]-v[ip2,0])**2 + (v[ip1,1]-v[ip2,1])**2)
    v1_mag = 1.0
    v2_mag = 1.0
    A[0,0] = (v[ip1,0]-v[i,0])/v1_mag; A[0,1] = (v[ip1,0]-v[ip2,0])/v2_mag;
    A[1,0] = (v[ip1,1]-v[i,1])/v1_mag; A[1,1] = (v[ip1,1]-v[ip2,1])/v2_mag;

    x = np.linalg.solve(A,f)
    b = x[1]

    #a = 1.0
    #b = 1.0
     
    Phix = a*vix*phi[i,:,:] + b*vip1x*phi[ip1,:,:]
    Phiy = a*viy*phi[i,:,:] + b*vip1y*phi[ip1,:,:]

    return Phix/norm_factor, Phiy/norm_factor

def vector_basis_test(n, i, v, phi, t, w_gp, ds, nu, nv, norm_factor=1.0):

    A = np.zeros((2*n,n))
    f = np.zeros((2*n))
    M = np.zeros((n,n))
    g = np.zeros((n))
    AA = np.zeros((2*n,2*n))
    
    alpha = np.zeros((n))
    beta = np.zeros((n))
    for j in range(n):

        jp1 = (j+1) % n
        jp2 = (j+2) % n
        jm1 = (j-1) % n

        phi1_int = np.sum(w_gp*(0.5*(1.0-t[:,0]))*ds[j,:])
        phi2_int = np.sum(w_gp*(0.5*(1.0+t[:,0]))*ds[j,:])
        den =       phi1_int*((v[j  ,0]-v[j-1,0])*nu[j] + (v[j,  1]-v[j-1,1])*nv[j])
        alpha[j] = -phi2_int*((v[jp1,0]-v[jp2,0])*nu[j] + (v[jp1,1]-v[jp2,1])*nv[j])/den
        beta[j] = 1.0/den

        A[2*j,j]     = alpha[j]*(v[j,0]-v[j-1,0]) 
        A[2*j,jm1]   =          (v[j,0]-v[jp1,0]) 
        f[2*j] =   1.0 - beta[j]*(v[j,0]-v[j-1,0])
        A[2*j+1,j]   = alpha[j]*(v[j,1]-v[j-1,1])
        A[2*j+1,jm1] =          (v[j,1]-v[jp1,1])
        f[2*j+1] = 1.0 - beta[j]*(v[j,1]-v[j-1,1])

        M[j,j] = alpha[j]*(v[j,0]-v[j-1,0])
        M[j,jm1] =  (v[j,0]-v[jp1,0])
        g[j] = 1.0 - beta[j]*(v[j,0]-v[j-1,0])

        #AA[j,  j]     = (v[j,0]-v[j-1,0]) 
        #AA[j,  n+jm1] = (v[j,0]-v[jp1,0]) 
        #AA[n+j,j]     = (v[j,1]-v[j-1,1])
        #AA[n+j,n+jm1] = (v[j,1]-v[jp1,1])
        #f[j] = 1.0
        #f[n+j] = 1.0
        

    #b = np.linalg.lstsq(A,f)[0]
    b = np.linalg.solve(M,g)
    #a = np.linalg.solve(AA,f)
    #b = a[n:]

    a = alpha*b + beta  

    ip1 = (i+1) % n     
    ip2 = (i+2) % n     
    vix = (v[i,0] - v[i-1,0])
    viy = (v[i,1] - v[i-1,1])
    vip1x = (v[ip1,0] - v[ip2,0])
    vip1y = (v[ip1,1] - v[ip2,1])
    Phix = a[i]*vix*phi[i,:,:] + b[i]*vip1x*phi[ip1,:,:]
    Phiy = a[i]*viy*phi[i,:,:] + b[i]*vip1y*phi[ip1,:,:]

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

    norm = np.zeros_like(v)
    for j in range(n):
      jp1 = (j+1) % n
      norm[j,0] = v[jp1,1] - v[j,1]
      norm[j,1] = -(v[jp1,0] - v[j,0])
      norm[j,:] = norm[j,:] / np.sqrt(np.square(norm[j,0]) + np.square(norm[j,1]))

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

