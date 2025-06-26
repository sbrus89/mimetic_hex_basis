import numpy as np
import matplotlib.pyplot as plt
import scipy

from coordinates import edge_normal

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
    den = (v[i,0]-v[i-1,0])*(v[i,1]-v[ip1,1]) - (v[i,1]-v[i-1,1])*(v[i,0]-v[ip1,0])
    a = ((v[i,1]-v[ip1,1]) + (v[ip1,0]-v[i,0]))/den

    vip1x = (v[ip1,0] - v[ip2,0])
    vip1y = (v[ip1,1] - v[ip2,1])
    vip1_mag = np.sqrt(vip1x**2 + vip1y**2)
    den = (v[ip1,0]-v[i,0])*(v[ip1,1]-v[ip2,1]) - (v[ip1,1]-v[i,1])*(v[ip1,0]-v[ip2,0])
    b = ((v[i,1]-v[ip1,1]) + (v[ip1,0]-v[i,0]))/den

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

    a = 1.0
    b = 1.0
     
    Phix = a*vix*phi[i,:,:] + b*vip1x*phi[ip1,:,:]
    Phiy = a*viy*phi[i,:,:] + b*vip1y*phi[ip1,:,:]

    return Phix/norm_factor, Phiy/norm_factor

def vector_basis_mimetic(n, i, v, phi, coef=False, norm=False):

    A = np.zeros((12*n,12*n))
    f = np.zeros((12*n,))
    a = np.zeros((12*n,))

    ip1 = (i+1) % n     
    ip2 = (i+2) % n     
    ip3 = (i+3) % n     
    im1 = (i-1) % n
    im2 = (i-2) % n

    #nui,   nvi   = edge_normal(v[i,  0] ,v[i,  1], v[ip1,0], v[ip1,1])
    #nuim2, nvim2 = edge_normal(v[im2,0] ,v[im2,1], v[im1,0], v[im1,1])
    #nuim1, nvim1 = edge_normal(v[im1,0] ,v[im1,1], v[i,  0], v[i,  1])
    #nuip1, nvip1 = edge_normal(v[ip1,0] ,v[ip1,1], v[ip2,0], v[ip2,1])
    #nuip2, nvip2 = edge_normal(v[ip2,0] ,v[ip2,1], v[ip3,0], v[ip3,1])
   
    for j in range(n): 
        jp1 = (j+1) % n     
        jp2 = (j+2) % n     
        jp3 = (j+3) % n     
        #jm1 = (j-1) % n
        #jm2 = (j-2) % n
        jm1 = j-1
        jm2 = j-2

        nuj,   nvj   = edge_normal(v[j,  0] ,v[j,  1], v[jp1,0], v[jp1,1])
        nujm2, nvjm2 = edge_normal(v[jm2,0] ,v[jm2,1], v[jm1,0], v[jm1,1])
        nujm1, nvjm1 = edge_normal(v[jm1,0] ,v[jm1,1], v[j,  0], v[j,  1])
        nujp1, nvjp1 = edge_normal(v[jp1,0] ,v[jp1,1], v[jp2,0], v[jp2,1])
        nujp2, nvjp2 = edge_normal(v[jp2,0] ,v[jp2,1], v[jp3,0], v[jp3,1])
        nujp3, nvjp3 = edge_normal(v[jp3,0] ,v[jp3,1], v[jm2,0], v[jm2,1])

        uj = (v[jp1,0] - v[j,0])
        vj = (v[jp1,1] - v[j,1])
        Lj = np.sqrt(uj**2 + vj**2)

        intu_j,   intv_j   = integrate_cell(j,  v)
        intu_jp1, intv_jp1 = integrate_cell(jp1,v)
        intu_jp2, intv_jp2 = integrate_cell(jp2,v)
        intu_jp3, intv_jp3 = integrate_cell(jp3,v)
        intu_jm1, intv_jm1 = integrate_cell(jm1,v)
        intu_jm2, intv_jm2 = integrate_cell(jm2,v)

        aui = 12*j+0
        bui = 12*j+1
        cui = 12*j+2
        dui = 12*j+3
        eui = 12*j+4
        fui = 12*j+5
        avi = 12*j+6
        bvi = 12*j+7
        cvi = 12*j+8
        dvi = 12*j+9
        evi = 12*j+10
        fvi = 12*j+11

        auim2 = 12*jm2+0
        buim2 = 12*jm2+1
        cuim2 = 12*jm2+2
        duim2 = 12*jm2+3
        euim2 = 12*jm2+4
        fuim2 = 12*jm2+5
        avim2 = 12*jm2+6
        bvim2 = 12*jm2+7
        cvim2 = 12*jm2+8
        dvim2 = 12*jm2+9
        evim2 = 12*jm2+10
        fvim2 = 12*jm2+11

        auim1 = 12*jm1+0
        buim1 = 12*jm1+1
        cuim1 = 12*jm1+2
        duim1 = 12*jm1+3
        euim1 = 12*jm1+4
        fuim1 = 12*jm1+5
        avim1 = 12*jm1+6
        bvim1 = 12*jm1+7
        cvim1 = 12*jm1+8
        dvim1 = 12*jm1+9
        evim1 = 12*jm1+10
        fvim1 = 12*jm1+11

        auip1 = 12*jp1+0
        buip1 = 12*jp1+1
        cuip1 = 12*jp1+2
        duip1 = 12*jp1+3
        euip1 = 12*jp1+4
        fuip1 = 12*jp1+5
        avip1 = 12*jp1+6
        bvip1 = 12*jp1+7
        cvip1 = 12*jp1+8
        dvip1 = 12*jp1+9
        evip1 = 12*jp1+10
        fvip1 = 12*jp1+11

        auip2 = 12*jp2+0
        buip2 = 12*jp2+1
        cuip2 = 12*jp2+2
        duip2 = 12*jp2+3
        euip2 = 12*jp2+4
        fuip2 = 12*jp2+5
        avip2 = 12*jp2+6
        bvip2 = 12*jp2+7
        cvip2 = 12*jp2+8
        dvip2 = 12*jp2+9
        evip2 = 12*jp2+10
        fvip2 = 12*jp2+11

        auip3 = 12*jp3+0
        buip3 = 12*jp3+1
        cuip3 = 12*jp3+2
        duip3 = 12*jp3+3
        euip3 = 12*jp3+4
        fuip3 = 12*jp3+5
        avip3 = 12*jp3+6
        bvip3 = 12*jp3+7
        cvip3 = 12*jp3+8
        dvip3 = 12*jp3+9
        evip3 = 12*jp3+10
        fvip3 = 12*jp3+11

        # u grad 
        #A[12*j+0,aui]   = 1.0
        #A[12*j+0,buim1] = 1.0
        #A[12*j+0,cuip1] = 1.0
        #A[12*j+0,duim2] = 1.0
        #A[12*j+0,euip2] = 1.0
        #A[12*j+0,fuip3] = 1.0

        #f[12*j+0] = 1.0

        # v grad
        #A[12*j+1,avi]   = 1.0
        #A[12*j+1,bvim1] = 1.0
        #A[12*j+1,cvip1] = 1.0
        #A[12*j+1,dvim2] = 1.0 
        #A[12*j+1,evip2] = 1.0 
        #A[12*j+1,fvip3] = 1.0 

        #f[12*j+1] = 1.0

        # edge j integral = 1
        A[12*j+2,aui] = nuj
        A[12*j+2,bui] = nuj
        A[12*j+2,avi] = nvj
        A[12*j+2,bvi] = nvj

        f[12*j+2] = 2.0/Lj

        # area integral of divergence
        A[12*j+3,aui] = intu_j 
        A[12*j+3,bui] = intu_jp1
        A[12*j+3,cui] = intu_jm1
        A[12*j+3,dui] = intu_jp2
        A[12*j+3,eui] = intu_jm2
        A[12*j+3,fui] = intu_jp3
        A[12*j+3,avi] = intv_j
        A[12*j+3,bvi] = intv_jp1
        A[12*j+3,cvi] = intv_jm1
        A[12*j+3,dvi] = intv_jp2
        A[12*j+3,evi] = intv_jm2
        A[12*j+3,fvi] = intv_jp3

        f[12*j+3] = 1.0

        # edge j-2 integral = 0
        A[12*j+4,cui] = nujm2 
        A[12*j+4,eui] = nujm2
        A[12*j+4,cvi] = nvjm2
        A[12*j+4,evi] = nvjm2

        # edge j-1 integral = 0
        A[12*j+5,aui] = nujm1
        A[12*j+5,cui] = nujm1
        A[12*j+5,avi] = nvjm1
        A[12*j+5,cvi] = nvjm1

        # edge j+1 integral = 0
        A[12*j+6,bui] = nujp1
        A[12*j+6,dui] = nujp1
        A[12*j+6,bvi] = nvjp1
        A[12*j+6,dvi] = nvjp1

        # edge j+2 integral = 0
        A[12*j+7,dui] = nujp2  
        A[12*j+7,fui] = nujp2
        A[12*j+7,dvi] = nvjp2
        A[12*j+7,fvi] = nvjp2

        # edge j+3 integral = 0
        A[12*j+8,eui] = nujp3  
        A[12*j+8,fui] = nujp3
        A[12*j+8,evi] = nvjp3
        A[12*j+8,fvi] = nvjp3

        A[12*j+0, aui] = 1.0
        A[12*j+1, bui] = 1.0 
        A[12*j+9, fui] = 1.0
        A[12*j+10,eui] = 1.0
        A[12*j+11,cui] = 1.0

    coeffs = ['au', 'bu', 'cu', 'du', 'eu', 'fu', 'av', 'bv', 'cv', 'dv', 'ev', 'fv']
    for k,row in enumerate(A):
        if k==0:
             print('     ', end=f' ')
             for m in range(n):
                 for coef in coeffs:
                     print(f' {coef}{m} ', end=' ') 
             print()
             print('     ', end=f' ')
             for l in range(len(row)):
                 print('{:5d}'.format(l), end=f' ')
             print()
        print('{:5d}'.format(k), end=f' ')
        for element in row:
            if element == 0:
                print('  -  ', end=f' ')  # Replace zero with dash and space
            else:
                print('{:5.2f}'.format(element), end=f' ') # Print the number and space
        print() # Newline after each row

    try:
        a = np.linalg.solve(A,f)
        #a, info = scipy.sparse.linalg.gmres(A,f)
        #print(info)
        #a,res,rank,s = np.linalg.lstsq(A,f)
    except np.linalg.LinAlgError as e: 
        print(f"Error with solve: {e}")
        a,res,rank,s = np.linalg.lstsq(A,f)
        print(rank)
    print(np.linalg.cond(A))
    #print(a)
    print()

    #Phix = a[8*i+0]*phi[i,:,:] + a[8*i+1]*phi[ip1,:,:] + a[8*i+2]*phi[im1,:,:] + a[8*i+3]*phi[ip2,:,:]
    #Phiy = a[8*i+4]*phi[i,:,:] + a[8*i+5]*phi[ip1,:,:] + a[8*i+6]*phi[im1,:,:] + a[8*i+7]*phi[ip2,:,:]

    Phix = a[12*i+0]*phi[i,:,:] + a[12*i+1]*phi[ip1,:,:] + a[12*i+2]*phi[im1,:,:] + a[12*i+3]*phi[ip2,:,:] + a[12*i+4]*phi[im2,:,:]  + a[12*i+5]*phi[ip3,:,:]
    Phiy = a[12*i+6]*phi[i,:,:] + a[12*i+7]*phi[ip1,:,:] + a[12*i+8]*phi[im1,:,:] + a[12*i+9]*phi[ip2,:,:] + a[12*i+10]*phi[im2,:,:] + a[12*i+11]*phi[ip3,:,:]

    return Phix, Phiy

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

        AA[j,  j]     = (v[j,0]-v[j-1,0]) 
        AA[j,  n+jm1] = (v[j,0]-v[jp1,0]) 
        AA[n+j,j]     = (v[j,1]-v[j-1,1])
        AA[n+j,n+jm1] = (v[j,1]-v[jp1,1])
        f[j] = 1.0
        f[n+j] = 1.0
        

    #b = np.linalg.lstsq(A,f)[0]
    #b = np.linalg.solve(M,g)
    #a = alpha*b + beta  

    a = np.linalg.solve(AA,f)
    b = a[n:]


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

def grad_wachpress(phi,rx,ry):

    n = phi.shape[0]

    sx = np.zeros_like(phi[0])
    sy = np.zeros_like(phi[0])
    for j in range(n):
        sx = sx + phi[j,:,:]*rx[j,:,:]
        sy = sy + phi[j,:,:]*ry[j,:,:]

    phix = np.zeros_like(phi)
    phiy = np.zeros_like(phi)
    for i in range(n):
        phix[i,:,:] = phi[i,:,:]*(rx[i,:,:] - sx) 
        phiy[i,:,:] = phi[i,:,:]*(ry[i,:,:] - sy) 

    return phix, phiy

def area_quadrature():

    qpta = 58
    qpta = np.zeros((58,3))
    qpta[ 0, :] = [ -3.333333333333330e-01,  -3.333333333333330e-01, 9.242120232184900e-02 ] ; 
    qpta[ 1, :] = [ -1.579643695921300e-02,  -1.579643695921300e-02, 2.799165273472400e-02 ] ; 
    qpta[ 2, :] = [ -9.684071260815740e-01,  -1.579643695921300e-02, 2.799165273472400e-02 ] ; 
    qpta[ 3, :] = [ -1.579643695921300e-02,  -9.684071260815740e-01, 2.799165273472400e-02 ] ; 
    qpta[ 4, :] = [ -8.737629904678500e-02,  -8.737629904678500e-02, 3.640676359642400e-02 ] ; 
    qpta[ 5, :] = [ -8.252474019064310e-01,  -8.737629904678500e-02, 3.640676359642400e-02 ] ; 
    qpta[ 6, :] = [ -8.737629904678500e-02,  -8.252474019064310e-01, 3.640676359642400e-02 ] ; 
    qpta[ 7, :] = [ -6.404980985340450e-01,  -6.404980985340450e-01, 6.272942332725100e-02 ] ; 
    qpta[ 8, :] = [ 2.809961970680910e-01,  -6.404980985340450e-01, 6.272942332725100e-02 ] ; 
    qpta[ 9, :] = [ -6.404980985340450e-01,  2.809961970680910e-01, 6.272942332725100e-02 ] ; 
    qpta[ 10, :] = [ -8.282115918785490e-01,  -8.282115918785490e-01, 3.221717019267300e-02 ] ; 
    qpta[ 11, :] = [ 6.564231837570990e-01,  -8.282115918785490e-01, 3.221717019267300e-02 ] ; 
    qpta[ 12, :] = [ -8.282115918785490e-01,  6.564231837570990e-01, 3.221717019267300e-02 ] ; 
    qpta[ 13, :] = [ -9.778827672956500e-01,  -9.778827672956500e-01, 3.676792575648000e-03 ] ; 
    qpta[ 14, :] = [ 9.557655345912990e-01,  -9.778827672956500e-01, 3.676792575648000e-03 ] ; 
    qpta[ 15, :] = [ -9.778827672956500e-01,  9.557655345912990e-01, 3.676792575648000e-03 ] ; 
    qpta[ 16, :] = [ -6.485948496992530e-01,  6.195349544551850e-01, 1.907568631330700e-02 ] ; 
    qpta[ 17, :] = [ 6.195349544551850e-01,  -6.485948496992530e-01, 1.907568631330700e-02 ] ; 
    qpta[ 18, :] = [ -9.709401047559330e-01,  6.195349544551850e-01, 1.907568631330700e-02 ] ; 
    qpta[ 19, :] = [ 6.195349544551850e-01,  -9.709401047559330e-01, 1.907568631330700e-02 ] ; 
    qpta[ 20, :] = [ -9.709401047559330e-01,  -6.485948496992530e-01, 1.907568631330700e-02 ] ; 
    qpta[ 21, :] = [ -6.485948496992530e-01,  -9.709401047559330e-01, 1.907568631330700e-02 ] ; 
    qpta[ 22, :] = [ -3.550890581610320e-01,  3.247034484638380e-01, 2.511133672332300e-02 ] ; 
    qpta[ 23, :] = [ 3.247034484638380e-01,  -3.550890581610320e-01, 2.511133672332300e-02 ] ; 
    qpta[ 24, :] = [ -9.696143903028061e-01,  3.247034484638380e-01, 2.511133672332300e-02 ] ; 
    qpta[ 25, :] = [ 3.247034484638380e-01,  -9.696143903028061e-01, 2.511133672332300e-02 ] ; 
    qpta[ 26, :] = [ -9.696143903028061e-01,  -3.550890581610320e-01, 2.511133672332300e-02 ] ; 
    qpta[ 27, :] = [ -3.550890581610320e-01,  -9.696143903028061e-01, 2.511133672332300e-02 ] ; 
    qpta[ 28, :] = [ -8.630827251863440e-01,  8.298101651740040e-01, 1.413451074304000e-02 ] ; 
    qpta[ 29, :] = [ 8.298101651740040e-01,  -8.630827251863440e-01, 1.413451074304000e-02 ] ; 
    qpta[ 30, :] = [ -9.667274399876600e-01,  8.298101651740040e-01, 1.413451074304000e-02 ] ; 
    qpta[ 31, :] = [ 8.298101651740040e-01,  -9.667274399876600e-01, 1.413451074304000e-02 ] ; 
    qpta[ 32, :] = [ -9.667274399876600e-01,  -8.630827251863440e-01, 1.413451074304000e-02 ] ; 
    qpta[ 33, :] = [ -8.630827251863440e-01,  -9.667274399876600e-01, 1.413451074304000e-02 ] ; 
    qpta[ 34, :] = [ -6.501104711514279e-01,  5.119105251906040e-01, 2.525554367249600e-02 ] ; 
    qpta[ 35, :] = [ 5.119105251906040e-01,  -6.501104711514279e-01, 2.525554367249600e-02 ] ; 
    qpta[ 36, :] = [ -8.618000540391760e-01,  5.119105251906040e-01, 2.525554367249600e-02 ] ; 
    qpta[ 37, :] = [ 5.119105251906040e-01,  -8.618000540391760e-01, 2.525554367249600e-02 ] ; 
    qpta[ 38, :] = [ -8.618000540391760e-01,  -6.501104711514279e-01, 2.525554367249600e-02 ] ; 
    qpta[ 39, :] = [ -6.501104711514279e-01,  -8.618000540391760e-01, 2.525554367249600e-02 ] ; 
    qpta[ 40, :] = [ -2.826243668420350e-01,  1.310653596252170e-01, 3.843285252084900e-02 ] ; 
    qpta[ 41, :] = [ 1.310653596252170e-01,  -2.826243668420350e-01, 3.843285252084900e-02 ] ; 
    qpta[ 42, :] = [ -8.484409927831820e-01,  1.310653596252170e-01, 3.843285252084900e-02 ] ; 
    qpta[ 43, :] = [ 1.310653596252170e-01,  -8.484409927831820e-01, 3.843285252084900e-02 ] ; 
    qpta[ 44, :] = [ -8.484409927831820e-01,  -2.826243668420350e-01, 3.843285252084900e-02 ] ; 
    qpta[ 45, :] = [ -2.826243668420350e-01,  -8.484409927831820e-01, 3.843285252084900e-02 ] ; 
    qpta[ 46, :] = [ -5.001217112903050e-01,  3.335660614538170e-01, 3.299329121160600e-02 ] ; 
    qpta[ 47, :] = [ 3.335660614538170e-01,  -5.001217112903050e-01, 3.299329121160600e-02 ] ; 
    qpta[ 48, :] = [ -8.334443501635120e-01,  3.335660614538170e-01, 3.299329121160600e-02 ] ; 
    qpta[ 49, :] = [ 3.335660614538170e-01,  -8.334443501635120e-01, 3.299329121160600e-02 ] ; 
    qpta[ 50, :] = [ -8.334443501635120e-01,  -5.001217112903050e-01, 3.299329121160600e-02 ] ; 
    qpta[ 51, :] = [ -5.001217112903050e-01,  -8.334443501635120e-01, 3.299329121160600e-02 ] ; 
    qpta[ 52, :] = [ -3.533089531977780e-01,  -2.753883373664800e-02, 8.141567721504400e-02 ] ; 
    qpta[ 53, :] = [ -2.753883373664800e-02,  -3.533089531977780e-01, 8.141567721504400e-02 ] ; 
    qpta[ 54, :] = [ -6.191522130655730e-01,  -2.753883373664800e-02, 8.141567721504400e-02 ] ; 
    qpta[ 55, :] = [ -2.753883373664800e-02,  -6.191522130655730e-01, 8.141567721504400e-02 ] ; 
    qpta[ 56, :] = [ -6.191522130655730e-01,  -3.533089531977780e-01, 8.141567721504400e-02 ] ; 
    qpta[ 57, :] = [ -3.533089531977780e-01,  -6.191522130655730e-01, 8.141567721504400e-02 ] ; 

    return qpta

def integrate_cell(j,v):

    fig = plt.figure()
    ax = fig.add_subplot(111)

    qpt = area_quadrature()

    vm = np.mean(v, axis=0);
    n = v.shape[0]
    intx = 0.0
    inty = 0.0
    quiver_inputs = []
    for i in range(n):   
         ip1 = (i+1) % n

         x = 0.5*(-v[i,0]*(qpt[:,0] + qpt[:,1]) + v[ip1,0]*(1.0 + qpt[:,0]) + vm[0]*(1.0 + qpt[:,1]))
         y = 0.5*(-v[i,1]*(qpt[:,0] + qpt[:,1]) + v[ip1,1]*(1.0 + qpt[:,0]) + vm[1]*(1.0 + qpt[:,1]))

         x = np.expand_dims(x, axis=-1)
         y = np.expand_dims(y, axis=-1)

         phi, rx, ry = wachpress(n, v, x, y, method='distance')        
         phix, phiy = grad_wachpress(phi,rx,ry) 

         levels = np.linspace(0.0,1.0,10)
         c = ax.tricontourf(np.squeeze(x),np.squeeze(y),np.squeeze(phi[j,:,:]), levels=levels)
         quiver_inputs.append((x,y,phix[j,:,0],phiy[j,:,0]))
         ax.plot([v[i,0], v[ip1,0]], [v[i,1], v[ip1,1]],color='k')
         ax.scatter(x,y)

         intx = intx + np.sum(qpt[:,2]*phix[j,:,0])
         inty = inty + np.sum(qpt[:,2]*phiy[j,:,0])

    global_max = 0.0
    for (_, _, u, v) in quiver_inputs:
        seg_max = np.hypot(u, v).max()
        global_max = max(global_max, seg_max)

    for (x, y, u, v) in quiver_inputs:
        ax.quiver(x, y, u, v, scale=5.0*global_max)
        

    plt.savefig(f'int_points_{j}.png', bbox_inches='tight')
    plt.close()

    return intx, inty

         
    
     
