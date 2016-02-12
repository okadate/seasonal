# coding: utf-8
# (c) 2015-12-30 Teruhisa Okada

"""
状態空間接近による季節変動調節法（柏木，1997）の実装
"""

import numpy as np
from numpy.linalg import inv
from scipy import optimize

eps = 1.0e-8


def kf(x, y, t, V, F, G, H, *dims):
    """
    Kalman filter
    """
    ny, nt, nk = dims
    Iy = np.eye(ny)
    xf = np.zeros_like(x)
    Vf = np.zeros_like(V)
    xf[0] = x[0]
    Vf[0] = V[0]
    for k in range(1,nk):
        x[k] = np.dot(G, xf[k-1])
        V[k] = np.dot(np.dot(G, Vf[k-1]), G.T) + H
        if k in t:
            K1 = np.dot(V[k], F.T)
            K2 = np.dot(F, K1) + Iy
            K = np.dot(K1, inv(K2))
            yk = y[np.where(t==k)[0][0]]
            res = yk - np.dot(F, x[k])
            xf[k] = x[k] + np.dot(K, res)
            Vf[k] = V[k] - np.dot(K, np.dot(F, V[k]))
        else:
            xf[k] = x[k]
            Vf[k] = V[k]
    return xf, Vf, x, V


def ks(x, y, t, V, F, G, H, *dims):
    """
    Kalman smoother
    """
    ny, nt, nk = dims
    xf, Vf, x, V = kf(x, y, t, V, F, G, H, *dims)
    xs = np.zeros_like(x)
    Vs = np.zeros_like(V)
    xs[-1] = xf[-1]
    Vs[-1] = Vf[-1]
    for k in range(nk-2, -1, -1):
        # Smoother gain
        A1 = np.dot(G.T, inv(V[k+1]))
        A = np.dot(Vf[k], A1)
        # Smoothing
        res = xs[k+1] - x[k+1]
        xs[k] = xf[k] + np.dot(A, res)
        res = Vs[k+1] - V[k+1]
        Vs[k] = Vf[k] + np.dot(A, np.dot(res, A.T))
    return xs, Vs, xf, Vf, x, V


def loglikelihood(x, y, t, V, F, *dims):
    """
    return sigma2, log likelihood
    """
    ny, nt, nk = dims
    Iy = np.eye(ny)
    cff = 0
    sigma2 = 0
    # for i, k in enumerate(t):
    #     res = y[i] - np.dot(F, x[k])
    #     print i, k, y[i], np.dot(F, x[k]), res
    #     K1 = np.dot(V[k], F.T)
    #     K2 = np.dot(F, K1) + Iy
    #     cff += np.log(np.abs(K2))
    #     sigma2 += np.dot(np.dot(res.T, inv(K2)), res)
    for k in range(nk):
        K1 = np.dot(V[k], F.T)
        K2 = np.dot(F, K1) + Iy
        cff += np.log(np.abs(K2))
        if k in t:
            yk = y[np.where(t==k)[0][0]]
            res = yk - np.dot(F, x[k])
            sigma2 += np.dot(np.dot(res.T, inv(K2)), res)
    dimy = nt * ny
    sigma2 = sigma2 / (nk * dimy)
    logL = nk * dimy * np.log(sigma2) + cff
    #print sigma2, -0.5 * logL[0][0]
    return sigma2, -0.5 * logL[0][0]


def parse_param(param):
    try:
        a, b = param
    except:
        print param
    # a = np.exp(a)
    # b = np.exp(b)
    a = max(eps, a)
    b = max(eps, b)
    return a, b


def model_KG(param, freq, *dims):
    """
    Kitagara and Gersch (1984)
    """
    a, b = parse_param(param)
    ny, nt, nk = dims
    nx = 2 + freq -1
    x = np.zeros((nk, nx))
    V = np.zeros((nk, nx, nx))
    y = np.zeros((nt, ny))
    R = np.zeros((nt, ny, ny))
    F = np.zeros((ny, nx))
    G = np.zeros((nx, nx))
    H = np.zeros((nx, nx))
    Ix = np.eye(nx)
    F[0,0], F[0,2] = 1, 1
    G[0,0], G[1,0], G[0,1] = 2, 1, -1
    G[2,2:] = -1
    for i in range(nx-3):
        G[3+i,2+i] = 1
    H[0,0] = a ** (-1)
    H[2,2] = b ** (-1)
    return x, y, V, R, F, G, H, Ix


def model_A(param, freq, *dims):
    """
    Kashiwagi (1997)
    """
    a, b = parse_param(param)
    ny, nt, nk = dims
    nx = 2 + freq
    x = np.zeros((nk, nx))
    V = np.zeros((nk, nx, nx))
    y = np.zeros((nt, ny))
    R = np.zeros((nt, ny, ny))
    F = np.zeros((ny, nx))
    G = np.zeros((nx, nx))
    H = np.zeros((nx, nx))
    Ix = np.eye(nx)
    F[0,0], F[0,2] = 1, 1
    G[0,0], G[1,0], G[0,1] = 2, 1, -1
    G[2,2:-1], G[2,-1] = -0.5, 0.5
    for i in range(nx-3):
        G[3+i,2+i] = 1
    H[0,0] = a ** (-1)
    H[2,2] = (2*b) ** (-1)
    return x, y, V, R, F, G, H, Ix


def run_kf(param, *args):
    a, b = parse_param(param)
    obs, obs_time, mod_time, freq = args
    nt = len(obs_time)
    nk = len(mod_time)
    ny = 1
    dims = (ny, nt, nk)
    x, y, V, R, F, G, H, Ix = model_KG(param, freq, *dims)
    #x, y, V, R, F, G, H, Ix = model_A(param, freq, *dims)
    y[:,0] = obs
    t = obs_time
    x0, lam, sigma02, logL0 = 0, 100, 1, 0
    for i in range(10):
        x[0] = x0
        V[0] = lam / sigma02 * Ix
        xf, Vf, x, V = kf(x, y, t, V, F, G, H, *dims)
        #sigma2, logL = loglikelihood(x, y, t, V, F, *dims)
        sigma2, logL = loglikelihood(xf, y, t, Vf, F, *dims)
        if np.abs(logL - logL0) < eps:
            break
        else:
            sigma02 = sigma2
            logL0 = logL
    #print 'a, b, sigma2, logL =', a, b, sigma2, logL
    return a, b, sigma2, logL, xf, Vf


def run_ks(param, *args):
    a, b = parse_param(param)
    obs, obs_time, mod_time, freq = args
    nt = len(obs_time)
    nk = len(mod_time)
    ny = 1
    dims = (ny, nt, nk)
    x, y, V, R, F, G, H, Ix = model_KG(param, freq, *dims)
    #x, y, V, R, F, G, H, Ix = model_A(param, freq, *dims)
    y[:,0] = obs
    t = obs_time
    x0, lam, sigma02, logL0 = 0, 100, 1, 0
    for i in range(10):
        x[0] = x0
        V[0] = lam / sigma02 * Ix
        xs, Vs, xf, Vf, x, V = ks(x, y, t, V, F, G, H, *dims)
        #sigma2, logL = loglikelihood(x, y, t, V, F, *dims)
        sigma2, logL = loglikelihood(xs, y, t, Vs, F, *dims)
        if np.abs(logL - logL0) < eps:
            break
        else:
            sigma02 = sigma2
            logL0 = logL
    print 'a, b, sigma2, logL =', a, b, sigma2, logL
    return a, b, sigma2, logL, xs, Vs


def J(param, *args):
    a, b, sigma2, logL, xf, Vf = run_kf(param, *args)
    return logL


def J_ks(param, *args):
    a, b, sigma2, logL, xs, Vs = run_ks(param, *args)
    return logL


def main(obs, obs_time, mod_time, fname, freq=12):
    print fname
    """
    main of seasonal analysis
    """
    ini_param = 1, 1
    args = (obs, obs_time, mod_time, freq)

    # compute opt parameter
    #res = optimize.minimize(J, ini_param, args=args, method='Nelder-Mead', options={'disp':True, 'maxiter':100})
    #res = optimize.minimize(J, ini_param, args=args, method='Powell', options={'disp':True, 'maxiter':50})
    res = optimize.minimize(J, ini_param, args=args, method='CG', options={'disp':False, 'maxiter':50})
    #res = optimize.minimize(J, ini_param, args=args, method='BFGS', options={'disp':True, 'maxiter':50})
    param = res.x

    # kalman smoother with opt parameter
    a, b, sigma2, logL, xs, Vs = run_ks(param, *args)

    return xs


if __name__ == '__main__':
    import seaborn as sns
    import test_seasonal    
    test_seasonal.test_sign()

