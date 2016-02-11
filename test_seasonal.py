# coding: utf-8
# (c) 2015-12-30 Teruhisa Okada

"""
状態空間接近による季節変動調節法（柏木，1997）の実装
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import optimize

import romspy
import seasonal


def _get_sign(freq):
    """
    return test observations
    """
    np.random.seed(100)
    t = np.arange(0,freq*5,1)
    # true
    true = np.sin(2*np.pi*t/freq) + t*0.1
    # observation
    obs = true + np.random.normal(0, 0.4, size=true.size)
    mask = np.abs(t-40) > 5
    return obs[mask], t[mask], true, t


def _get_obs(datafile):
    """
    return observations
    """
    df = pd.read_csv(datafile, parse_dates=1, index_col=1, encoding='Shift_JIS', na_values='*')
    df.columns = ['station', 'layer', 'depth', 'bottm', 'temp', 'salt', 'light', 'DOp', 'turb', 'chlo']
    df.index.name = 'date'
    return df


def _read_mp_c(datafile, vname):
    df = pd.read_csv(datafile, parse_dates=1, index_col=1, encoding='Shift_JIS', na_values='*')
    df.columns = ['station', 'layer', 'depth', 'bottom', 'u', 'v']
    df.index.name = 'date'
    df = df.interpolate()
    return df[vname][df.bottom=='B']


def _plot3(x, y, **kw):
    fig, ax = plt.subplots(3,1,figsize=(10,10))
    ax[0].plot(x[:,0], '-', **kw)
    ax[1].plot(x[:,2], '-', **kw)
    ax[2].plot(x[:,0] + x[:,2], '-', **kw)
    ax[2].plot(y, '.', **kw)


def _plot4(x, y, y_time, **kw):
    fig, ax = plt.subplots(3,1,figsize=(9,6))
    t = x[:,0]
    s = x[:,2]
    x = t + s
    r = y - x[y_time]
    ax[0].plot(x, '-', label='trend+seasonal', **kw)
    ax[0].plot(t, '-', label='trend', **kw)
    ax[0].plot(y_time, y, '.', label='observation', **kw)
    ax[1].plot(s, '-', label='seasonal', **kw)
    ax[2].plot(y_time, r,'.-', label='residual', **kw)
    vmin, vmax, vstd = y.min(), y.max(), y.std()
    ax[0].set_ylim(vmin, vmax)
    ax[1].set_ylim(-0.5*(vmax-vmin), 0.5*(vmax-vmin))
    ax[2].set_ylim(-0.5*(vmax-vmin), 0.5*(vmax-vmin))
    for a in ax:
        a.legend(loc='best')


def test_sign():
    param = 10, 10
    freq = 12
    obs_value, obs_time, true, true_time = _get_sign(freq)
    args = (obs_value, obs_time, true_time, freq)
    res = optimize.minimize(seasonal.J, param, args=args, method='Nelder-Mead', options={'disp':True})
    param = res.x

    a, b, sigma2, logL, xs, Vs = seasonal.run_ks(param, *args)
    #a, b, sigma2, logL, xs, Vs = seasonal.run_kf(param, *args)
    _plot4(xs, obs_value, obs_time)

    fig = plt.gcf()
    ax = fig.get_axes()
    ax[0].plot(true, '-', label='true')
    ax[0].legend(loc='best')
    plt.savefig('test/sign.png', bbox_inches='tight')
    plt.show()

    #df = pd.DataFrame({'true':true, 'obs':obs, 'trend':xs[:,0], 'seasonal':xs[:,2]})
    #df['residual'] = df.obs - df.trend - df.seasonal
    #df.to_csv('test/sign.csv')


def test_chlo():
    #datafile = 'F:/okada/Dropbox/Data/kanku_wq_201508.csv'
    datafile = 'F:/okada/Dropbox/Data/kobe_c_201208.csv'
    df = _get_obs(datafile)
    df = df.interpolate()
    obs = df.chlo[df.layer==1.0].values
    param = 1, 1
    freq = 24
    args = (obs, freq)
    res = optimize.minimize(seasonal.J, param, args=args, method='Nelder-Mead', options={'disp':True})
    param = res.x

    a, b, sigma2, logL, xs, Vs = seasonal.run_ks(param, *(obs, freq))
    seasonal._plot4(xs, obs)
    romspy.savefig('test/chlo.png')
    plt.show()


def test_current(vname='u'):
    datafile = 'F:/okada/Dropbox/Data/kobe_c_201208.csv'
    obs = _read_mp_c(datafile, vname=vname)
    obs = obs[:200]
    param = 1, 1
    freq = 25
    args = (obs, freq)
    res = optimize.minimize(seasonal.J, param, args=args, method='Nelder-Mead', options={'disp':True})
    param = res.x

    a, b, sigma2, logL, xs, Vs = seasonal.run_ks(param, *(obs, freq))
    seasonal._plot4(xs, obs)
    romspy.savefig('test/{}_200h.png'.format(vname))
    plt.show()

    df = pd.DataFrame({'obs':obs, 'trend':xs[:,0], 'seasonal':xs[:,2]})
    df['residual'] = df.obs - df.trend - df.seasonal
    df.to_csv('test/{}_200h.csv'.format(vname))


if __name__ == '__main__':
    import seaborn as sns
    test_sign()
