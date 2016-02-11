# coding: utf-8
# (c) 2015-12-30 Teruhisa Okada

"""
状態空間接近による季節変動調節法（柏木，1997）の実装
"""

import matplotlib.pyplot as plt
import pandas as pd
from scipy import interpolate
import datetime
import numpy as np
import os

import romspy
import seasonal

mapfile = 'F:/okada/Dropbox/notebook/deg_OsakaBayMap_okada.bln'


def main(pre_tmp, main_tmp, *args):
    """
    Seasonal analysis for kokyo data

    prefile: input file
    main_tmp: output file name
    """
    freq = 12
    start, end, vname, layer, stafile = args
    prefile = pre_tmp.format(**locals())
    df = pd.read_csv(prefile, parse_dates='date', index_col='date')

    sta = _read_stations(stafile)
    for number in sta.index:
        sid = sta.id[number]
        fname = main_tmp.format(**locals())
        index = (df.number==number) & (df.layer==layer)
        if index.any() == True:
            df2 = df[index].resample('M', convention='start')
            obs = df2[vname].dropna()
            print number, len(obs)
            if len(obs) > freq:
                obs_value = obs.values
                obs_date = obs.index
                mod_date = pd.date_range(obs_date[0], obs_date[-1], freq='M')
                obs_time, mod_time = _cal_time2(obs_date, mod_date)
                xs = seasonal.main(obs_value, obs_time, mod_time, fname, freq)

                # write out seasonal + trend
                mod = pd.DataFrame({'trend':xs[:,0], 'seasonal':xs[:,2]}, index=mod_date)
                df2 = pd.concat([mod, obs], axis=1)
                df2['ts'] = df2.trend + df2.seasonal
                df2['residual'] = df2[vname] - df2.ts
                df2.to_csv('{}.csv'.format(fname))

                # print out plot4
                title = '{vname} (layer={layer}), Sta.{sid}'.format(**locals())
                _plot4(df2, vname, fname, title)
            else:
                obs.name = vname
                obs.to_csv('{}.csv'.format(fname), header=True)


def _read_stations(stafile):
    df = pd.read_csv(stafile, encoding='Shift_JIS', index_col='number')
    df['lat'] = df.lat_deg + df.lat_min / 60 + df.lat_sec / 3600
    df['lon'] = df.lon_deg + df.lon_min / 60 + df.lon_sec / 3600
    df = df[['name','lon','lat','id']]
    return df


def _read_seasonal(sfile):
    print sfile
    try:
        df = pd.read_csv(sfile, parse_dates='date', index_col='date')
    except:
        df = pd.read_csv(sfile, parse_dates=0, index_col=0)
    if 'trend' in df.columns:
        df['ts'] = df.trend + df.seasonal
    return df


def _cal_time2(obs_date, mod_date, freq='M'):
    mod_time = np.arange(len(mod_date))
    obs_time = []
    for od in obs_date:
        ot = np.where(mod_date==od)[0][0]
        obs_time.append(ot)
    return np.array(obs_time), mod_time


def _plot4(obs, vname, fname, title):
    import seaborn as sns
    """
    Plotting for seasonal analized data and observation

    obs: "dataframe" of observations
    vname: variable name
    fname: output file name
    """
    if vname == 'temp':
        vmin, vmax = 0, 35
    elif vname == 'DO':
        vmin, vmax = 0, 12
    elif vname == 'COD':
        vmin, vmax = 0, 12
    fig, ax = plt.subplots(3,1,figsize=(9,6))
    ax[0].plot(obs.index, obs.trend.values, '-', label='trend')
    ax[0].plot(obs.index, obs.ts.values, '-', label='trend+seasonal')
    ax[0].plot(obs.index, obs[vname].values, '.', label='observations')
    ax[1].plot(obs.index, obs.seasonal.values, '-', label='seasonal')
    ax[2].plot(obs.index, obs.residual.values, '.-', label='residual')
    ax[0].set_ylim(vmin, vmax)
    ax[1].set_ylim(-0.5*(vmax-vmin), 0.5*(vmax-vmin))
    ax[2].set_ylim(-0.5*(vmax-vmin), 0.5*(vmax-vmin))
    for a in ax:
        legend = a.legend(ncol=3)  # loc='best', , frameon=True
        legend.get_frame().set_facecolor('w')
        #legend.get_frame().set_alpha(0.5)
    ax[0].set_title(title)
    pngfile = fname+'.png'
    _mkdir(pngfile)
    plt.savefig(pngfile, bbox_inches='tight', dpi=300)
    plt.close()


def post(main_tmp, post_tmp, *args, **kw):
    """
    Horizontal interpolate

    main_tmp: input file
    post_tmp: output file
    """
    start, end, vname, layer, stafile = args
    skip = kw.pop('skip',[])
    plot_obs = kw.pop('plot_obs',False)

    sta = _read_stations(stafile)
    for number in sta.index:
        sid = sta.id[number]
        if sid in skip:
            continue
        sfile = main_tmp.format(**locals()) + '.csv'
        # sfile1 = sfile.replace(str(layer), '1')
        if os.path.exists(sfile):
            df = _read_seasonal(sfile)
            if plot_obs:
                df[number] = df[vname]
            elif 'ts' in df.columns:
                df[number] = df.ts
            else:
                continue
            try:
                ts = pd.concat([ts, df[number]], axis=1)
            except:
                ts = df[number]
        # elif os.path.exists(sfile1):
        #     df = _read_seasonal(sfile1)
        #     df[number] = df.ts
        #     try:
        #         ts = pd.concat([ts, df[number]], axis=1)
        #     except:
        #         ts = df[number]
    _plot_map(ts, post_tmp, *args, **kw)


def _plot_map(ts, post_tmp, *args, **kw):
    """
    plot for "ts" dataframe

    ts: input DataFrame
    post_tmp: output file
    """
    start, end, vname, layer, stafile = args
    sta = _read_stations(stafile)
    if vname == 'DO':
        vmin, vmax = 0, 10
    elif vname == 'COD':
        vmin, vmax = 0, 8
    elif vname == 'temp':
        vmin, vmax = 10, 30

    for t in ts.index:
        time_str = datetime.datetime.strftime(t, '%Y-%m')
        s2 = pd.concat([sta, ts[time_str].T], axis=1).dropna()
        if len(s2) > 0:
            xs = s2.lon.values
            ys = s2.lat.values
            zs = s2.ix[:,-1].values
            x, y, z = _rbf(xs, ys, zs, **kw)
            #x, y, z = _grid(xs, ys, zs, bin=50)
            plt.pcolor(x, y, z, vmin=vmin, vmax=vmax)
            plt.colorbar()
            romspy.basemap(mapfile)
            plt.scatter(xs, ys, c=zs, s=30, vmin=vmin, vmax=vmax)
        else:
            romspy.basemap(mapfile)
        figfile = post_tmp.format(**locals())
        _mkdir(figfile)
        plt.title('{vname} (layer={layer}) {time_str}'.format(**locals()))
        plt.savefig(figfile, bbox_inches='tight')
        plt.clf()
        #exit()


def _mask(x, y, z):
    from scipy import stats
    import numpy.ma as ma
    A = [34.680803, 135.390475]  # amagasaki
    B = [34.462681, 135.128398]  # sumoto
    lon = [A[1], B[1]]
    lat = [A[0], B[0]]
    a, b, r, _, _ = stats.linregress(lon, lat)
    mask = (y - a*x - b) > 0
    #print mask
    return ma.masked_array(z, mask=mask)


def _rbf(xs, ys, zs, **kw):
    print 'xs, ys, zs =', xs.size, ys.size, zs.size
    bin = kw.pop('bin', 50)
    smooth = kw.pop('smooth', 0.01)
    A = [34.724319, 135.461000]  # osaka
    B = [34.174980, 134.821156]  # nushima
    C = [34.300725, 135.079974]  # misaki

    zs = np.array(zs)
    rbf = interpolate.Rbf(xs, ys, zs, smooth=smooth, **kw)
    #x = np.linspace(xs.min(), xs.max(), bin)
    #y = np.linspace(ys.min(), ys.max(), bin)
    x = np.linspace(C[1], A[1], bin)
    y = np.linspace(C[0], A[0], bin)
    Y, X = np.meshgrid(y, x)
    Z = _mask(X, Y, rbf(X, Y))
    return X, Y, Z


def _grid(xs, ys, zs, bin):
    print 'xs, ys, zs =', xs.size, ys.size, zs.size
    points = np.append([xs], [ys], axis=0).T
    values = np.array(zs)
    x = np.linspace(xs.min(), xs.max(), bin)
    y = np.linspace(ys.min(), ys.max(), bin)
    Y, X = np.meshgrid(y, x)
    Z = interpolate.griddata(points, values, (X, Y), method='cubic')
    return X, Y, Z


def _mkdir(filepath):
    print filepath
    dirpath = os.path.dirname(filepath)
    if not os.path.exists(dirpath):
        print 'mkdir', dirpath
        os.makedirs(dirpath)
