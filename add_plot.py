# coding: utf-8
# (c) 2015-12-30 Teruhisa Okada

"""
状態空間接近による季節変動調節法（柏木，1997）の実装
"""

import matplotlib.pyplot as plt
import pandas as pd
import os

import romspy
import seasonal_kokyo as sk

mapfile = 'F:/okada/Dropbox/notebook/deg_OsakaBayMap_okada.bln'


def _read_stations(stafile):
    df = pd.read_csv(stafile, encoding='Shift_JIS', index_col='number')
    df['lat'] = df.lat_deg + df.lat_min / 60 + df.lat_sec / 3600
    df['lon'] = df.lon_deg + df.lon_min / 60 + df.lon_sec / 3600
    df = df[['name','lon','lat','id']]
    return df


def _plot_col(df, vname, fname, title):
    if vname == 'temp':
        vmin, vmax = 0, 35
    elif vname == 'DO':
        vmin, vmax = 0, 12
    elif vname == 'COD':
        vmin, vmax = 0, 12
    plt.figure(figsize=(6,6))
    plt.scatter(df[vname].values, df.ts.values, alpha=0.5)
    plt.plot([vmin,vmax], [vmin,vmax], 'k-', alpha=0.5)
    plt.xlim(vmin, vmax)
    plt.ylim(vmin, vmax)
    plt.xlabel('observations')
    plt.ylabel('trend+seasonal')
    plt.title(title)
    col_png = fname+'_col.png'
    sk._mkdir(col_png)
    romspy.savefig(col_png)


def plot_col(main_tmp, title_tmp, *args):
    start, end, vname, layer, stafile = args
    sta = _read_stations(stafile)
    for number in sta.index:
        sid = sta.id[number]
        fname = main_tmp.format(**locals())
        if os.path.exists(fname+'.csv'):
            df = pd.read_csv(fname+'.csv')
            if 'ts' in df.columns:
                title = title_tmp.format(**locals())
                _plot_col(df, vname, fname, title)


def plot4(main_tmp, title_tmp, *args):
    start, end, vname, layer, stafile = args
    sta = _read_stations(stafile)
    for number in sta.index:
        sid = sta.id[number]
        fname = main_tmp.format(**locals())
        df = pd.read_csv(fname+'.csv', index_col=0, parse_dates=0)
        title = title_tmp.format(**locals())
        sk._plot4(df, vname, fname, title)


def plot_stations(stafile):
    df = _read_stations(stafile)
    plt.figure(figsize=(8,8))
    plt.scatter(df.lon.values, df.lat.values)
    for i in df.index:
        #plt.text(df.lon[i], df.lat[i], df.id[i], backgroundcolor='w')
        plt.annotate(str(df.id[i]), xy=(df.lon[i], df.lat[i]), 
                xycoords='data',
                xytext=(df.lon[i]-0.025, df.lat[i]+0.015), 
                #textcoords='offset points',
                arrowprops=dict(arrowstyle="-")
                )
    romspy.basemap(mapfile)
    plt.gca().patch.set_facecolor('gray')
    plt.gca().patch.set_alpha(0.2)
    pngfile = stafile.replace('csv', 'png')
    romspy.savefig(pngfile)
    plt.show()


if __name__ == '__main__':
    #import seaborn as sns

    # input files
    stafile = 'F:/okada/Data/kokyo_osaka/kokyo_stations_osaka.csv'
    title_tmp = '{vname} (layer={layer}), Sta.{sid}'

    # output files
    outdir = 'F:/okada/Data/kokyo_osaka/seasonal2/'
    main_tmp = outdir + '02_main/{vname}_{sid}_{layer}'

    # parameters
    start, end = 1981, 2009
    vname = 'COD'
    layer = 11
    args = (start, end, vname, layer, stafile)

    #plot_col(main_tmp, title_tmp, *args)
    #plot4(main_tmp, title_tmp, *args)
    plot_stations(stafile)