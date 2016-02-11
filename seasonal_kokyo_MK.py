# coding: utf-8
# (c) 2015-12-30 Teruhisa Okada

"""
状態空間接近による季節変動調節法（柏木，1997）の実装
"""

import pandas as pd
import datetime
import numpy as np
import os

import seasonal_kokyo as sk


def _read_MK(txt0, txt1, txt2):
    print txt0, txt1, txt2
    na_values = [99999, 9999]
    parser = lambda date: datetime.datetime.strptime(date.decode('utf-8'), '%Y %m%d')
    df0 = pd.read_csv(txt0, na_values=na_values, parse_dates=[[6,7]], date_parser=parser)
    df = df0.ix[:, [0,3,9,10,16,18]]  # 'date', 'number', 'layer', 'depth', 'temp', 'H'
    df1 = pd.read_csv(txt1, na_values=na_values)
    df = pd.concat([df, df1.ix[:, [9,13]]], axis=1)  # 'DO', 'COD'
    df.columns = ['date', 'number', 'layer', 'depth', 'temp', 'H', 'DO', 'COD']
    if os.path.exists(txt2):
        df2 = pd.read_csv(txt2, na_values=na_values)
        df2 = df2.ix[:, [7,9]]  # 'TN', 'TP'
        df2.columns = ['TN', 'TP']
        df = pd.concat([df, df2], axis=1)
    return df


def _read_dataset(txt_tmp, *args):
    years, kinds, stations = args
    for s in stations:
        for year in years:
            txt0 = txt_tmp.format(year, kinds[0], s)
            txt1 = txt_tmp.format(year, kinds[1], s)
            txt2 = txt_tmp.format(year, kinds[2], s)
            df1 = _read_MK(txt0, txt1, txt2)
            try:
                df = pd.concat([df, df1], axis=0)
            except:
                df = df1
    return df


def pre(txt_tmp, pre_tmp, *args):
    """
    Read MK txt files

    prefile: output file
    """
    start, end, vname, layer, stafile = args
    years = np.arange(start, end+1)
    kinds = [0,2,3]
    stations = [27,28]
    args = (years, kinds, stations)
    df = _read_dataset(txt_tmp, *args)
    prefile = pre_tmp.format(**locals())
    df.to_csv(prefile, index=None)
    return df


if __name__ == '__main__':

    # input files
    txt_tmp = 'F:/okada/Data/kokyo/3_sea/MK{0}{1:02d}{2}_3.txt'
    stafile = 'F:/okada/Dropbox/2016_seasonal2/kokyo_stations.csv'

    # output files
    outdir = 'F:/okada/2016_Results/seasonal2/'
    pre_tmp = outdir + '01_pre_DO_COD_TN_TP_{start}_{end}_3.csv'
    main_tmp = outdir + '02_main/{vname}_{sid}_{layer}'
    post_tmp = outdir + '03_post/{vname}/{vname}_{layer}_{time_str}.png'
    post2_tmp = outdir + '03_post/{vname}_obs/{vname}_{layer}_{time_str}.png'

    # parameters
    start, end = 1981, 2009
    vname = 'COD'
    layer = 11
    args = (start, end, vname, layer, stafile)

    # run
    code = '1'
    if '1' in code:
        pre(txt_tmp, pre_tmp, *args)
    if '2' in code:
        sk.main(pre_tmp, main_tmp, *args)
    if '3' in code:
        sk.post(main_tmp, post_tmp, skip=[22], *args)
    if '4' in code:
        sk.post(main_tmp, post2_tmp, plot_obs=True, *args)
