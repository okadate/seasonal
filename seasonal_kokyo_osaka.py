# coding: utf-8
# (c) 2015-12-30 Teruhisa Okada

"""
状態空間接近による季節変動調節法（柏木，1997）の実装
"""

import pandas as pd
import datetime

import seasonal_kokyo as sk


def _read_kokyo(datafile, vname):
    print datafile
    parser = lambda date: datetime.datetime.strptime(date.decode('utf-8'), '%Y %m%d %H%M')
    df = pd.read_csv(datafile, na_values='*', parse_dates=[[5,8,9]], date_parser=parser)  # , encoding='Shift_JIS')
    df.columns = ['date','area','station','layer','area_name','sta_name','mmdd','n','nan',vname]
    df['number'] = 100 * df.area + df.station
    return df[['date','number','layer',vname]]


def pre(csv_tmp, pre_tmp, *args):
    """
    Read csv file downloded by Osaka Pref.

    csv_tmp: input file
    pre_tmp: output file
    """
    start, end, vname, layer, stafile = args
    csvfile = csv_tmp.format(**locals())
    prefile = pre_tmp.format(**locals())
    df = _read_kokyo(csvfile, vname)
    df.to_csv(prefile, index=None)
    return df


if __name__ == '__main__':

    # input files
    csv_tmp = 'F:/okada/Data/kokyo_osaka/{vname}_{start}_{end}.csv'
    stafile = 'F:/okada/Data/kokyo_osaka/kokyo_stations_osaka.csv'

    # output files
    outdir = 'F:/okada/Data/kokyo_osaka/seasonal2/'
    pre_tmp = outdir + '01_pre_{vname}_{start}_{end}.csv'
    main_tmp = outdir + '02_main/{vname}_{sid}_{layer}'
    post_tmp = outdir + '03_post/{vname}/{vname}_{layer}_{time_str}.png'
    post2_tmp = outdir + '03_post/{vname}_obs/{vname}_{layer}_{time_str}.png'
    #post2_tmp = outdir + '03_post/{vname}_obs_line/{vname}_{layer}_{time_str}.png'

    # parameters
    start, end = 1972, 2010
    vname = 'COD'
    layer = 11
    args = (start, end, vname, layer, stafile)

    # run
    code = '34'
    if '1' in code:
        pre(csv_tmp, pre_tmp, *args)
    if '2' in code:
        sk.main(pre_tmp, main_tmp, *args)
    if '3' in code:
        sk.post(main_tmp, post_tmp, *args, skip=[22], function='linear')
    if '4' in code:
        sk.post(main_tmp, post2_tmp, *args, plot_obs=True, function='linear')
