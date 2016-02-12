# coding: utf-8
# (c) 2016-02-12 Teruhisa Okada

import numpy as np
import netCDF4
from numba.decorators import jit
from datetime import datetime

import romspy
from romspy.make import make_ini_zero


@jit
def moving_mean_jit(a, r=3):
    m = np.zeros_like(a)
    for i in range(len(a)):
        if i < r:
            m[i] = np.mean(a[:i+r])
        elif i > 31-r:
            m[i] = np.mean(a[i-r:])
        elif i >= r:
            m[i] = np.mean(a[i-r:i+r])
    return m


def cal_std(var):
    tmax, kmax, jmax, imax = var.shape
    v_std = np.zeros((kmax, jmax, imax))
    for k in xrange(kmax):
        for i in xrange(imax):
            for j in xrange(jmax):
                v = var[:,k,j,i]
                if v[-1] < 1000:
                    mm = moving_mean_jit(v, r=5)
                    v_std[k,j,i] = np.std(v - mm)
                else:
                    #v_std[k,j,i] = np.NaN
                    pass
    return v_std


def main(i):
    his_tmp = '/home/okada/ism-i/apps/OB500P/param_v1/NL1/ob500_his_{0:04d}.nc'
    std_tmp = '/home/okada/Data/ob500_std_i_param_v1_NL1_{0:04d}.nc'
    grdfile = '/home/okada/Data/ob500_grd-11_3.nc'
    mapfile = '/home/okada/romspy/romspy/deg_OsakaBayMap_okada.bln'

    vnames = ['temp', 'salt', 'NO3', 'NH4', 'chlorophyll', 'phytoplankton', 'zooplankton', 
              'LdetritusN', 'SdetritusN', 'oxygen', 'PO4', 'LdetritusP', 'SdetritusP']

    hisfile = his_tmp.format(i)
    stdfile = std_tmp.format(i)

    his = romspy.hview.Dataset(hisfile, grdfile, mapfile)
    var_std = {}
    for vname in vnames:
        print 'start {}({})'.format(vname, i)
        var_std[vname] = cal_std(his.nc[vname])
    his.nc.close()

    make_ini_zero(grdfile, stdfile, dstart=datetime(2012,i,16,0,0), biofile=True)

    std = netCDF4.Dataset(stdfile, 'a')
    for vname in vnames:
        std[vname][:] = var_std[vname]
    std.close()

if __name__ == '__main__':
    import os
    import sys
    for i in range(1,13):
        pid = os.fork()
        if pid == 0:
            main(i)
            print 'Finished precessing-{}'.format(i)
            sys.exit()
    print 'Main proccessing is waiting'
    os.wait()
