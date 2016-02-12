# coding: utf-8
# (c) 2016-02-12 Teruhisa Okada

import netCDF4
import shutil

std_tmp = '/home/okada/Data/ob500_std_i_param_v1_NL1_{0:04d}.nc'
std_main = '/home/okada/Data/ob500_std_i_param_v1_NL1.nc'
std_zeros = '/home/okada/Data/ob500_std_i_zeros.nc'
grdfile = '/home/okada/Data/ob500_grd-11_3.nc'

vnames = ['temp', 'salt', 'NO3', 'NH4', 'chlorophyll', 'phytoplankton', 'zooplankton', 
          'LdetritusN', 'SdetritusN', 'oxygen', 'PO4', 'LdetritusP', 'SdetritusP']

shutil.copyfile(std_zeros, std_main)
main = netCDF4.Dataset(std_main, 'a')

for i in range(12):
    stdfile = std_tmp.format(i+1)
    nc = netCDF4.Dataset(stdfile, 'r')
    for vname in vnames:
        main[vname][i] = nc[vname][:]
    nc.close()

main.close()
