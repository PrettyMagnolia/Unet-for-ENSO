import numpy as np
import os
import netCDF4 as nc
import torch
import matplotlib as mpl
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import torch.nn.functional as F


def keep_data_size(mod_data, obs_data):
    a = torch.cat((mod_data, obs_data), 1)
    b = a.to(torch.float32)
    return b


def write_nc(out_folder, newName, output, SICdata):
    # 看训练效果的一个变化
    a = np.array(output.cpu().detach().numpy())  # tensor转换成array
    SICdata = np.array(SICdata.cpu().detach().numpy())

    pathOut = os.path.join(out_folder, newName)

    f_w = nc.Dataset(pathOut, 'w', format='NETCDF4')

    # 确定基础变量的维度信息。相对与坐标系的各个轴(x,y,z)
    f_w.createDimension('Y', 384)
    f_w.createDimension('X', 420)

    ##创建变量。参数依次为：‘变量名称’，‘数据类型’，‘基础维度信息’
    f_w.createVariable('Y', np.float64, ('Y'))
    f_w.createVariable('X', np.float64, ('X'))

    # 写入变量Y的数据。
    Y = np.arange(0, 384)
    f_w.variables['Y'][:] = Y
    # 写入变量X的数据
    X = np.arange(0, 420)
    f_w.variables['X'][:] = X

    # 新创建一个多维度变量，并写入数据
    f_w.createVariable('SIC', np.float64, ('Y', 'X'))
    f_w.createVariable('ASRME', np.float64, ('Y', 'X'))

    f_w.variables['SIC'][:] = a
    f_w.variables['ASRME'][:] = SICdata
    f_w.close()


def make_plot(lat, lon, data, title):
    dpi_lat = 180 / lat
    dpi_lon = 360 / lon
    lats = np.arange(-90 + 0.5 * dpi_lat, 90, dpi_lat)
    lons = np.arange(0.5 * dpi_lon, 360, dpi_lon)
    plt.rcParams['figure.dpi'] = 200
    # 创建画布以及ax
    fig = plt.figure()
    # 比例尺为 1:110m（m 表示 million）
    # 等距圆柱投影 ccrs.PlateCarree 作为地图投影
    ax = fig.add_subplot(111, projection=ccrs.PlateCarree(central_longitude=0))
    # ax.contourf(lon, lat, sst1, cmap='RdBu_r')
    gl = ax.gridlines(draw_labels=True, xlines=False, ylines=False)
    gl.xlines = False
    gl.ylines = False
    ax.coastlines()
    ax.add_feature(cfeature.LAND)  # 添加陆地
    ax.contourf(lons, lats, data, cmap='RdBu_r')
    plt.savefig(r"./outcome/" + title)

def make_test_data(mod_data, obs_data, device):
    mod_data = F.interpolate(torch.from_numpy(mod_data).unsqueeze(0).unsqueeze(0), scale_factor=0.25, mode='nearest')
    obs_data = F.interpolate(torch.from_numpy(obs_data).unsqueeze(0).unsqueeze(0), scale_factor=0.25, mode='nearest')

    mod_data = mod_data.to(device)
    obs_data = obs_data.to(device)

    a = torch.cat((mod_data, obs_data), 1)
    b = a.to(torch.float32)
    return b
