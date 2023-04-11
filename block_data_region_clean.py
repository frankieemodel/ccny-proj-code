# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.5
#   kernelspec:
#     display_name: base
#     language: python
#     name: python3
# ---

# +
import matplotlib
import cartopy.crs as ccrs
import cartopy
import matplotlib.ticker as mticker
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.patches as mpatches
from IPython.display import set_matplotlib_formats
plt.style.use('tableau-colorblind10')
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import warnings 
warnings.filterwarnings("ignore")
import xarray as xr
import numpy as np
import datetime as dt

import os
import datetime_tools as dtools
import month_indices as mi
import matplotlib.colors as colors
from matplotlib.cm import ScalarMappable

# +
# CHOICES
## CHOOSE FILE TO SELECT DAYS
# to generalize, need to get var name from file 

f = '/Volumes/T7/Data/blockdata_2000-2021.nc'
ds = xr.open_dataset(f, decode_times=False)
block_data = ds.blocklocation.values
# block data from 2000 to 2021, cl data 2001 to 2021,
# so need all but first year of block data
years = np.arange(2001,2022)
block_data = block_data[1:,:]

# find indices of blockin over threshold
th = 0.75

## PLOTTING CHOICES
# either BLOCK DATA: 0 or CLOUD DATA: 1
plotting = 0 # BLOCK DATA
# plotting = 1 # CLOUD DATA

# for cloud data choose a variable to plot using indices
vars = ['cldfrac', 'cldtau', 'cldtau-lin', 'lwp', 
    'iwp', 'cldpress', 'cldtemp', 'cldhght']
var = vars[7] # 0 for cldfrac, 2 for linear vis optical depth, -1 cld height etc

# CHOOSE LAT/LON bounds for region
# choice of lat/lon in +/- format
# lat bounds
latb = np.array([45,60])
# latb = np.array([45,60])
# latbl = [45,60]
# latb = np.array([0,15])
# latbl = [0,15]
# latb = [x + 0.5  if x ==0 else x for x in latbl]
# latb = [x + 0.5 if any(x == 0 for x in latbl) else x for x in latbl ]
# latb = np.asarray(latb)

# lon bounds
# 70W, 30W (shifting 30w from original position)
# original position was 40W, 0.5W bc it got weird when I
# tried 0 as coord


# lonbl = [-40,0]
# lonbl = [-70,-30]
# lonbl = [-180,-140]
lonbl = [0,40]

# lonbl = [0,40]
# lonbl = [20,-20]
# lono = np.asarray([-40,0])
# lono = [x + 0.1  if x ==0 else x for x in lonbl]
if any(x == 0 for x in lonbl):
    if any(x < 0 for x in lonbl):
        lono = [x - 0.5 for x in lonbl]
    else:
        lono = [x + 0.5 for x in lonbl]
else:
    lono = [x for x in lonbl]
# lono = [x + 0.5 if any(x == 0 for x in lonbl) else x for x in lonbl ]
lono = np.asarray(lono)

# lono = np.array([-40.5,-0.5])
# lono = np.array([-70,-30])
# lono = np.array([-180,-140])
# lono = [-70,30]

lonb = lono%360
print(lonb[0],lonb[1])


# +
# plotting fxn
# INITIALIZING PLOTTING

# User made function for plotting
def add_coast(ax1,ccrs):
    """ this is a simple function to add coastlines to a plot"""
    gl = ax1.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                      linewidth=1.0, color='gray', alpha=0.5, linestyle='--')

    gl.top_labels = False
    gl.right_labels = False
    #gl.xlocator = mticker.FixedLocator([-75,-60,-45])
    #gl.ylocator = mticker.FixedLocator([24,30,36,42,48,54])
    gl.xlabel_style = {'size': 12}
    gl.ylabel_style = {'size': 12}
    #gl.xlines = False
    #gl.ylines = False
    ax1.coastlines(resolution = '50m', color = 'k', linewidth = 0.5)
    ax1.add_feature(cartopy.feature.LAKES,edgecolor='k')
    return None

# to boost plot resolution
set_matplotlib_formats('retina')

# cmap_full = 'seismic'
# cmap_full = 'YlOrRd'
cmap_full = 'plasma'
projection_crs = ccrs.PlateCarree()


# +
# Get lat/lon vals from block data 

print(block_data.shape)
lat = ds.lat.values
lon = ds.lon.values

# get indices of lat/lon bounds chosen for region at start
lati = [abs(lat-latb[i]).argmin() for i in range(2)]
loni = [abs(lon-lonb[i]).argmin() for i in range(2)]

# print index and corresponding lat/lon
[print(lati[x],lat[lati[x]]) for x in range(2)]
[print(loni[x],lon[loni[x]]) for x in range(2)]

# # these will be lat/lon dim of the resulting region
# latdim = lati[1] - lati[0]
# londim = loni[1] - loni[0]

# -

# pick out the block to use to find the indices
# if loni[0] < loni[1]:
#     reg_block = block_data[:,:,lati[0]:lati[1],loni[0]:loni[1]]
# else:
#     reg_block = block_data[:,:,lati[0]:lati[1],loni[1]:loni[0]]
reg_block = block_data[:,:,lati[0]:lati[1],loni[0]:loni[1]]
print(reg_block.shape)

# +
# find indices
# create an array for the %of region covered 
# by blocking each day of each yr ie array 20x365

meana_lon = np.nanmean(reg_block, axis=3)
print(meana_lon.shape)
meana_ll = np.nanmean(meana_lon, axis=2)
print(meana_ll.shape)
# find indices of vals over threshold
ibb = np.where(meana_ll>th)
# print num of indices ie days over th
print(len(ibb[0]))
# ibb is 2 arrays, one is year and one is day
# use to select data from for example cloud_data by:
# cloud_data[ibb[i],ibb[i],:,:] to get data from each day

yrs, days = ibb[0], ibb[1]


# +
# import anomaly data
if plotting != 0:

    # import anomaly data
    afile = '/Volumes/T7/Data/anom_all/anomdata_' + var + '.nc'
    af = xr.open_dataset(afile, decode_times=False)
    var_name = list(af.data_vars)[0]
    cl_data = af[var_name].to_numpy()
    print(cl_data.shape)

    an_data = cl_data[yrs,days,:]
    print(an_data.shape)
    an_avg = np.nanmean(an_data, axis=0)
    print(an_avg.shape)

    # import cloud data
    cfile = '/Volumes/T7/Data/cloud_data_all/clouddata_' \
        + var + '.nc'
    cf = xr.open_dataset(cfile, decode_times=False)
    var_name = list(cf.data_vars)[0]
    cl_data = cf[var_name].to_numpy()
    print(cl_data.shape)

    cl_data = cl_data[yrs,days,:]
    print(cl_data.shape)
    cl_avg = np.nanmean(cl_data, axis=0)
    print(cl_avg.shape)




# +
# Plotting formatting stuff
# doesn't depend on what's being plotted
# number of time steps or days included
tmstps = str(len(yrs))

# percent (as integer) of blocking used as threshold
thp = int(th*100)

# for lat/lon in title
deg_sign = u'\N{DEGREE SIGN}'

svlat = {}
svlon = {}
for i in range(2):
    if latb[i] > 0:
        svlat[i] = str(abs(latb[i])) + 'N'
    else:
        svlat[i] = str(abs(latb[i])) + 'S'
    if lono[i] > 0:
        svlon[i] = str(abs(lono[i])) + 'E'
    else:
        svlon[i] = str(abs(lono[i])) + 'W'
    svlat[i] = svlat[i].replace('.', 'p')
    svlon[i] = svlon[i].replace('.', 'p')

# +
## CHOOSE DATA TO PLOT
## CHANGE THIS!! need to plot either 1 or 2 plots...
# depends on switch at start now
# plot_data = cl_data # block_data or cl_data
# plot_file = cf # ds for block or cf for cloud
if plotting == 0:
    plot_data = block_data
    plot_file = ds
    seldata = plot_data[yrs,days,:]
    seldata_avg = np.nanmean(seldata, axis=0)
    # get info about the varible from the metadata
    ## get info for plotting

    var_name = list(plot_file.data_vars)[0]
    unit = plot_file[var_name].units
    std_name = plot_file[var_name].long_name
    std_name = std_name.split(',')[0]

    svname = std_name.replace(' ', '-')
    #get lon/lat values from the data file
    lat = plot_file.lat.values
    lon = plot_file.lon.values
    dmax = np.max(seldata_avg)
    dmin = np.min(seldata_avg)
    if abs(dmax) > abs(dmin):
        max_val = dmax
        min_val = -dmax
    else:
        max_val = abs(dmin)
        min_val = dmin
    
    # color scheme
    cmap_full = 'seismic'

    # creating plot
    fig = plt.figure(figsize=(11,8.5))
    ax = plt.axes(projection=projection_crs)

    ticks = np.linspace(min_val, max_val,11)

    # plotting data
    cf1= ax.contourf(
        lon, lat, seldata_avg,
        transform=projection_crs, cmap=cmap_full, levels = 30,
        norm = colors.TwoSlopeNorm(vcenter=0, vmin=min_val, vmax=max_val),
        )
    
    # calling user defined function above to add map elements
    add_coast(ax,ccrs)

    # making box to outline region selected
    ax.add_patch(mpatches.Rectangle(xy=[lono[0], latb[0]], 
            width=lono[1]-lono[0], 
            height=latb[1]-latb[0],
            facecolor='none', edgecolor='k',
            transform=ccrs.PlateCarree()
            )
        )
    plt.colorbar(cf1, ax=ax, shrink=.53, aspect=20, pad=0.03,
            label=unit, ticks = ticks)

    plt.title(std_name + " - Averaging days with >" + str(thp) + "% blocking in region" +
            '\nNumber of days included: ' + tmstps)
    plt.savefig('Figures/region_blocking_composites/regblockavg_' 
            + svlat[0] + 'to' + svlat[1] 
            + '_' + svlon[0] + 'to' + svlon[1] 
            + '_'  + svname  + '.png', 
            Transparent = False, dpi = 100 
            )
    plt.show()

# plotting anom and cloud data
else:
    # need to loop over the two vars
    das = [cl_avg, an_avg]
    fls = [cf, af]
    for x in range(2):
        plot_file = fls[x]
        seldata_avg = das[x]
        print(plot_file)

        # get info about the varible from the metadata

        var_name = list(plot_file.data_vars)[0]
        unit = plot_file[var_name].units

        if unit == 'percent':
            unit = '%'
        std_name = plot_file[var_name].long_name
        std_name = std_name.split(',')[0]

        std_name = std_name.split(':')[1]

        if x == 0:
            svname = "cldata_" + var
            cmap_full = 'plasma'
        else:
            svname = "anomdata_" + var
            cmap_full = 'seismic'

        #get lon/lat values from the data file
        lat = plot_file.lat.values
        lon = plot_file.lon.values

        dmax = np.max(seldata_avg)
        dmin = np.min(seldata_avg)
        if abs(dmax) > abs(dmin):
            max_val = dmax
            min_val = -dmax
        else:
            max_val = abs(dmin)
            min_val = dmin

        # PLOT
        fig = plt.figure(figsize=(11,8.5))
        ax = plt.axes(projection=projection_crs)
        
        if x == 1:
            ticks = np.linspace(min_val, max_val,11)
    
            cf1= ax.contourf(
                lon, lat, seldata_avg,
                transform=projection_crs, cmap=cmap_full, levels = 30,
                # norm = colors.TwoSlopeNorm(vcenter=0, vmin=-3.75, vmax=3.75),
                norm = colors.TwoSlopeNorm(vcenter=0, vmin=min_val, vmax=max_val),
                # norm = colors.TwoSlopeNorm(vcenter=0),
                # vmin = min_val, vmax = max_val,
                )
            
            add_coast(ax,ccrs) #calling user defined function above

            ax.add_patch(mpatches.Rectangle(xy=[lono[0], latb[0]], 
                            width=lono[1]-lono[0], 
                            height=latb[1]-latb[0],
                            facecolor='none', edgecolor='k',
                            transform=ccrs.PlateCarree())
                        )
            
            cbar = fig.colorbar(
                ScalarMappable(norm=cf1.norm, cmap=cf1.cmap),
                ticks=ticks, ax = ax, 
                shrink = .53, aspect=20, pad=0.03, label=unit
                )

            # formats colorbar numbers so they're only 1 decimal
            cbar.ax.set_yticklabels(['{:.1f}'.format(i) for i in ticks])

            plt.title(std_name + " (Anomalies) - Averaging days with >" 
                    + str(thp) + "% blocking in region \
                    \nNumber of days included: " + tmstps)
       
        else:
            cf1= ax.contourf(
                lon, lat, seldata_avg,
                transform=projection_crs, cmap=cmap_full, levels = 30,
                )
            
            add_coast(ax,ccrs) #calling user defined function above

            ax.add_patch(mpatches.Rectangle(xy=[lono[0], latb[0]], 
                            width=lono[1]-lono[0], 
                            height=latb[1]-latb[0],
                            facecolor='none', edgecolor='k',
                            transform=ccrs.PlateCarree())
                        )
            
            plt.colorbar(cf1, ax=ax, shrink=.53, aspect=20, pad=0.03,
                # label=unit, ticks = ticks)
                label=unit)
            plt.title(std_name + " - Averaging days with >" + str(thp) 
                    + "% blocking in region" 
                    + '\nNumber of days included: ' + tmstps
                    )
        
        plt.savefig('Figures/region_blocking_composites/regblockavg_' 
            + svlat[0] + 'to' + svlat[1] + '_' 
            + svlon[0] + 'to' + svlon[1] + '_'  
            + svname  + '.png', 
            Transparent = False, dpi = 100 
            )
        
        plt.show()



# -


