import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.gridspec import GridSpec
import seaborn as sns
import rasterio
import rioxarray as rxr
import glob
import helpers_plots as hlpplots
import proc_helper_functions as hlp


## set some file paths:

# folder with standardized geojsons, use only non-circles:
folder_new = '/Users/leahartl/Desktop/inventare_2025/mergedfiles/split_vanishing/'
# get the "proc2" or "proc3" files (using proc2 for data upload,
# proc3 additionally contains aspect information)
fls_new = glob.glob(folder_new+'*GI5_proc3.geojson')

## older GI:
# GI3 folder - select the files that have elevation information (added in prepro)
fls_GI3 = glob.glob('/Users/leahartl/Desktop/inventare_2025/GI/GI_3_new/*_elevation.geojson')
# GI2 folder
fls_GI2 = glob.glob('/Users/leahartl/Desktop/inventare_2025/GI/GI_2/*.shp')
# GI1 folder
fls_GI1 = glob.glob('/Users/leahartl/Desktop/inventare_2025/GI/GI_1/*.shp')
# GILIA folder
fls_GILIA = glob.glob('/Users/leahartl/Desktop/inventare_2025/GI/GI_LIA/*.shp')

# get vanishing glaciers:
fls_vanishing= glob.glob(folder_new+'*_vanishing_glaciers_GI3outline.geojson')
# vanished glaciers:
fls_vanished = glob.glob(folder_new+'**_vanished_glaciers_GI3outline.geojson')





## ------------ load GI files ----------------
# load GI5 files:
GI5 = hlp.getGI(fls_new, -18)
GI5['area'] = GI5.geometry.area

# load GI3 files:
GI3 = hlp.getGI(fls_GI3, -18)
GI3['area'] = GI3.geometry.area

# load GI2 files:
GI2 = hlp.getGI(fls_GI2, -4)
GI2.rename(columns={'year':'year_GI2', 'area': 'area_GI2'}, inplace=True)

# load GI1 files:
GI1 = hlp.getGI(fls_GI1, -4)
GI1['year_GI1'] = 1969
GI1.rename(columns={'area': 'area_GI1'}, inplace=True)

# load GI_LIA files:
GILIA = hlp.getGI(fls_GILIA, -4)

# get the intermediate inventories and return as one gdf:
subregs = hlp.subregions(GI5)

# get the vanished and vanishing glaciers: 
# save geojson of centroids to file, make and save dataframe with summary stats
# "out/vanishing_glaciers.geojson"
# "out/goneglaciers_summary.csv"
# return joint dataframe of both vanishing and vanished glaciers:
goneglaciers = hlp.getGoneGlaciers(fls_vanishing, fls_vanished)


# print('GI5 regions: n=', len(GI5.region.unique()), GI5.region.unique())
# print('GI3 regions: n=', len(GI3.region.unique()),  GI3.region.unique())
# print('GI2 regions: n=', len(GI2.region.unique()),  GI2.region.unique())
# print('GI1 regions: n=', len(GI1.region.unique()),  GI1.region.unique())
# print('GILIA regions: n=', len(GILIA.region.unique()), GILIA.region.unique())


## ------------ load GI files ----------------


## ------------ Process GI5 and GI3 ----------------
# merge GI3 and GI5 on ID to have one dataframe for comparisons, keep GI5 vanished glaciers in outer merge:
GI_merge = GI5[['id', 'region', 'name', 'area', 'min_elev', 'max_elev', 'median_elev', 'year',
                'outline_qf', 'mean_slope', 'circmean_aspect', 'x', 'y']].merge(GI3[['id', 'region',
                'name', 'area', 'min_elev', 'max_elev', 'median_elev', 'mean_slope', 'circmean_aspect', 'x', 'y', 'year']],
                left_on='id', right_on='id', suffixes=('_GI5', '_GI3'), how='outer')

# fill nan area instances in GI5 (vanishing glaciers) with 0
GI_merge['area_GI5'] = GI_merge['area_GI5'].fillna(0)
# fill nan year instances in GI5 (vanishing glaciers) with 2023
GI_merge['year_GI5'] = GI_merge['year_GI5'].fillna(2023)

# add uncertainty columns for GI5:
GI_merge = hlp.add_reluncertainties(GI_merge, 'area_GI5')
# add uncertainty columns for GI3 (different categories, as in GI3 paper; Aberman et al 2010 values)
GI_merge = hlp.add_reluncertaintiesGI3(GI_merge, 'area_GI3')

# add uncertainties in change:
# use this for the sum of the squares:
# GI_merge['unc_change'] = np.sqrt(GI_merge['unc_abs']**2 + GI_merge['unc_abs_GI3']**2)
# use this for linear sum:
GI_merge['unc_change'] = GI_merge['unc_abs_GI5'] + GI_merge['unc_abs_GI3']

# print(GI_merge[['id', 'area_GI3', 'area_GI5', 'year_GI5', 'year_GI3']].head())
# get GI3 to GI5 loss rates (area change as percentage of GI3 area, per year)
GI_merge['loss_rate'] = 100*((GI_merge['area_GI5'] - GI_merge['area_GI3'])/(GI_merge['year_GI5'].astype(int)-GI_merge['year_GI3'].astype(int)))/GI_merge['area_GI3']

# group data by region
grouped = hlp.prepareTable(GI_merge, GI3, GI5, goneglaciers)

# make some output tables:
# reformat the grouped dataframe for an output csv table:
# "out/summary_area_changesGI3GI5.csv"
hlp.makeTable(grouped, GI_merge, GI3, GI5)

# get area and number of glaciers per year of data coverage, write to file
# get area and number of glaciers per data type, write to file
# "out/data_years.csv"
# "out/datatype_area_count.csv"
hlp.datayears(GI5)


# prepare data frame for stacked area/bar charts. Area since LIA
df_prc, df_abs = hlp.sinceLIA(GILIA, GI1, GI2, GI3, GI5)

# compute loss rates across the GI and write to output tables. Also find lost glaciers GI1GI2
# and GI2GI3 and export tables with positive change rates
# 'out/rates_T.csv', 'out/rates_1.csv', lost glaciers GI1GI2, lost glaciers GI2GI3, various tables for >0 change
GI1GI2, GI2GI3, all_mrg = hlp.lossrates(GI_merge, subregs, GILIA, GI1, GI2, GI3, GI5, goneglaciers)



#######  OUTPUT #######
# get hypsometry all regions, save to csv
# extract data from DEM for elevation plots (SLOW!!)
demFn = '/Users/leahartl/Desktop/inventare_2025/DEM_BEV/ogd_10m_at_clipped.tif'
demAspect = '/Users/leahartl/Desktop/inventare_2025/DEM_BEV/ogd_10m_Aspect.tif'

print('getting elevation data - study region')
# get area that has disappeared:
GI3m = GI3.dissolve()
GI5m = GI5.dissolve()
arealost = GI3m.overlay(GI5m, how='difference')
# produces 'df_area_elevation.csv'
hlp.getHyps(demFn, '', GI3, GI5, arealost, 'elevation')
# get hypsometry per region 
print('getting elevation data - regional')
# produces 'regions_medianElevation.csv'
hlp.getHypsRegions(demFn, GI5)

# make figure showing stacked area since LIA and historgrams of change rates:
# 'figures/loss_stacked_1850_panelsBARS.png'    
hlpplots.loss_stacked_BARS(GI1GI2, GI2GI3, all_mrg, df_prc, df_abs)
# make composite figure showing violin plots of glacier area and median elevation in 
# gi3, 5 and for the vanishing glaciers & bars of change rates per size class & scatter of change rates & vanishing glaciers
# 'figures/glacierwise_overview'   
hlpplots.rates_glacierwise1(GI1GI2, GI2GI3, all_mrg, GI3, GI5, goneglaciers)

## requires the following csv files to be present in "outfolder"
## 'summary_area_changesGI3GI5.csv', 'df_area_elevation.csv', 'regions_medianElevation.csv'
outfolder = '/Users/leahartl/Desktop/inventare_2025/processing/out/'
hlpplots.figsfromcsv(outfolder)


## run function to get buffer uncertainty - this writes a table to file.
## produces "out/buffertable.csv", SLOW!!
hlp.get_bufferUnc(GI5)


## make boxplots and category table - table currently used in Appendix. Contains area stats 
## for Outline Quality and other categorical flags
## also writes some ROGI comparison stats to table.
hlp.fig_box(GI5)

## make figure for paper: Piecharts with log scale scatter plot of AGI5 glaciers
hlpplots.piecharts(GI5)

# make pie chart plots for appendix:
hlpplots.piecharts_2(GI5)
hlpplots.piecharts_3(GI5)

# not needed
# hlpplots.fig_hist(GI5)


# print some information:
print('crevs and >0.01: ', GI5.loc[(GI5['area']>=0.01) & (GI5['crevs']==1)].shape)
print('crevs and >0.01, tot area: ', GI5.loc[(GI5['area']>=0.01) & (GI5['crevs']==1), 'area'].sum()*1e-6)

print('GI5 unc sum', GI_merge['unc_abs_GI5'].sum()*1e-6)
print('GI5 ar sum', GI_merge['area_GI5'].sum()*1e-6)

print('GI3 unc sum', GI_merge['unc_abs_GI3'].sum()*1e-6)
print('GI3 ar sum', GI_merge['area_GI3'].sum()*1e-6)


##  print some info for individual glaciers:
# Eiskar Ferner (id = 20001)
eisk = GI_merge.loc[GI_merge['id'] == 20001]

# convert to km2 and print:
print('Eiskar info:', eisk[['name_GI5', 'area_GI3', 'area_GI5', 'unc_abs_GI5', 'unc_abs_GI3','unc_change']])
print('Eiskar area change, km2:', 1e-6*(eisk['area_GI5']-eisk['area_GI3']))
print('Eiskar area change uncertainty, km2:', 1e-6*eisk['unc_change'].values[0])

# find glacier with largest absolute change:
maxchange = GI_merge.loc[(GI_merge['area_GI5'] - GI_merge['area_GI3']).abs() == (GI_merge['area_GI5'] - GI_merge['area_GI3']).abs().max()]
print('largest absolute change at:', maxchange)
print('largest absolute change:', maxchange['name_GI5'], (maxchange['area_GI5'] - maxchange['area_GI3'])*1e-6,'±', 1e-6*maxchange['unc_change'].values[0])



plt.show()



