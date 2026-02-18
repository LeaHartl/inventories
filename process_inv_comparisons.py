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
#import helpers_plots as hlpplots
import proc_helper_functions as hlp


# folder with standardized geojsons, use only non-circles:
folder_new = '/Users/leahartl/Desktop/inventare_2025/mergedfiles/split_vanishing/'

# get all GI5 "proc2" files in folder:
fls_new = glob.glob(folder_new+'*GI5_proc2.geojson')

# GI3 folder - select the files that have elevation information
fls_GI3 = glob.glob('/Users/leahartl/Desktop/inventare_2025/GI/GI_3_new/*_elevation.geojson')

# GI2 folder
fls_GI2 = glob.glob('/Users/leahartl/Desktop/inventare_2025/GI/GI_2/*.shp')
# GI1 folder
fls_GI1 = glob.glob('/Users/leahartl/Desktop/inventare_2025/GI/GI_1/*.shp')
# GILIA folder
fls_GILIA = glob.glob('/Users/leahartl/Desktop/inventare_2025/GI/GI_LIA/*.shp')




# load and deal with GLIMS export (extract only the Sommer et al 2020 ouutlines based on the FAU affiliation)
sommer = gpd.read_file('/Users/leahartl/Desktop/inventare_2025/Data/glims/glims_download_34733/glims_polygons.shp')
sommer = sommer.loc[sommer['chief_affl'] == 'Friedrich-Alexander University (FAU)']
# create a year 
sommer['year'] = pd.to_datetime(sommer['src_date']).dt.year
# create a column identifying the three time steps listed in Sommer et al 2020:
sommer['invNr'] = np.nan 
sommer.loc[sommer['year']<=2001, 'invNr'] = 1
sommer.loc[sommer['year']==2011, 'invNr'] = 2
sommer.loc[sommer['year']>=2012, 'invNr'] = 3


GI5 = hlp.getGI(fls_new, -18)
GI5['area']=GI5.geometry.area
# print(GI5[['name', 'id', 'region']].loc[GI5['id']==22006])


# add relative and absolte unc:
GI5 = hlp.add_reluncertainties(GI5, 'area')
GI5['area'] = GI5.geometry.area

GI3 = hlp.getGI(fls_GI3, -18)
GI3['area'] = GI3.geometry.area
# add relative and absolte unc:
GI3 = hlp.add_reluncertaintiesGI3(GI3, 'area')
GI3['area'] = GI3.geometry.area


GI2 = hlp.getGI(fls_GI2, -4)
GI2.rename(columns={'Year':'yearGI2', 'area': 'area_GI2'}, inplace=True)
GI2['area'] = GI2.geometry.area
# add relative and absolte unc:
GI2 = hlp.add_reluncertaintiesGI3(GI2, 'area')
GI2['area'] = GI2.geometry.area

GI5 = GI5.sort_values(by='region')
GI3 = GI3.sort_values(by='region')
GI2 = GI2.sort_values(by='region')


GI5_large = GI5.loc[GI5['area']>(0.01*1e6)]
GI3_large = GI3.loc[GI3['area']>(0.01*1e6)]
GI2_large = GI2.loc[GI2['area']>(0.01*1e6)]

print(GI2.region.unique())
print(GI3.region.unique())
print(GI5.region.unique())

wYGI2 = hlp.arWeightYear(GI2)
wYGI3 = hlp.arWeightYear(GI3)
wYGI5 = hlp.arWeightYear(GI5)

GI4 = gpd.read_file('/Users/leahartl/Desktop/svenja/GI_4_2015/GI_4_2015.shp')
GI4 = GI4.to_crs(GI5.crs)
GI4['area'] = GI4.geometry.area
GI4_large = GI4.loc[GI4['area']>(0.01*1e6)]

rgi7 = gpd.read_file('/Users/leahartl/Desktop/inventare_2025/RGI2000-v7.0-G-11_central_europe/RGI2000-v7.0-G-11_central_europe.shp')
rgi7 = rgi7.to_crs(GI5.crs)
austria = gpd.read_file('/Users/leahartl/Desktop/inventare_2025/VGD_Oesterreich_gst_20221002/VGD.shp')
#https://www.data.gv.at/katalog/dataset/51bdc6dc-25ae-41de-b8f3-938f9056af62#resources

inv_gabi = gpd.read_file('/Users/leahartl/Desktop/inventare_2025/c3s_gi_rgi11_s2_2015_v2/c3s_gi_rgi11_s2_2015_v2.shp')
inv_gabi = inv_gabi.to_crs(GI5.crs)


dglam1 = gpd.read_file('/Users/leahartl/Desktop/inventare_2025/preds_2015_dl4gam/inv_preds_calib.shp')
dglam2 = gpd.read_file('/Users/leahartl/Desktop/inventare_2025/preds_2023_dl4gam/2023_preds_calib.shp')

dglam1 = dglam1.to_crs(GI5.crs)
dglam2 = dglam2.to_crs(GI5.crs)

# otz1718 = gpd.read_file('/Users/leahartl/Desktop/inventare_2025/GI/Oetztaler_Alpen_GI5/Oetztaler_Alpen_GI5_pangaea.shp')
# otz1718 = otz1718.to_crs(GI5.crs)




def intersect_w_states(inv, austria):
    inv['centroid'] = inv.geometry.centroid #Create a centroid point column
    inv['polygeom'] = inv.geometry #Save the polygon geometry to switch back to after the join
    inv = inv.set_geometry('centroid')
    inv_AT = gpd.sjoin(inv, austria.to_crs(crs=inv.crs), how="inner", predicate="intersects")
    inv_AT = inv_AT.set_geometry('polygeom')
    inv_AT.rename({'polygeom':'geometry'}, inplace=True)
    inv_AT = inv_AT.set_geometry('geometry')
    inv_AT.drop(columns=['centroid', 'polygeom'], inplace=True)
    inv_AT['area'] = inv_AT.geometry.area

    return(inv_AT)


sommer = sommer.to_crs(GI5.crs)
sommer = intersect_w_states(sommer, austria)
sommer['area'] = sommer.geometry.area
sommer['area_km'] = sommer.geometry.area*1e-6
sommer1 = sommer.loc[sommer.invNr==1]
sommer2 = sommer.loc[sommer.invNr==2]
sommer3 = sommer.loc[sommer.invNr==3]

sommer1.to_file('/Users/leahartl/Desktop/inventare_2025/Data/glims/sommer1.gpkg')
sommer2.to_file('/Users/leahartl/Desktop/inventare_2025/Data/glims/sommer2.gpkg')
sommer3.to_file('/Users/leahartl/Desktop/inventare_2025/Data/glims/sommer3.gpkg')
print(sommer3.sort_values(by='area_km'))
print(sommer1.sort_values(by='area_km'))

print(sommer1.year.min(), sommer1.year.max())
print(sommer2.year.min(), sommer2.year.max())
print(sommer3.year.min(), sommer3.year.max())

stop

inv_gabi_AT = intersect_w_states(inv_gabi, austria)
inv_gabi_AT['area'] = inv_gabi_AT.geometry.area
# add year column for c3sInv: 
inv_gabi_AT['Date'] = pd.to_datetime(inv_gabi_AT['Date'].astype(str))
inv_gabi_AT['year'] = inv_gabi_AT['Date'].dt.year

wYC3S = hlp.arWeightYear(inv_gabi_AT)

inv_gabi_AT.to_file('/Users/leahartl/Desktop/inventare_2025/Data/inv_gabi_AT_Paul2020.geojson')

rgiAT = intersect_w_states(rgi7, austria)
rgiAT['area'] = rgiAT.geometry.area
# rgiAT.to_file('/Users/leahartl/Desktop/inventare_2025/Data/rgiAT.geojson')

# GI2.to_file('/Users/leahartl/Desktop/inventare_2025/Data/GI2.geojson')
# GI3.to_file('/Users/leahartl/Desktop/inventare_2025/Data/GI3.geojson')
# GI5.to_file('/Users/leahartl/Desktop/inventare_2025/Data/GI5.geojson')

dglam1 = intersect_w_states(dglam1, austria)
dglam1['area'] = dglam1.geometry.area

dglam2 = intersect_w_states(dglam2, austria)
dglam2['area'] = dglam2.geometry.area



df = pd.DataFrame(columns=[wYGI2, wYGI3, wYGI5], index=['area', 'unc'])
df.loc['area', wYGI2] = GI2['area'].sum()*1e-6
df.loc['unc', wYGI2] = GI2['unc_abs_GI3'].sum()*1e-6
df.loc['n', wYGI2] = len(GI2['id'].unique())

df.loc['area', wYGI3] = GI3['area'].sum()*1e-6
df.loc['unc', wYGI3] = GI3['unc_abs_GI3'].sum()*1e-6
df.loc['n', wYGI3] = len(GI3['id'].unique())

df.loc['area', wYGI5] = GI5['area'].sum()*1e-6
df.loc['unc', wYGI5] = GI5['unc_abs_GI5'].sum()*1e-6
df.loc['n', wYGI5] = len(GI5['id'].unique())
df = df.T
print(df)

df_L = pd.DataFrame(columns=[wYGI2, wYGI3, wYGI5], index=['area', 'unc'])
df_L.loc['area', wYGI2] = GI2_large['area'].sum()*1e-6
df_L.loc['unc', wYGI2] = GI2_large['unc_abs_GI3'].sum()*1e-6
df_L.loc['n', wYGI2] = len(GI2_large['id'].unique())

df_L.loc['area', wYGI3] = GI3_large['area'].sum()*1e-6
df_L.loc['unc', wYGI3] = GI3_large['unc_abs_GI3'].sum()*1e-6
df_L.loc['n', wYGI3] = len(GI3_large['id'].unique())

df_L.loc['area', wYGI5] = GI5_large['area'].sum()*1e-6
df_L.loc['unc', wYGI5] = GI5_large['unc_abs_GI5'].sum()*1e-6
df_L.loc['n', wYGI5] = len(GI5_large['id'].unique())
df_L = df_L.T
print(df_L)


GI4df = pd.DataFrame(columns=[2015], index=['area', 'unc'])
GI4df.loc['area', 2015] = GI4['area'].sum()*1e-6
GI4df.loc['n', 2015] = len(GI4['nr'].unique())
GI4df = GI4df.T

GI4df_L = pd.DataFrame(columns=[2015], index=['area', 'unc'])
GI4df_L.loc['area', 2015] = GI4_large['area'].sum()*1e-6
GI4df_L.loc['n', 2015] = len(GI4_large['nr'].unique())
GI4df_L = GI4df_L.T

C3sdf = pd.DataFrame(columns=[wYC3S], index=['area', 'unc'])
C3sdf.loc['area', wYC3S] = inv_gabi_AT['area'].sum()*1e-6
C3sdf.loc['n', wYC3S] = len(inv_gabi_AT['GLACIER_NR'].unique())
C3sdf = C3sdf.T


rgidf = pd.DataFrame(columns=[2003], index=['area', 'unc'])
rgidf.loc['area', 2003] = rgiAT['area'].sum()*1e-6
rgidf.loc['n', 2003] = len(rgiAT['rgi_id'].unique())
rgidf = rgidf.T



# sommer
wYS1 = hlp.arWeightYear(sommer1)
wYS2 = hlp.arWeightYear(sommer2)
wYS3 = hlp.arWeightYear(sommer3)

df_som = pd.DataFrame(columns=[wYS1, wYS2, wYS3], index=['area', 'unc'])
df_som.loc['area', wYS1] = sommer1['area'].sum()*1e-6
df_som.loc['unc', wYS1] = 0#sommer1['unc_abs'].sum()*1e-6
df_som.loc['n', wYS1] = len(sommer1['glac_id'].unique())

df_som.loc['area', wYS2] = sommer2['area'].sum()*1e-6
df_som.loc['unc', wYS2] = 0#sommer2['unc_abs'].sum()*1e-6
df_som.loc['n', wYS2] = len(sommer2['glac_id'].unique())

df_som.loc['area', wYS3] = sommer3['area'].sum()*1e-6
df_som.loc['unc', wYS3] = 0#sommer3['unc_abs'].sum()*1e-6
df_som.loc['n', wYS3] = len(sommer3['glac_id'].unique())
df_som = df_som.T
print('sommer: ', df_som)


# dglam
# dglamdf = pd.DataFrame(columns=[2015, 2023], index=['area', 'unc'])
# dglamdf.loc['area', 2015] = 353.81#dglam1['area'].sum()*1e-6
# dglamdf.loc['n', 2015] = len(dglam1['entry_id'].unique())
# dglamdf.loc['area', 2023] = 294.78#dglam2['area'].sum()*1e-6
# dglamdf.loc['n', 2023] = len(dglam2['entry_id'].unique())
# dglamdf = dglamdf.T

# overwrite w codrut's nrs
dglamdf = pd.DataFrame(columns=[wYC3S, 2023], index=['area', 'unc'])
dglamdf.loc['area', wYC3S] = 293.83
dglamdf.loc['unc', wYC3S] = 36.48
dglamdf.loc['n', wYC3S] = 411
dglamdf.loc['area', 2023] = 240.70
dglamdf.loc['unc', 2023] = 45.72
dglamdf.loc['n', 2023] = 398
dglamdf = dglamdf.T

# print(df)
# print(GI4df)
# print(C3sdf)
# print(rgidf)


fig, ax = plt.subplots(1, 1, figsize=(8, 6))
# ax = ax.flatten()
ax.errorbar(df.index, df.area, yerr=df.unc, marker='o', markersize=8, linestyle='--', linewidth=1, color='k', label='AGI2, AGI3, AGI5')
ax.errorbar(GI4df.index, GI4df.area, yerr=0, marker='o', markersize=8, linestyle='', color='slateblue', label='AGI4; Buckel et al, 2018')
ax.errorbar(df_som.index, df_som.area, yerr=df_som.unc, marker='o', markersize=8, linestyle='--', linewidth=1, color='OliveDrab', label='Sommer et al, 2020')

ax.errorbar(C3sdf.index, C3sdf.area, yerr=C3sdf.area*0.033, marker='o', markersize=8, linestyle='', color='darkred', label='Paul et al, 2020')
ax.errorbar(rgidf.index, rgidf.area, yerr=rgidf.area*0.104, marker='o', markersize=8, linestyle='', color='grey', label='RGI 7.0, 2023')
ax.errorbar(dglamdf.index, dglamdf.area, yerr=dglamdf.unc, marker='o', markersize=8,  linestyle='--', linewidth=1.2 ,color='orange', label='Diaconu et al, 2025')

# too small:
# ax.errorbar(df_L.index, df_L.area, yerr=df_L.unc, markerfacecolor='none',markeredgecolor='k', marker='o', markersize=8, linestyle='--', linewidth=1, color='k', label='AGI1,2,3; >0.01 km2')
# ax.errorbar(GI4df_L.index, GI4df_L.area, yerr=0, marker='o', markerfacecolor='none',markersize=8, linestyle='', markeredgecolor='slateblue', label='AGI4; > 0.01km2')

ax.set_xlim(1995, 2025)


for txt, txt2, x, y in zip(df['n'].values[:-1],df_L['n'].values[:-1], df.index.values[:-1], df['area'].values[:-1]):
    ax.annotate('n= '+str(txt)+'('+str(txt2)+')', (x, y), xytext=(x+0.5, y+0.1))
for txt, txt2, x, y in zip(df['n'].values[-1:], df_L['n'].values[-1:], df.index.values[-1:], df['area'].values[-1:]):
    ax.annotate('n= '+str(txt)+'\n('+str(txt2)+')', (x, y), xytext=(x-1.5, y+25))

for txt, txt2, x, y in zip(GI4df['n'].values, GI4df_L['n'].values, GI4df.index.values, GI4df['area'].values):
    ax.annotate('n= '+str(txt)+'\n('+str(txt2)+')', (x, y), xytext=(x-2.2, y-4), color='slateblue')


for txt, x, y in zip(C3sdf['n'].values, C3sdf.index.values, C3sdf['area'].values):
    ax.annotate('n= '+str(txt), (x, y), xytext=(x+0.5, y+0.05), color='darkred')

for txt, x, y in zip(rgidf['n'].values, rgidf.index.values, rgidf['area'].values):
    ax.annotate('n= '+str(txt), (x, y), xytext=(x+0.5, y+15), color='grey')


for txt, x, y in zip(df_som['n'].values, df_som.index.values, df_som['area'].values):
    ax.annotate('n= '+str(txt), (x, y), xytext=(x-2.1, y-20), color='OliveDrab')

# currently not including numbers for DGLAM to avoid detailed explanations/confusion in the caption
# for txt, x, y in zip(dglamdf['n'].values, dglamdf.index.values, dglamdf['area'].values):
#     ax.annotate('n= '+str(txt), (x, y), xytext=(x-2.6, y-14))

ax.legend()
ax.set_ylabel('Glacier area in Austria [km$^2$]', fontsize=12)
ax.set_xlabel('Year', fontsize=12)
ax.grid('both')


fig.savefig('figures/compare_inventories.png', dpi=200, bbox_inches='tight')
plt.show()





