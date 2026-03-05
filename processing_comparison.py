
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches
import matplotlib.cm as cm
import matplotlib.colors as colors

import rioxarray as rxr
from rasterio.plot import show
import rasterio as rio
import glob
from matplotlib.gridspec import GridSpec

from shapely.geometry import box

from matplotlib_scalebar.scalebar import ScaleBar
# get the files:
# folder:
fldr = '/Users/leahartl/Desktop/inventare_2025/roundrobin/Upload_outlines_v1/'
# get all file paths in folder:
fls = glob.glob(fldr+'*.geojson')


# orthos:
# ortho_silv_12008 = '/Users/leahartl/Desktop/svenja/Silv_12008_data/Ortho2023.tif'
def circles(dat):
    dat_exp = dat.explode()
    # get area and boundary length of the individual polygons:
    dat_exp['area_exp'] = dat_exp.geometry.area
    dat_exp['length_exp'] = dat_exp.geometry.length

    # find circles based on radius calculations
    # compute radius of polygon from area (r1) and length (r2)
    dat_exp['r1'] = np.sqrt(dat_exp['area_exp'] / np.pi)
    dat_exp['r2'] = dat_exp['length_exp']/(2*np.pi)
    dat_exp['rdif'] = (100*(dat_exp['r1'] - dat_exp['r2'])/dat_exp['r1']).round(decimals=2)
    dat_exp['rdif'] = dat_exp['rdif'].abs()

    # rdif for circles is 0. 5% difference seems to reliably catch ellipses. Set threshold:
    # circles have rdif < 5:
    # remove circle polygons:
    dat_nocircles = dat_exp.loc[dat_exp['rdif'].abs() >= 5]

    # find IDs that have both non-circles and circles:
    # group by ID number and get minimum rdif per ID
    check = dat_exp.groupby('id')['rdif'].apply(np.minimum.reduce).reset_index(name='min')
    # group by ID number and get maximum rdif per ID
    check['max'] = dat_exp.groupby('id')['rdif'].apply(np.maximum.reduce).reset_index(name='max')['max']



    # filter ID numbers that have both min rdif < 5 (circle) and max rdif >= 5 (non circle)
    # currently excluding outline_qf criterion. to inlcude it, add: OR outline_qf = 3
    vanishing_fragments = check.loc[((check['min'].abs() < 5) & (check['max'].abs() >= 5))]# | (check['outline_qf']==3)]

    # print(vanishing\_fragments)
    # print(r, ', # vanishing fragments (circles, other):', len(vanishing_fragments))

    # get the polygons that meet the above criteria:
    vanishing_fragments = dat_exp.loc[dat_exp['id'].isin(vanishing_fragments['id'])]


    vanishing_glaciers = check.loc[(check['max'].abs() < 5)]# | (check['outline_qf'] == 3)]
    vanishing_glaciers.index = vanishing_glaciers['id'].values
    # print(vanishing_glaciers)

    # count vanishng fragments and add in a new column: 
    group_van = vanishing_fragments[['name', 'id']].groupby('id').count()
    group_normal = dat_nocircles[['name', 'id']].groupby('id').count()
    group_normal = group_normal.rename(columns={'name': 'nr_frag'})

    mergegroup = group_van.merge(group_normal, right_index=True, left_index=True, how='outer')
    mergegroup['van_fragm'] = mergegroup['name']-mergegroup['nr_frag']
    mergegroup['iscircle'] = 'no'
    vanishing_glaciers['iscircle'] = 'yes'

    # dat_all = mergegroup[['id', 'name', 'van_fragm', 'iscircle', 'geometry']].dissolve(by='id')
    # dat_all = pd.concat([dat_all, vanishing_glaciers])

    dat_all = dat_nocircles.merge(mergegroup[['van_fragm', 'iscircle']], left_on='id', right_index=True, how='outer')
    # dat_all.loc[dat_all['id'].isin(vanishing_glaciers['id'].values), 'iscircle'] = 'yes'


    dat_all = dat_all[['id', 'name', 'van_fragm', 'iscircle', 'geometry']].dissolve(by='id')
    dat_all = pd.concat([dat_all, vanishing_glaciers])

    return(dat_all)


def getbox(bounds):
    box1 = box(bounds[0], bounds[1], bounds[2], bounds[3])
    df = pd.DataFrame(columns=['box'])
    df['box'] = 'box'
    geodf = gpd.GeoDataFrame(df, geometry=[box1])
    # geodf.set_crs(epsg=31254,\ inplace=True)
    #geodf = geodf.to_crs(epsg=4326)
    print(geodf)
    return (geodf)


# get files from folder
def getdata(fls):
    df = pd.DataFrame(columns=['id', 'name', ])
    # for f in fls:
    df['fn'] = fls

    gdfs = []
    van = []
    norm = []
    for f in fls:
        f_list = f.split('/')
        df.loc[df.fn == f, 'who'] = f_list[-1][0:-19]

        dat = gpd.read_file(f)
        # print(dat.crs)
        dat = dat.to_crs(epsg=31287)
        print(dat.head())
        if 'Id' in dat.columns:
            dat.rename(columns={'Id':'id'}, inplace=True)
        if 'name' not in dat.columns:
            dat['name'] = ''

        dat.geometry = dat.geometry.make_valid()


        # normal, vanishing, dat = circles(dat)
        dat = circles(dat)

        dat['area'] = dat.geometry.area
        dat.loc[dat['iscircle'] == 'yes', 'area'] = 0
        dat['who'] = f_list[-1][0:-19]
        
        gdfs.append(dat)


    gdf = pd.concat(gdfs)
    # vangdf = pd.concat(van)
    # normgdf = pd.concat(norm)

    gdf.to_file('out/RR_nocirclrs.geojson')
    return(gdf)


gdf = getdata(fls)

gdf_cntrs = gdf[['name','geometry']].dissolve(by=gdf.index)
gdf_cntrs = gdf_cntrs.to_crs(epsg=4236)
gdf_cntrs['x'] = gdf_cntrs ['geometry'].centroid.x
gdf_cntrs['y'] = gdf_cntrs ['geometry'].centroid.y


gb = pd.DataFrame(columns=['medar', 'sd'])
gb['medar'] = (gdf['area'].groupby(gdf.index).median()*1e-6).astype(float).round(decimals=3).values
gb['sd'] = (gdf['area'].groupby(gdf.index).std()*1e-6).astype(float).round(decimals=3).values
gb['prc'] = (100*gb['sd']/gb['medar']).astype(float).round(decimals=1)
gb.index = gdf['area'].groupby(gdf.index).std().index
gb = gb.sort_values(by='medar')

gb = gb.merge(gdf_cntrs[['x', 'y']], left_index=True, right_index=True)
gb['xy'] = gb['x'].round(decimals=3).astype(str)+', '+gb['y'].round(decimals=3).astype(str)
print(gb)

gb.to_csv('out/RR_stats.csv')

names = gdf.who.unique()

alias = []
clrshex = ['#e6194B', '#000000', '#ffe119', '#4363d8', '#f58231',
     '#42d4f4', '#f032e6', '#fabed4', '#469990', '#dcbeff',
      '#9A6324', '#fffac8','#800000', '#aaffc3', '#000075', '#C0C0C0']#'#ffffff']#'#bfff00']#'#a9a9a9']#, '#ffffff']

for j, c in enumerate(clrshex):
    #     hx = colors.rgb2hex(c, keep_alpha=True)
    #     clrshex.append(hx)
    alias.append(j)
    # print(clrshex)



gdf['clrs'] = '0'
gdf['alias'] = '0'
for i, n in enumerate(names):
    gdf.loc[gdf['who']==n, 'clrs'] = clrshex[i]
    gdf.loc[gdf['who']==n, 'alias'] = alias[i]


gdf['area_km'] = gdf['area']*1e-6
# normal['area_km'] = normal['area']*1e-6
print(gdf[['id','who','alias', 'clrs', 'iscircle','geometry']])
print(gdf[['who','alias', 'clrs']].groupby('who').first())
# gdf[['who','alias', 'clrs']].groupby('who').first().to_csv('out/RoundRobin_numbers_people_colors.csv')

# stop

def plts(gdf):
    fig, ax = plt.subplots(2, 3, figsize=(9,6), sharex=True)

    gls = ['Madleinf.', 'NN', 'Arventalk.', 'Wurtenk.', 'Seekarlesf.', 'Pasterze']
    # get GI3:
    gi3 = gpd.read_file('/Users/leahartl/Desktop/inventare_2025/roundrobin/examples_RR_GI3.geojson')
    gi3.to_crs(epsg=31287, inplace=True)
    gi3['area_GI3_km'] = gi3.geometry.area*1e-6

    ax = ax.flatten()
    
    gdf = gdf.sort_values(by='area_km')
    lst = []
    gls = []
    for i, gl in enumerate(gdf.index.unique()):
        subset1 = gdf.loc[gdf.index==gl]
        gi3Sub = gi3.loc[gi3['id']==gl]
        print(subset1)

        n = subset1.loc[~subset1['name'].isnull(), 'name'].values[0]
        if n == 'Wurtenkees + Toteis':
            n = 'Wurten Kees'

        print(n)

        subset = gdf.loc[gdf.index==gl, 'area_km'].values
        # add extra 0 area for Madlein Ferner because JC did not put a polygon in his file but did categorize this as a vanished glacier.
        # check again to see if this causes any issues with the analyst attributions!!
        if gl == 13029:
            subset = np.append(subset, 0)
        lst.append(subset)
        gls.append(gl)

        parts = ax[i].violinplot(subset, showextrema=False)
        ax[i].set_xticklabels('')
        #ax[i].scatter(np.ones(len(subset)), subset1['area_km'], c=subset1['clrs'], edgecolor='k')
        for a in subset1['alias'].sort_values():
            ax[i].scatter(1, subset1.loc[subset1['alias']==a, 'area_km'], c=subset1.loc[subset1['alias']==a,'clrs'], edgecolor='k', label=a)
            
       
        # ax[i].set_title(n+' ('+str(int(gl))+' , n:' + str(len(subset))+')')
        ax[i].set_title(n+', n:' + str(len(subset)))

        if gl == 6013:
            ax[i].hlines(gi3Sub['area_GI3_km2'], 0.8, 1.2, colors='k', label='AGI3 area', linestyle='--')

        for pc in parts['bodies']:
            pc.set_facecolor('lightgrey')

    # bplot = ax.boxplot(lst,
    #                        tick_labels=gls)

    ax[0].set_ylabel('Area [km$^2$]')
    ax[2].legend(loc='upper left', title='Analysts:', ncol=8, bbox_to_anchor=(-2.5, 1.5))
    ax[3].set_ylabel('Area [km$^2$]')
    ax[1].set_yticks([0, 0.01, 0.02])
    ax[4].set_yticks([0.75, 0.80, 0.85])
    # ax.set_title('Round Robin')
    # ax.set_ylim(0, 1.5)
    # plt.tight_layout()
    for a, an in zip(ax, ['a', 'b', 'c', 'd', 'e', 'f']):
     a.annotate(an,
             xy=(0.08, 0.92), xycoords='axes fraction', fontsize=12,
             ha="center", va="center",
             bbox=dict(boxstyle="square,pad=0.2",
             fc="silver", ec="k", lw=2))

    fig.savefig('figures/roundrobin.png', bbox_inches='tight', dpi=200)



plts(gdf)
plt.show()
stop



