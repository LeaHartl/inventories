
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


# orthos Seekarlesf:
SK_L = '/Users/leahartl/Desktop/inventare_2025/Data/seekarlesF_Large.tif'
SK_N = '/Users/leahartl/Desktop/inventare_2025/Data/seekarlesF_sml_N.tif'
SK_S = '/Users/leahartl/Desktop/inventare_2025/Data/seekarlesF_sml_S.tif'

agi5 = gpd.read_file('/Users/leahartl/Desktop/inventare_2025/mergedfiles/split_vanishing/Oetztaler_Alpen_GI5_proc3.geojson')
agi5SK = agi5.loc[agi5['id'] == 14033]


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
    dat_normal_exp = dat_exp.loc[dat_exp['rdif'].abs() >= 5]
    dat_circ = dat_exp.loc[dat_exp['rdif'].abs() < 5]

    # find IDs that have both non-circles and circles:
    # group by ID number and get minimum rdif per ID
    check = dat_exp.groupby('id')['rdif'].apply(np.minimum.reduce).reset_index(name='min')
    # group by ID number and get maximum rdif per ID
    check['max'] = dat_exp.groupby('id')['rdif'].apply(np.maximum.reduce).reset_index(name='max')['max']


    # filter ID numbers that have both min rdif < 5 (circle) and max rdif >= 5 (non circle)
    # currently excluding outline_qf criterion. to inlcude it, add: OR outline_qf = 3
    vanishing_fragments = check.loc[((check['min'].abs() < 5) & (check['max'].abs() >= 5))]# | (check['outline_qf']==3)]

    print(vanishing_fragments)
    print('# vanishing fragments (circles, other):', len(vanishing_fragments))

    # get the polygons that meet the above criteria and save to file:
    vanishing_fragments = dat_exp.loc[dat_exp['id'].isin(vanishing_fragments['id'])]
    #only circles:
    vanishing_fragments = vanishing_fragments.loc[vanishing_fragments['rdif'].abs() < 5]
    #vanishing_fragments.to_file(folder_new+'split_vanishing/'+r+'_GI5_vanishingfragments.geojson')


    ## deal with ID numbers ("glaciers") that have no "normal" parts, only circles our outline qf 3:
    # find IDs that have only "vanishing feature" circles, no normally mapped non-circle polygons:

    # Depending on usage, adjust the filters for vanishing and vanished to include or exclude the QF! 
    # current version excludes QF 3:
    vanishing_glaciers = check.loc[(check['max'].abs() < 5)]# | (check['outline_qf'] == 3)]

    print(vanishing_glaciers)
    print(r'# vanishing glaciers:', len(vanishing_glaciers))
    dat_vanishing = dat_exp.loc[dat_exp['id'].isin(vanishing_glaciers['id'])]


    # work on the polygons that are not circles:
    # make sure all geometries are valid:
    dat_normal_exp.geometry = dat_normal_exp.geometry.make_valid()

    # re-dissolve on id:
    dat_normal = dat_normal_exp.dissolve(by='id')
    # ensure the quality flags from the original file are preserved: 
    # keep only the geometry and index (ID) of the dissolved file
    dat_normal = dat_normal[['geometry']]
    # merge on the id number with the original file:
    dat_normal = dat_normal.merge(dat.drop(columns='geometry'), left_on='id', right_on='id', how='inner')

  
    # count vanishng fragments and add in a new column: 
    group_van = vanishing_fragments[['name', 'id']].groupby('id').count()
    group_normal = dat_normal_exp[['name', 'id']].groupby('id').count()
    group_normal = group_normal.rename(columns={'name': 'nr_frag'})
    mergegroup = group_van.merge(group_normal, right_index=True, left_index=True, how='outer')
    mergegroup['van_fragm'] = mergegroup['name']-mergegroup['nr_frag']
    dat_normal = dat_normal.merge(mergegroup['van_fragm'], left_on='id', right_index=True)

    # number of normally mapped fragments (not circles)
    dat_normal = dat_normal.merge(group_normal['nr_frag'], left_on='id', right_index=True)

    print(mergegroup)
    print(dat_normal)


    # save vanishing glaciers to file (circle polygons):
    #GI5_vanishing.drop(columns=['area_exp', 'length_exp', 'r1', 'r2', 'rdif'], inplace=True)
    # GI5_vanishing.to_file(folder_new+'split_vanishing/'+r+'_GI5_vanishing_polygon.geojson')

    # uncomment to save centroids to file:
    # GI5_vanishing_point = GI5_vanishing.copy()
    # GI5_vanishing_point.geometry = GI5_vanishing_point.centroid
    # GI5_vanishing_point.to_file(folder_new+'split_vanishing/'+r+'_2023_vanishing_point.geojson')

    return(dat_normal, vanishing_fragments)


def getbox(bounds, refcrs):
    box1 = box(bounds[0], bounds[1], bounds[2], bounds[3])
    df = pd.DataFrame(columns=['box'])
    df['box'] = 'box'
    geodf = gpd.GeoDataFrame(df, geometry=[box1])
    geodf.set_crs(refcrs, inplace=True)
    # geodf = geodf.to_crs(epsg=4326)
    print(geodf)
    return (geodf)


# get files from folder
def getdata(fls):
    df = pd.DataFrame(columns=['id', 'name', ])
    # for f in fls:
    df['fn'] = fls

    gdfs = []
    van = []
    for f in fls:
        f_list = f.split('/')
        df.loc[df.fn == f, 'who'] = f_list[-1][0:-19]
        # df.loc[df.fn == f, 'id'] = f_list[1].split('/')[1]
        # df.loc[df.fn == f, 'glID'] = f_list[2]
        # df.loc[df.fn == f, 'year'] = f_list[3]

        dat = gpd.read_file(f)
        # print(dat.crs)
        dat = dat.to_crs(epsg=31287)
        print(dat.head())
        if 'Id' in dat.columns:
            dat.rename(columns={'Id':'id'}, inplace=True)
        if 'name' not in dat.columns:
            dat['name'] = ''

        dat.geometry = dat.geometry.make_valid()

        dat, vanishing = circles(dat)

        dat['area'] = dat.geometry.area
        #dat.loc[dat['iscircle'] == 'yes', 'area'] = 0
        dat['who'] = f_list[-1][0:-19]
        vanishing['who'] = f_list[-1][0:-19]
        # dat = dat.dissolve(by='id')

        gdfs.append(dat)

        van.append(vanishing)

        print(dat)

    
    gdf = pd.concat(gdfs)
    vangdf = pd.concat(van)
    # print(gdf)
    # stop


    return(gdf, vangdf)

gdf, vanishing = getdata(fls)

print(gdf, vanishing[['id','name','who']])

# f, ax = plt.subplots()
# vanishing.plot(ax=ax)
# plt.show()
# stop

def plts(gdf, SK_L, SK_N, SK_S, vanishing, agi5SK):
    fig = plt.figure(figsize=(9, 8))#, layout="constrained")

    gs = GridSpec(2, 3, figure=fig)
    ax = fig.add_subplot(gs[0, 0:])

    sk = gdf.loc[gdf['id']==14033]
    print(sk)
    # fig, ax = plt.subplot_mosaic([['a', 'b'], ['c', 'd']], figsize=(9, 6), layout='constrained')
    # fig, ax = plt.subplots(1, 1, layout='constrained', figsize=(10, 7))
    # ax_ins0 = ax.inset_axes([0, -1.3, 1.2, 1.0])
    
    ax_ins0 = fig.add_subplot(gs[1, 0:])
    ax_ins1 = ax.inset_axes([0.94, -1.2, 1.0, 1.8])
    
    # wurt
    with rio.open(SK_L) as src1:
        s = src1.read()
        sk = sk.to_crs(src1.crs)
        print(src1.crs)
        show(s, transform=src1.transform, ax=ax)
        #show(s, transform=src1.transform, ax=ax_ins0)
        bounds = src1.bounds

    with rio.open(SK_N) as src2:
        s2 = src2.read()
        # sk = sk.to_crs(src1.crs)
        # print(src1.crs)
        show(s2, transform=src2.transform, ax=ax_ins0)
        #show(s, transform=src1.transform, ax=ax_ins0)
        bounds_s2 = src2.bounds

    with rio.open(SK_S) as src3:
        s3 = src3.read()
        # sk = sk.to_crs(src1.crs)
        # print(src1.crs)
        show(s3, transform=src3.transform, ax=ax_ins1)
        #show(s, transform=src1.transform, ax=ax_ins0)
        bounds_s3 = src3.bounds

    sk.boundary.plot(ax=ax, color='cyan', linewidth=0.5)
    vanishing.centroid.plot(ax=ax, color='cyan', marker='*', markersize=300, edgecolor='k')
    sk.boundary.plot(ax=ax_ins0, color='cyan', linewidth=0.5)
    sk.boundary.plot(ax=ax_ins1, color='cyan', linewidth=0.5)


    ax.set_xlim(207250, 209300)
    ax.set_ylim(344300, bounds[3])
    ax.set_xticks([207500, 208000, 208500, 209000])


    ax_ins0.set_xlim(208150, 208420)
    ax_ins0.set_ylim(345040, 345200)
    
    ax_ins1.set_xlim(208270, 208540)
    ax_ins1.set_ylim(bounds_s3[1], bounds_s3[3])


    agi5SK = agi5SK.to_crs(src1.crs)
    buffout = agi5SK.geometry.buffer(2)
    buffin = agi5SK.geometry.buffer(-2)

    buffLout = agi5SK.geometry.buffer(20)
    buffLin = agi5SK.geometry.buffer(-20)

    ax.add_artist(ScaleBar(dx=1, location="lower right", font_properties={"size": 14}))
    ax_ins0.add_artist(ScaleBar(dx=1, location="lower right", font_properties={"size": 14}))
    ax_ins1.add_artist(ScaleBar(dx=1, location="upper left", font_properties={"size": 14}))

    ax.set_yticks([344600, 345000, 345400])
    ax.tick_params(axis='y', labelrotation=90)


    box1 = getbox([208150, 345040, 208420, 345200], sk.crs)
    box2 = getbox([208270, bounds_s3[1], 208540, bounds_s3[3]], sk.crs)

    box1.boundary.plot(ax=ax, color='red')
    box2.boundary.plot(ax=ax, color='red')


    for a in [ax_ins0, ax_ins1]:
        a.set_xticks([])
        a.set_yticks([])
        a.set_xticklabels('')
        a.set_yticklabels('')

        buffout.boundary.plot(ax=a, color='tomato', alpha=0.8)
        buffin.boundary.plot(ax=a, color='tomato', alpha=0.8)

        buffLout.boundary.plot(ax=a, color='tomato', alpha=0.8, linestyle='--')
        buffLin.boundary.plot(ax=a, color='tomato', alpha=0.8, linestyle='--')


    ax.annotate("a",
             xy=(0.05, 0.92), xycoords='axes fraction', fontsize=14,
             ha="center", va="center",
             bbox=dict(boxstyle="square,pad=0.2",
             fc="silver", ec="k", lw=2))
    ax_ins0.annotate("b",
             xy=(0.05, 0.92), xycoords='axes fraction', fontsize=14,
             ha="center", va="center",
             bbox=dict(boxstyle="square,pad=0.2",
             fc="silver", ec="k", lw=2))
    ax_ins1.annotate("c",
             xy=(0.05, 0.88), xycoords='axes fraction', fontsize=14,
             ha="center", va="center",
             bbox=dict(boxstyle="square,pad=0.2",
             fc="silver", ec="k", lw=2))



    
    RR = Line2D([0], [0], label='RR outlines', color='cyan', linewidth=1, linestyle='-')
    star = Line2D([0], [0], label='Vanishing fragment', color='cyan', marker='*', linestyle='', markersize=12, markeredgecolor='k')
    buf1 = Line2D([0], [0], label='±2 m buffer', color='tomato', linewidth=1, linestyle='-')
    buf2 = Line2D([0], [0], label='±20 m buffer', color='tomato', linewidth=1, linestyle='--')
    hls = [RR, star, buf1, buf2]

    fig.legend(handles=hls, loc='upper left', bbox_to_anchor=(0.80, 0.86), ncol=2)
    # plt.tight_layout()
    fig.savefig('figures/RR_exmp_2.png', bbox_inches='tight', dpi=200)
    plt.show()


plts(gdf, SK_L, SK_N, SK_S, vanishing, agi5SK)


    