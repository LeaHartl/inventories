
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


# ortho Wurtenkees:
SK_L = '/Users/leahartl/Desktop/inventare_2025/Data/seekarlesF_Large.tif'
SK_N = '/Users/leahartl/Desktop/inventare_2025/Data/seekarlesF_sml_N.tif'
SK_S = '/Users/leahartl/Desktop/inventare_2025/Data/seekarlesF_sml_S.tif'

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

        dat = circles(dat)

        dat['area'] = dat.geometry.area
        dat.loc[dat['iscircle'] == 'yes', 'area'] = 0
        dat['who'] = f_list[-1][0:-19]
        # dat = dat.dissolve(by='id')

        gdfs.append(dat)

        print(dat)

    
    gdf = pd.concat(gdfs)
    # print(gdf)
    # stop


    return(gdf)

# gdf = getdata(fls)

# print(gdf)


def plts(orthoW, orthoSoldL, orthoSoldS):
    fig, ax = plt.subplots(2, 1, figsize=(7,7))#, sharex=True)
    ax = ax.flatten()
    wurt = gpd.read_file('/Users/leahartl/Desktop/inventare_2025/mergedfiles/split_vanishing/Sonnblickgruppe_GI5_proc2.geojson')
    
    otz = gpd.read_file('/Users/leahartl/Desktop/inventare_2025/mergedfiles/split_vanishing/Oetztaler_Alpen_GI5_proc2.geojson')
    

    # fig, ax = plt.subplot_mosaic([['a', 'b'], ['c', 'd']], figsize=(9, 6), layout='constrained')
    # fig, ax = plt.subplots(1, 1, layout='constrained', figsize=(10, 7))
    ax_ins0 = ax[0].inset_axes([1.08, 0.0, 1.2, 1.0])
    # wurt
    with rio.open(orthoW) as src1:
        s = src1.read()
        wurt = wurt.to_crs(src1.crs)
        print(src1.crs)
        show(s, transform=src1.transform, ax=ax[0])
        show(s, transform=src1.transform, ax=ax_ins0)
        bounds = src1.bounds

    wurt.boundary.plot(ax=ax[0], color='cyan')
    wurt.boundary.plot(ax=ax_ins0, color='cyan')
    # ax[0].set_xlim(bounds[0], bounds[2])
    # ax[0].set_ylim(bounds[1], bounds[3])
    
    ax[0].set_xlim(425356, 426289)
    ax[0].set_ylim(210342, 211181)

    # ax[0].set_xlim(425887, 426262)
    # ax[0].set_ylim(210605, 211000)
    #smaller:
    ax_ins0.set_xlim(426023, 426228)
    ax_ins0.set_ylim(210712, 210890)
    # pritn(wurt.crs)

    box = getbox([426023, 210712, 426228, 210890], wurt.crs)
    # box = box.to_crs(wurt.crs)
    box.boundary.plot(ax=ax[0], color='red')

    ax[0].add_artist(ScaleBar(dx=1, location="lower right", font_properties={"size": 14}))
    ax_ins0.add_artist(ScaleBar(dx=1, location="lower right", font_properties={"size": 14}))

    ax_ins1 = ax[1].inset_axes([1.08, -0.1, 1.2, 1.2])

    with rio.open(sold_L) as src2:
        s = src2.read()
        otz = otz.to_crs(src2.crs)
        print(src1.crs)
        show(s, transform=src2.transform, ax=ax[1])
        bounds2 = src2.bounds

    with rio.open(sold_S) as src3:
        s = src3.read()
        otz = otz.to_crs(src3.crs)
        show(s, transform=src3.transform, ax=ax_ins1)
        bounds3 = src3.bounds

    otz.boundary.plot(ax=ax[1], color='cyan')
    otz.boundary.plot(ax=ax_ins1, color='cyan')

    ax[1].set_ylim(bounds2[1], bounds2[3])
    ax[1].set_xlim(bounds2[0], 218100)#481)

    ax_ins1.set_ylim(338005, 338800)
    ax_ins1.set_xlim(217005, 218070)

    # box2 = getbox([338005, 217005, 338800, 218070], otz.crs)
    box2 = getbox([217005, 338005, 218070, 338800], otz.crs)
    # box = box.to_crs(wurt.crs)
    box2.boundary.plot(ax=ax[1], color='red')


    for a in [ax[0], ax_ins0, ax[1], ax_ins1]:
        a.set_xticks([])
        a.set_yticks([])
        a.set_xticklabels('')
        a.set_yticklabels('')

    # ax[0].indicate_inset_zoom(ax_ins0, edgecolor="black")
    ax[1].add_artist(ScaleBar(dx=1, location="lower left", font_properties={"size": 14}))
    ax_ins1.add_artist(ScaleBar(dx=1, location="lower right", font_properties={"size": 14}))

    
    ax[0].annotate("a",
             xy=(0.05, 0.92), xycoords='axes fraction', fontsize=14,
             ha="center", va="center",
             bbox=dict(boxstyle="square,pad=0.2",
             fc="silver", ec="k", lw=2))
    ax_ins0.annotate("b",
             xy=(0.05, 0.92), xycoords='axes fraction', fontsize=14,
             ha="center", va="center",
             bbox=dict(boxstyle="square,pad=0.2",
             fc="silver", ec="k", lw=2))
    ax[1].annotate("c",
             xy=(0.05, 0.92), xycoords='axes fraction', fontsize=14,
             ha="center", va="center",
             bbox=dict(boxstyle="square,pad=0.2",
             fc="silver", ec="k", lw=2))
    ax_ins1.annotate("d",
             xy=(0.05, 0.92), xycoords='axes fraction', fontsize=14,
             ha="center", va="center",
             bbox=dict(boxstyle="square,pad=0.2",
             fc="silver", ec="k", lw=2))

    plt.tight_layout()
    fig.savefig('figures/skiresorts.png', bbox_inches='tight', dpi=200)
    plt.show()


plts(wurt, sold_L, sold_S)

    