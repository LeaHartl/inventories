
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


# orthos
vern_L = '/Users/leahartl/Desktop/inventare_2025/Data/Vernagt_L.tif'
vern_S = '/Users/leahartl/Desktop/inventare_2025/Data/Vernagt_S.tif'

def getbox(bounds, refcrs):
    box1 = box(bounds[0], bounds[1], bounds[2], bounds[3])
    df = pd.DataFrame(columns=['box'])
    df['box'] = 'box'
    geodf = gpd.GeoDataFrame(df, geometry=[box1])
    geodf.set_crs(refcrs, inplace=True)
    # geodf = geodf.to_crs(epsg=4326)

    return (geodf)



def plts(orthoL, orthoS):
    fig, ax = plt.subplots(1, 1, figsize=(7,7))#, sharex=True)
    # ax = ax.flatten()
    MS = gpd.read_file('out/subset_oetztal_MS.geojson')
    
    IGF = gpd.read_file('out/subset_oetztal_IGF.geojson')
    

    # fig, ax = plt.subplot_mosaic([['a', 'b'], ['c', 'd']], figsize=(9, 6), layout='constrained')
    # fig, ax = plt.subplots(1, 1, layout='constrained', figsize=(10, 7))
    ax_ins0 = ax.inset_axes([0.0, 1.1, 0.7, 0.7])

    # wurt
    with rio.open(orthoS) as src1:
        s = src1.read()
        MS = MS.to_crs(src1.crs)
        IGF = IGF.to_crs(src1.crs)
        print(src1.crs)
        show(s, transform=src1.transform, ax=ax)
        # show(s, transform=src1.transform, ax=ax_ins0)
        bounds = src1.bounds

    with rio.open(orthoL) as src2:
        s2 = src2.read()
        show(s2, transform=src2.transform, ax=ax_ins0)
        bounds2 = src2.bounds


    
    MS.boundary.plot(ax=ax, color='blue')#, linewidth=1)
    IGF.boundary.plot(ax=ax, color='cyan')#, linewidth=1)
    
   
    MS.boundary.plot(ax=ax_ins0, color='blue')
    IGF.boundary.plot(ax=ax_ins0, color='cyan')
    

    ax.set_xlim(bounds[0], bounds[2])
    ax.set_ylim(bounds[1], bounds[3])

    ax_ins0.set_xlim(bounds2[0], bounds2[2])
    ax_ins0.set_ylim(bounds2[1], bounds2[3])


    box = getbox(bounds, IGF.crs)
    # box = box.to_crs(wurt.crs)
    box.boundary.plot(ax=ax_ins0, color='red')

    ax.add_artist(ScaleBar(dx=1, location="lower right", font_properties={"size": 10}))
    ax_ins0.add_artist(ScaleBar(dx=1, location="lower right", font_properties={"size": 10}))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels('')
    ax.set_yticklabels('')


    ax.annotate("b",
             xy=(0.05, 0.92), xycoords='axes fraction', fontsize=14,
             ha="center", va="center",
             bbox=dict(boxstyle="square,pad=0.2",
             fc="silver", ec="k", lw=2))
    ax_ins0.annotate("a",
             xy=(0.05, 0.92), xycoords='axes fraction', fontsize=14,
             ha="center", va="center",
             bbox=dict(boxstyle="square,pad=0.2",
             fc="silver", ec="k", lw=2))

    l1 = Line2D([0], [0], label='Analyst A', color='blue', linewidth=1, linestyle='-')
    l2 = Line2D([0], [0], label='Analyst B', color='cyan', linewidth=1, linestyle='-')
    
    

    # plt.tight_layout()
    fig.legend([l1, l2], ['Analyst A', 'Analyst B'], loc='upper right', bbox_to_anchor=(0.9, 0.9))
    fig.savefig('figures/Vernagt.png', bbox_inches='tight', dpi=200)
    plt.show()


plts(vern_L, vern_S)

    