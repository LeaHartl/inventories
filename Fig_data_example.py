
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

import matplotlib.colors as mcolors


class MidpointNormalize(mcolors.Normalize):
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        mcolors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        v_ext = np.max( [ np.abs(self.vmin), np.abs(self.vmax) ] )
        x, y = [-v_ext, self.midpoint, v_ext], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))



# ortho = '/Users/leahartl/Desktop/svenja/Silv_12008_data/Ortho2023.tif'
# difdem = '/Users/leahartl/Desktop/svenja/Silv_12008_data/DifDem20232017_12008.tif'
# HS = '/Users/leahartl/Desktop/svenja/Silv_12008_data/HS2023_12008.tif'


ortho ='/Users/leahartl/Desktop/inventare_2025/Data/Beispieldaten_Grosselendkees/Ortho_2022_crop_reproj.tif'
HS1 = '/Users/leahartl/Desktop/inventare_2025/Data/Beispieldaten_Grosselendkees/HS_2010.tif'
HS2 = '/Users/leahartl/Desktop/inventare_2025/Data/Beispieldaten_Grosselendkees/HS_2023_reproj.tif'
difdem = '/Users/leahartl/Desktop/inventare_2025/Data/Beispieldaten_Grosselendkees/DoD_2023_2010_crop.tif'

AnkGI5 = gpd.read_file('/Users/leahartl/Desktop/inventare_2025/mergedfiles/split_vanishing/Ankogel_Hochalmspitzgruppe_GI5_proc2.geojson')
AnkGI3 = gpd.read_file('/Users/leahartl/Desktop/inventare_2025/GI/GI_3_new/Ankogel_Hochalmspitzgruppe_elevation.geojson')

oetzGI5 = gpd.read_file('/Users/leahartl/Desktop/inventare_2025/mergedfiles/split_vanishing/Oetztaler_Alpen_GI5_proc2.geojson')
oetzGI3 = gpd.read_file('/Users/leahartl/Desktop/inventare_2025/GI/GI_3_new/Oetztaler_Alpen_elevation.geojson')


fig, ax = plt.subplots(2, 2, figsize=(10, 7))
ax = ax.flatten()

norm = MidpointNormalize(midpoint =0, vmin=-20, vmax=5)
cmap = 'bwr_r'


# hillshade 1
with rio.open(HS1) as src2:
    s2 = src2.read()
    show(s2, transform=src2.transform, ax=ax[0], cmap='Greys_r',)
    # show(s2, transform=src2.transform, ax=ax[2], cmap='Greys_r',)
    bounds_s2 = src2.bounds
    print(src2.crs)


with rio.open(HS2) as src4:
    s4 = src4.read()
    show(s4, transform=src4.transform, ax=ax[1], cmap='Greys_r',)
    show(s4, transform=src4.transform, ax=ax[3], cmap='Greys_r',)# zorder=100)
    # show(s4, transform=src4.transform, ax=ax[3], cmap='Greys_r',)# zorder=100)
    bounds_s4 = src4.bounds
    print(src4.crs)

with rio.open(difdem) as src3:
    s3 = src3.read()
    # show(s3, transform=src3.transform, ax=ax[2])
    im = show(s3, transform=src3.transform, ax=ax[3], cmap=cmap, norm=norm, alpha=0.8)#, zorder=200)
    bounds_s3 = src3.bounds
    print(src3.crs)

with rio.open(ortho) as src1:
    s = src1.read()
    show(s, transform=src1.transform, ax=ax[2], alpha=1)
    bounds = src1.bounds
    print(src1.crs)


AnkGI3 = AnkGI3.to_crs(src1.crs)
AnkGI5 = AnkGI5.to_crs(src1.crs)

im1 = im.get_images()[1]
# fig.subplots_adjust(bottom=0.02)
cax = fig.add_axes([0.91, 0.12, 0.02, 0.32])
cbar = fig.colorbar(im1, cax=cax, orientation='vertical')
cbar.set_label('Elevation change [m]')


ax[0].add_artist(ScaleBar(dx=1, location="lower left", font_properties={"size": 12}))

for a in [ax[0], ax[1], ax[2], ax[3]]:
    a.set_xlim(bounds[0], 449980)#bounds[2])
    a.set_ylim(bounds[1], bounds[3])
    a.set_xticks([447000, 448000, 449000])#, 450000])

    AnkGI3.boundary.plot(ax=a, color='k', linestyle='--', linewidth=0.5, label='AGI 3')
    AnkGI5.boundary.plot(ax=a, color='k', linestyle='-', linewidth=0.5, label='AGI 5')

for a in [ax[1], ax[3]]:
    # a.set_xticks([])
    # a.set_yticks([])
    # a.set_xticklabels('')
    a.set_yticklabels('')
for a in [ax[0], ax[1]]:
    # a.set_xticks([])
    # a.set_yticks([])
    # a.set_xticklabels('')
    a.set_xticklabels('')

ax[0].set_title('Großelend Kees, hillshade 2010')
ax[1].set_title('Großelend Kees, hillshade 2023')
ax[3].set_title('Großelend Kees, DoD 2023-2010')
ax[2].set_title('Großelend Kees, orthophoto 2022')

ax[2].legend(loc='upper center')

ax[0].annotate("a",
         xy=(0.05, 0.92), xycoords='axes fraction', fontsize=14,
         ha="center", va="center",
         bbox=dict(boxstyle="square,pad=0.2",
         fc="silver", ec="k", lw=2))
ax[1].annotate("b",
         xy=(0.05, 0.92), xycoords='axes fraction', fontsize=14,
         ha="center", va="center",
         bbox=dict(boxstyle="square,pad=0.2",
         fc="silver", ec="k", lw=2))
ax[2].annotate("c",
         xy=(0.05, 0.88), xycoords='axes fraction', fontsize=14,
         ha="center", va="center",
         bbox=dict(boxstyle="square,pad=0.2",
         fc="silver", ec="k", lw=2))
ax[3].annotate("d",
         xy=(0.05, 0.88), xycoords='axes fraction', fontsize=14,
         ha="center", va="center",
         bbox=dict(boxstyle="square,pad=0.2",
         fc="silver", ec="k", lw=2))

fig.subplots_adjust(wspace=0.005, hspace=0.12)

# oetz examples
ortho1 ='/Users/leahartl/Desktop/inventare_2025/Data/oetz_example/example2015.tif'
ortho2 ='/Users/leahartl/Desktop/inventare_2025/Data/oetz_example/example2020.tif'
ortho3 ='/Users/leahartl/Desktop/inventare_2025/Data/oetz_example/example2023.tif'

ax00 = ax[2].inset_axes([-0.01, -1.1, 0.72, 1])
ax01 = ax[2].inset_axes([0.72, -1.1, 0.72, 1])
ax02 = ax[2].inset_axes([1.45, -1.1, 0.72, 1])

for a_ins, o in zip([ax00, ax01, ax02], [ortho1, ortho2, ortho3]):
    with rio.open(o) as src_1:
        s = src_1.read()
        show(s, transform=src_1.transform, ax=a_ins)
        bounds0 = src_1.bounds
        print(src_1.crs)

    a_ins.set_xlim(bounds0[0], bounds0[2])
    a_ins.set_ylim(bounds0[1], bounds0[3])
    a_ins.set_xticks([218200, 218600, 219000])
    a_ins.tick_params(axis='x', labelrotation=90)
    oetzGI3 = oetzGI3.to_crs(src_1.crs)
    oetzGI5 = oetzGI5.to_crs(src_1.crs)
    oetzGI3.boundary.plot(ax=a_ins, color='k', linestyle='--', linewidth=0.5, label='AGI 3')
    oetzGI5.boundary.plot(ax=a_ins, color='k', linestyle='-', linewidth=0.5, label='AGI 5')


ax00.add_artist(ScaleBar(dx=1, location="lower right", font_properties={"size": 12}))

for a in [ax01, ax02]:
    # a.set_xticks([])
    # a.set_yticks([])
    # a.set_xticklabels('')
    a.set_yticklabels('')
# for a in [ax[0], ax[1]]:
#     # a.set_xticks([])
#     # a.set_yticks([])
#     # a.set_xticklabels('')
#     a.set_xticklabels('')

ax00.annotate("e",
         xy=(0.05, 0.88), xycoords='axes fraction', fontsize=14,
         ha="center", va="center",
         bbox=dict(boxstyle="square,pad=0.2",
         fc="silver", ec="k", lw=2))
ax01.annotate("f",
         xy=(0.05, 0.88), xycoords='axes fraction', fontsize=14,
         ha="center", va="center",
         bbox=dict(boxstyle="square,pad=0.2",
         fc="silver", ec="k", lw=2))
ax02.annotate("g",
         xy=(0.05, 0.88), xycoords='axes fraction', fontsize=14,
         ha="center", va="center",
         bbox=dict(boxstyle="square,pad=0.2",
         fc="silver", ec="k", lw=2))
ax00.set_title('N. Schalf Ferner, 2015')
ax01.set_title('N. Schalf Ferner, 2020')
ax02.set_title('N. Schalf Ferner, 2023')


fig.savefig('figures/data_example_GrossElendKees.png', bbox_inches='tight', dpi=200)
plt.show()



    