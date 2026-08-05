import numpy as np
import pandas as pd
import geopandas as gpd
import json
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
import matplotlib.colors as colors

import seaborn as sns
import rasterio
import rioxarray as rxr
import glob
import matplotlib.colors as mcolors
import proc_helper_functions as hlp
import contextily as cx



class MidpointNormalize(mcolors.Normalize):
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        mcolors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        v_ext = np.max( [ np.abs(self.vmin), np.abs(self.vmax) ] )
        x, y = [-v_ext, self.midpoint, v_ext], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))


def load_palette(json_path, palette_name):
    """Loads a color palette from the JSON file by its name."""
    with open(json_path, 'r') as f:
        palettes = json.load(f)
    
    # Filter to find the palette by name
    try:
        data = next(p for p in palettes if p['name'] == palette_name)
        return LinearSegmentedColormap.from_list(data['id'], data['colors'])
    except StopIteration:
        raise ValueError(f"Palette '{palette_name}' not found in {json_path}")




# load AGI5: 
# folder with standardized geojsons, use only non-circles:
folder_new = '/Users/leahartl/Desktop/inventare_2025/mergedfiles/split_vanishing/'
# proc3 additionally contains aspect information)
fls_new = glob.glob(folder_new+'*GI5_proc3.geojson')
GI5 = hlp.getGI(fls_new, -18)
GI5['area'] = GI5.geometry.area
GI5['area_km'] = GI5['area']*1e-6
print(GI5.columns)



# "digitization uncertainty": load outlines, compute stats:
def RR_bs():
    # load files: 
    dir_gl1 = '/Users/leahartl/Desktop/inventare_2025/RR_bernd/Round_Robin/Arvental-Kees-S'
    dir_gl2 = '/Users/leahartl/Desktop/inventare_2025/RR_bernd/Round_Robin/NN'
    dir_gl3 = '/Users/leahartl/Desktop/inventare_2025/RR_bernd/Round_Robin/Wurtenkees'

    RR = gpd.read_file('/Users/leahartl/Desktop/inventare_2025/processing/inventories/out/RR_nocirclrs.geojson', index_col=0)
    RR.index = RR['index']
    df = pd.DataFrame(columns=['Arventalkees', 'NN', 'Wurtenkees'], index=[1, 2, 3])

    for gl, glname in zip([dir_gl1, dir_gl2, dir_gl3], ['Arventalkees', 'NN', 'Wurtenkees']):
        #[6013, 3028, 4038]

        ol1 = gpd.read_file(glob.glob(gl + '/1/*.shp')[0])
        ol2 = gpd.read_file(glob.glob(gl + '/2/*.shp')[0])
        ol3 = gpd.read_file(glob.glob(gl + '/3/*.shp')[0])
        print(ol1.crs)

        df.loc[1, glname] = 1e-6*ol1.area.sum()
        df.loc[2, glname] = 1e-6*ol2.area.sum()
        df.loc[3, glname] = 1e-6*ol3.area.sum()

        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        ol1.boundary.plot(ax=ax, color='r')
        ol2.boundary.plot(ax=ax, color='r')
        ol3.boundary.plot(ax=ax, color='r', label='Digitization experiment')

        if glname=='Arventalkees':
            RR1 = RR.loc[(RR.index==6013) & (RR.iscircle=='no')].to_crs(ol1.crs)
            RR1.boundary.plot(ax=ax, color='k', zorder=0, label='Multi-analyst experiment')
            ax.legend(loc='upper left')
            ax.set_title(glname)
            ax.set_xlabel('meters')
            ax.set_ylabel('meters')
            ax.grid('both')
            fig.savefig('figures/RR_digit_'+glname+'.png', bbox_inches='tight')
            plt.show()
        if glname=='NN':
            RR2 = RR.loc[(RR.index==3028) & (RR.iscircle=='no')].to_crs(ol1.crs)
            RR2.boundary.plot(ax=ax, color='k', zorder=0, label='Multi-analyst experiment')
            ax.legend(loc='upper left')
            ax.set_title(glname)
            ax.set_xlabel('meters')
            ax.set_ylabel('meters')
            ax.grid('both')
            fig.savefig('figures/RR_digit_'+glname+'.png', bbox_inches='tight')
            plt.show()
            # plt.close()
        if glname=='Wurtenkees':
            RR3 = RR.loc[(RR.index==4038) & (RR.iscircle=='no')].to_crs(ol1.crs)
            # RR3 = RR3.loc[RR3.iscircle=='no']
            RR3.boundary.plot(ax=ax, color='k', zorder=0, label='Multi-analyst experiment')
            ax.legend(loc='upper left')
            ax.set_title(glname)
            ax.set_xlabel('meters')
            ax.set_ylabel('meters')
            ax.grid('both')
            fig.savefig('figures/RR_digit_'+glname+'.png', bbox_inches='tight')
            plt.show()

        
        
        # print(RR)

        # bounds = RR.buffer(10).total_bounds
        # ax.set_xlim(bounds[0], bounds[2])
        # ax.set_ylim(bounds[1], bounds[3])



    dfrange = df.max() - df.min()
    dfstdv = df.std()

    df.loc['mean', :] = df.mean()
    df.loc['stdv', :] = dfstdv
    df.loc['uncprc', :] = 100*df.loc['stdv', :] / df.loc['mean', :]
    df.loc['range', :] = dfrange

    print(df)
    plt.show()












def rev_fig_combined(GI5):
    fig = plt.figure(figsize=(12, 8))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.6], width_ratios=[1.4, 1, 1])

    ax1 = fig.add_subplot(gs[0, 0])   # top left
    ax2 = fig.add_subplot(gs[0, 1], projection='polar')   # top right
    ax22 = fig.add_subplot(gs[0, 2], projection='polar')   # top right
    ax3 = fig.add_subplot(gs[1, :])   # bottom spans both columns

    dat = pd.read_csv('/Users/leahartl/Desktop/inventare_2025/processing/inventories/out/merged_GI3GI5.csv')

    # set bins and circle sizes for all panels: 
    bins = [0, 0.01, 0.1, 0.5, 1, 5, np.inf]
    # Marker sizes corresponding to each bin
    marker_sizes = [8, 20, 50, 100, 200, 500]
    # Assign each glacier to a bin
    # bin_idx = np.digitize(GI5.area_km, bins) - 1
    bin_idx = np.digitize(dat.area_GI5*1e-6, bins) - 1
    sizes = np.array(marker_sizes)[bin_idx]

    # discrete intervals
    bounds = np.arange(-8, 1, 1)   # [-90, -80, ..., -10, 0]
    palette = plt.cm.get_cmap("YlOrRd_r", len(bounds)-1).copy()
    palette.set_over('grey')
    norm = colors.BoundaryNorm(bounds, palette.N)

    # panel a: scatter plot mean aspect vs median elevation
    ax1.scatter(dat.circmean_aspect_GI5, dat.median_elev_GI5, s=sizes, alpha=0.7, edgecolor='k', c=dat.loss_rate, cmap=palette, norm=norm)
    ax1.set_xticks(np.arange(0, 360+45, 45))
    ax1.set_xlabel("Aspect [°]")
    ax1.set_ylabel("Median elevation [m a.s.l.]")
    # minor ticks
    ax1.yaxis.set_minor_locator(MultipleLocator(100))

    sm1 = ScalarMappable(norm=norm, cmap=palette)
    sm1.set_array([])

    divider1 = make_axes_locatable(ax1)
    cax1 = divider1.append_axes("top", size="3%", pad=0.1)

    cbar1 = fig.colorbar(sm1, cax=cax1, orientation='horizontal', extend='max')
    cbar1.ax.xaxis.set_label_position('top')
    cbar1.ax.xaxis.tick_top()
    cbar1.set_label("AGI3 to AGI5 area change rate [% yr$^{-1}$]")
    # cax1.set_label_position('top')


    # polar plot area change rate vs aspect 
    labels = np.arange(0, 360, 45)

    # Wrap aspects so North falls in the first bin
    aspect = dat["circmean_aspect_GI3"] % 360
    aspect_shift = (aspect + 22.5) % 360

    edges = np.arange(0, 361, 45)

    dat["aspect_bin"] = pd.cut(aspect_shift, bins=edges, labels=labels, include_lowest=True, right=False)
    #print(dat[["circmean_aspect_GI5", "aspect_bin"]])

    # Mean loss rate per sector
    # percentage /yr
    mean_loss = dat.groupby("aspect_bin")["loss_rate"].mean().reindex(labels)
    # print(mean_loss)
    # stop
    # km loss / yr
    meanKM_loss = dat.groupby("aspect_bin")["KMrate"].mean().reindex(labels)
    # area GI5
    mean_area = dat.groupby("aspect_bin")["area_GI5"].sum().reindex(labels)

    # Angles in radians
    theta0 = np.deg2rad(labels)

    # # Close the curve
    theta0 = np.append(theta0, theta0[0])
    r = np.append(mean_loss.values, mean_loss.values[0])
    r_km = np.append(meanKM_loss.values, meanKM_loss.values[0])
    r_area = np.append(1e-6*mean_area.values, 1e-6*mean_area.values[0])

    print(labels)
    print(r)
    # ax.plot(theta0, r, lw=2)
    # ax.fill(theta0, r, alpha=0.25)


    ax2.plot(theta0, r_area, lw=1, color='k', zorder=1)
    ax2.fill(theta0, r_area, alpha=0.25, color='grey')

    ax22.plot(theta0, r, lw=1, color='k', zorder=1)
    ax22.fill(theta0, r, alpha=0.25, color='grey')

    # ax22.scatter(dat['circmean_aspect_GI5'], dat['loss_rate'], s=2)
    # ax22.set_ylim(-10, 0)
 

    # Compass orientation
    for a in [ax2, ax22]:
        a.set_theta_zero_location("N")
        a.set_theta_direction(-1)

        a.set_xticks(np.deg2rad(labels))
        a.set_xticklabels(["N", "NE", "E", "SE", "S", "SW", "W", "NW"])

        for label in a.get_yticklabels():
            label.set_zorder(1000)
            label.set_path_effects([pe.withStroke(linewidth=3, foreground="white")])

    ax2.set_rlabel_position(180)
    ax22.set_rlabel_position(0)
    #ax22.set_ylim(-3.3, -2.6)
    # ax22.set_yticks(np.arange(-2.6, -3.3, 0.15))

    ax2.set_title("AGI5 glacier area [km$^2$]",  pad=24)#" by 45° aspect sector")
    ax22.set_title("AGI3 to AGI5, mean change rate [% yr$^{-1}$]", pad=24)# by 45° aspect sector")


    ax3.set_xlim(9.5, 14.0)
    ax3.set_ylim(46.5, 47.8)

    GI5 = GI5.to_crs(epsg=4326)
    GI5_large = GI5.loc[GI5.area_km > 0.1]
    GIpts = GI5.copy()
    GIpts['geometry'] = GI5.geometry.centroid
    # GIpts = GIpts.sort_values(by='area_km', ascending=False)

    bin_idx1 = np.digitize(GIpts['area_km'], bins) - 1
    sizes1 = np.array(marker_sizes)[bin_idx1]
    

    countries = gpd.read_file('/Users/leahartl/Desktop/WSS/ne_10m_admin_0_countries/ne_10m_admin_0_countries.shp')
    countries.boundary.plot(ax=ax3, color='k', linewidth=0.5)

    cmapEl = load_palette('palettes.json', 'Drought Index')

    GIpts.plot(ax=ax3, column=GIpts["median_elev"], cmap=cmapEl, markersize=sizes1, alpha=0.9, edgecolor='k')

    # ----- Colorbar -----
    normEl = Normalize(
        vmin=GIpts["median_elev"].min(),
        vmax=GIpts["median_elev"].max()
    )
    sm = ScalarMappable(norm=normEl, cmap=cmapEl)
    sm.set_array([])

    divider = make_axes_locatable(ax3)
    cax = divider.append_axes("right", size="3%", pad=-0.9)
 
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_label("Median elevation [m a.s.l.]")

    # ----- Size legend -----
    labels = ["<=0.01",">0.01–0.1",">0.1–0.5",">0.5–1",">1–5",">5"]

    # marker_sizes = [20, 40, 70, 100, 150, 220]

    handles = [
        Line2D(
            [], [], linestyle='',
            marker='o',
            markersize=np.sqrt(s),   # scatter s is area; legend uses diameter
            markerfacecolor='lightgray',
            markeredgecolor='k',
            label=lab
            )
        for s, lab in zip(marker_sizes, labels)
        ]

    ax3.legend(handles=handles, title="Glacier size [km²]", loc="lower left",frameon=True, bbox_to_anchor=(-0.32, 0.2),
        labelspacing=1.2,     # increase vertical distance between labels
        handleheight=2.0)      # give large markers more room)
    ax3.set_xlabel('Longitude [°]')
    ax3.set_ylabel('Latitude [°]')

    ax1.annotate(
            'a',
            xy=(0.05, 1), xycoords='axes fraction',
            xytext=(-0.8, -0.5), textcoords='offset fontsize',
            fontsize='medium', verticalalignment='top', #fontfamily='serif',
            bbox=dict(facecolor='lightgrey', edgecolor='k', pad=3.0))
    ax2.annotate(
            'b',
            xy=(-0.05, 1), xycoords='axes fraction',
            xytext=(-0.8, -0.5), textcoords='offset fontsize',
            fontsize='medium', verticalalignment='top', #fontfamily='serif',
            bbox=dict(facecolor='lightgrey', edgecolor='k', pad=3.0))
    ax22.annotate(
            'c',
            xy=(-0.05, 1), xycoords='axes fraction',
            xytext=(-0.8, -0.5), textcoords='offset fontsize',
            fontsize='medium', verticalalignment='top', #fontfamily='serif',
            bbox=dict(facecolor='lightgrey', edgecolor='k', pad=3.0))
    ax3.annotate(
            'd',
            xy=(0.04, 1), xycoords='axes fraction',
            xytext=(-0.8, -0.5), textcoords='offset fontsize',
            fontsize='medium', verticalalignment='top', #fontfamily='serif',
            bbox=dict(facecolor='lightgrey', edgecolor='k', pad=3.0))


    fig.savefig('figures/extra_fig.png', bbox_inches='tight', dpi=200)



RR_bs()

# rev_fig_combined(GI5)


# plt.show()



 