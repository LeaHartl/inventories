import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.patches as mpatches
import seaborn as sns
import rasterio
import rioxarray as rxr
import glob
import matplotlib.colors as mcolors


class MidpointNormalize(mcolors.Normalize):
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        mcolors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        v_ext = np.max( [ np.abs(self.vmin), np.abs(self.vmax) ] )
        x, y = [-v_ext, self.midpoint, v_ext], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))




def figsfromcsv(outfolder):
    # load data for plot:
    df_area_elevation = pd.read_csv(outfolder+'df_area_elevation.csv', index_col=0)
    summtab = pd.read_csv(outfolder+'summary_area_changesGI3GI5.csv', index_col=0)
    dfReg = pd.read_csv(outfolder+'regions_medianElevation.csv', index_col=0)
    dfReg['prc'] = 100*dfReg['Areakm']/dfReg['Areakm'].sum()

    fig = plt.figure(figsize=(12, 7))
    gs = GridSpec(2, 4, wspace=0.3)# left=0.05, right=0.48, wspace=0.05)
    ax0 = fig.add_subplot(gs[:, 0])

    ax0.step(10*10*df_area_elevation['GI3']*1e-6, df_area_elevation.index, label='AGI3', color='grey')
    ax0.step(10*10*df_area_elevation['GI5']*1e-6, df_area_elevation.index, label='AGI5', color='k')

    ax0.set_ylabel('Elevation [m.a.s.l.]', fontsize=12)
    ax0.set_xlabel('Glacier area [km$^2$]', fontsize=12)
    ax0.legend(loc='lower right')
    ax0.set_ylim(1800, 3750)
    ax0.grid('both')

    ax0lost = ax0.twiny()
    # percentage of area per bin (multiply by cell size (10*10) to get area)
    ax0lost.step(100*df_area_elevation['lostAR']*10*10 / (10*10*df_area_elevation['lostAR']).sum(), df_area_elevation.index, label='Area loss (%)', color='red', linestyle='--')
    ax0lost.set_xlabel('Area loss [% of total loss]', fontsize=12, color='red')
    ax0lost.set_xlim(0, 12)
    ax0lost.tick_params(axis='x', labelcolor='red')
    ax0lost.legend(loc='upper right')





    summtab['ar'] = summtab['arkm_str'].str.split('±').str[0].astype(float)
    names = [0.2, 0.5, 0.7, 0.9, 1.1, 1.3, 2.8]
    bns = [0, 1, 5, 15, 30, 45, 60, 200]
    summtab['sizeBin'] = pd.cut(summtab['ar'], bins=bns, labels=names, include_lowest=True)

    print(summtab)
    mrg = pd.merge(summtab[['ar', 'perc_of_total', 'loss_rate', 'perc_loss', 'sizeBin']], dfReg[['medianElev', 'Lon']], left_index=True, right_index=True)
    

    print(mrg)


    m = {'Oetztaler_Alpen': 'Ötztal Alps', 'Venedigergruppe': 'Venediger Group', 'Zillertaler_Alpen': 'Zillertal Alps',
         'Stubaier_Alpen': 'Stubai Alps', 'Glocknergruppe': 'Glockner Group', 'Ankogel_Hochalmspitzgruppe': 'Ankogel Group', 
         'Silvrettagruppe': 'Silvretta', 'Sonnblickgruppe': 'Sonnblick', 'Granatspitzgruppe': 'Granatspitz Group',
          'Salzburger_Kalkalpen': 'Salzburg Limestone Alps', 'Raetikon': 'Rätikon', 'Defreggergruppe': 'Defregger Group',
          'Lechtaler_Alpen': 'Lechtaler Alps', 'Karnische_Alpen': 'Carnic Alps', 'Allgaeuer_Alpen': 'Allgäu Alps',
          'Schobergruppe': 'Schober Group', 'Samnaungruppe': 'Samnaun Group', 'Silvretta': 'Silvretta Group', 
          'Verwallgruppe': 'Verwall Group'}

    for val, key in m.items():
        mrg.index = mrg.index.str.replace(val, key, regex=True)

    
    mrg = mrg.sort_values(by='Lon')
    mrg['offset_z'] = [40, 40, 40, -390, -300, 40, 80, 40, 40, 40, -480, 40, 40, 40, -390, -300, 40, -640, 40, 40]
    mrg['offset_x'] = [0, 0, 0.01, -0.05, 0.11, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    print(mrg)
    max_allowed=0
    min_allowed=-90
    norm = plt.Normalize(min_allowed, max_allowed)#MidpointNormalize( midpoint =0, vmin=-90, vmax=30)
    palette = plt.cm.get_cmap("Reds_r").copy()
    palette.set_over('grey', 1)
    cmap = 'Reds_r'
    # f, a = plt.subplots(1, 1, figsize=(10, 4))

    a = fig.add_subplot(gs[:, 1:])
    m = a.scatter(mrg.Lon, mrg.medianElev, s=200*mrg['sizeBin'].astype(float), c=mrg['perc_loss'], edgecolor='darkgrey', cmap=palette, vmin=min_allowed, vmax=max_allowed)#cmap=cmap, norm=norm, )
    cb = fig.colorbar(m, extend='max')
    cb.set_label('Area change [% of AGI3 area]', fontsize=12)
    # a.set_ylim(2200, 3200)
    a.set_ylim(1800, 3750)
    # a.set_ylabel('median elevation [m a.s.l.]', fontsize=12)
    a.set_xlabel('Longitude [°]', fontsize=12)
    a.grid('both')

    ptch = []
    lbl = ['<1 km$^2$', '1-5 km$^2$', '5-15 km$^2$', '15-30 km$^2$', '30-45 km$^2$', '45-60 km$^2$', '>60 km$^2$']
    # for i, n in enumerate(names): 
    #     p = Line2D([0], [0], marker='o', linestyle='None', label=sizes[i], markeredgecolor='darkgrey', color='white', markersize=n**2, zorder=20)
    #     ptch.append(p)
    
    # msizes = [3, 4, 5, 6, 7]
    markers = []
    for i, size in enumerate(names): 
        markers.append(plt.scatter([],[], s=size*200, label=lbl[i], edgecolor='darkgrey', c='white'))

    # extra = Line2D([],[],linestyle='')
    # print(markers)
    # markers1 = markers[:]
    # markers1 = markers1.insert(-2, extra)

    for x, z, offset, offset_x, txt in zip(mrg['Lon'].values, mrg['medianElev'].values, mrg['offset_z'].values,mrg['offset_x'].values, mrg.index.values):
        a.text(x+offset_x, z+offset, txt,
               rotation='vertical',
               verticalalignment='bottom',
               horizontalalignment='center')
                      # transform=a.transAxes)


    plt.legend(handles=markers, loc='lower right', bbox_to_anchor=(1.0, -0.2), ncol=4)

    ax0.annotate(
            'a',
            xy=(-0.05, 0.98), xycoords='axes fraction',
            xytext=(-1, -0.5), textcoords='offset fontsize',
            fontsize='medium', verticalalignment='top', #fontfamily='serif',
            bbox=dict(facecolor='lightgrey', edgecolor='k', pad=3.0))


    a.annotate(
            'b',
            xy=(0.05, 0.98), xycoords='axes fraction',
            xytext=(-1, -0.5), textcoords='offset fontsize',
            fontsize='medium', verticalalignment='top', #fontfamily='serif',
            bbox=dict(facecolor='lightgrey', edgecolor='k', pad=3.0))
    

    fig.savefig('figures/scatter_regions.png', bbox_inches='tight', dpi=200)
 

# def loss_stacked(GI1GI2, GI2GI3, all_mrg, df_prc, df_abs):

#     # load in situ table:
#     ins = pd.read_csv('/Users/leahartl/Desktop/inventare_2025/processing/out/compare_debris.csv')
#     print(ins)
#     print(all_mrg)

#     subs = all_mrg.loc[all_mrg['id'].isin(ins['id'].values)]
#     print(subs)

#     # loss rates - four panels: stacked area since LIA, various versions of glacierwise loss 
#     # Define custom colors 
#     custom_colors = ["k", "grey", "#003f5c","#2f4b7c","#665191","#a05195","#d45087","#f95d6a","#ff7c43","#ffa600", "silver", "darkgrey"]
#     print(df_prc)
#     df_abs = df_abs.sort_values(by=1850, ascending=False)
#     df_abs = df_abs[[1850, 1969, 1998, 2006, 2023]]
#     print(df_abs)

#     df_abs_r = df_abs[[1969, 1998, 2006, 2023]]
#     df_abs_r['r1969'] = (df_abs[1969]-df_abs[1850]) / (1969-1850)
#     df_abs_r['r1998'] = (df_abs[1998]-df_abs[1969]) / (1998-1969)
#     df_abs_r['r2006'] = (df_abs[2006]-df_abs[1998]) / (2006-1998)
#     df_abs_r['r2023'] = (df_abs[2023]-df_abs[2006]) / (2023-2006)

#     print(df_abs_r)

#     # stacked area chart:
#     # fig, ax = plt.subplots(1, 1, figsize=(12, 7))  # Set the figure size
#     fig = plt.figure(figsize=(12, 10), layout="constrained")
#     gs = GridSpec(3, 3, figure=fig)
#     ax = fig.add_subplot(gs[0:2, :])


#     m = {'Oetztaler_Alpen': 'Ötztal Alps', 'Venedigergruppe': 'Venediger Group', 'Zillertaler_Alpen': 'Zillertal Alps',
#          'Stubai_Alps': 'Stubai Alps', 'Glocknergruppe': 'Glockner Group', 'Ankogel_Hochalmspitzgruppe': 'Ankogel Group', 
#          'Silvrettagruppe': 'Silvretta', 'Sonnblickgruppe': 'Sonnblick', 'Granatspitzgruppe': 'Granatspitzg Group',
#           'Salzburger_Kalkalpen': 'Salzburg Limestone Alps', 'Raetikon': 'Rätikon'}

#     df_prc.index = ['Ötztal Alps','Venediger Group','Zillertal Alps','Stubai Alps', 'Glockner Group', 
#             'Ankogel Group', 'Silvretta', 'Sonnblick','Granatspitz Group', 'Dachstein',
#             'Salzburg Limestone Alps', 'Rätikon']
#     df_abs.index = ['Ötztal Alps','Venediger Group','Zillertal Alps','Stubai Alps', 'Glockner Group', 
#             'Ankogel Group', 'Silvretta', 'Sonnblick','Granatspitz Group', 'Dachstein',
#             'Salzburg Limestone Alps', 'Rätikon']
#     #df_prc.index.replace(pd.Series(m).astype(str), regex=True)

#     # ax=ax.flatten()
#     # ax.stackplot([1850, 1969, 1998, 2006, 2023], df_prc.values, labels=df_prc.index, colors=custom_colors)
#     ax.stackplot([1850, 1969, 1998, 2006, 2023], df_abs.values, labels=df_abs.index, colors=custom_colors, alpha=0.7)
#     ax.grid('both')
#     ax.set_ylim(0, 950)
#     ax.set_xlim(1849, 2025)
#     # ax.set_xlim(1969, 2025)
#     ax.tick_params(axis='both', which='major', labelsize=12)
#     ax.set_xlabel('Year', fontsize=12)
#     # ax.set_ylabel('Percentage of 1850 area [%]', fontsize=12)
#     ax.set_ylabel('Glacier area [$km^2$]', fontsize=12)
#     ax.legend(ncol=2, loc='lower left', bbox_to_anchor=(0.01, 0.01), fontsize=12)
#     ax.vlines([1850, 1969, 1998, 2006, 2023], 0, 1000, colors='silver')

#     ax.annotate('AGILIA',
#             xy=(1854, 800), xycoords='data',
#             xytext=(-0.8, -0.5), textcoords='offset fontsize',
#             fontsize='medium', verticalalignment='top', #fontfamily='serif',
#             bbox=dict(facecolor='white', edgecolor='k', pad=3.0))
#     ax.annotate('AGI1',
#             xy=(1967, 800), xycoords='data',
#             xytext=(-0.8, -0.5), textcoords='offset fontsize',
#             fontsize='medium', verticalalignment='top', #fontfamily='serif',
#             bbox=dict(facecolor='white', edgecolor='k', pad=3.0))
#     ax.annotate('AGI2',
#             xy=(1996, 800), xycoords='data',
#             xytext=(-0.8, -0.5), textcoords='offset fontsize',
#             fontsize='medium', verticalalignment='top', #fontfamily='serif',
#             bbox=dict(facecolor='white', edgecolor='k', pad=3.0))
#     ax.annotate('AGI3',
#             xy=(2004, 800), xycoords='data',
#             xytext=(-0.8, -0.5), textcoords='offset fontsize',
#             fontsize='medium', verticalalignment='top', #fontfamily='serif',
#             bbox=dict(facecolor='white', edgecolor='k', pad=3.0))
#     ax.annotate('AGI5',
#             xy=(2021, 800), xycoords='data',
#             xytext=(-0.8, -0.5), textcoords='offset fontsize',
#             fontsize='medium', verticalalignment='top', #fontfamily='serif',
#             bbox=dict(facecolor='white', edgecolor='k', pad=3.0))

    
   
#     # ax1.scatter((all_mrg['area_GI5']-all_mrg['area_GI3'])*1e-6/(all_mrg['year']-all_mrg['Year']), all_mrg['area_GI3']*1e-6, c='k', s=10, edgecolor='grey', linewidth=0.4, label='remote sensing data only')
#     # ax1.scatter((subs['area_GI5']-subs['area_GI3'])*1e-6/(subs['year']-subs['Year']), subs['area_GI3']*1e-6, c='red', s=10, edgecolor='grey', linewidth=0.4, label='inclusion of in situ surveys')

#     # ax1.set_ylabel('Area GI3 [$km^2$]', fontsize=12)
#     # ax1.set_xlabel('Change rate GI3 - GI5 [$km^2 a^{-1}$]', fontsize=12)

#     # ax2 = fig.add_subplot(gs[1, 1])
#     # ax2.scatter(all_mrg['loss_rate'], all_mrg['area_GI3']*1e-6, c='k', s=10, edgecolor='grey', linewidth=0.4)
#     # # ax2.semilogy(all_mrg['loss_rate'], all_mrg['area_GI3']*1e-6, c='k', marker='o', markersize=6, markeredgecolor='grey', linestyle=None)
#     # ax2.scatter(subs['loss_rate'], subs['area_GI3']*1e-6, c='red', s=10, edgecolor='grey', linewidth=0.4)
#     # ax2.set_ylabel('Area GI3 [$km^2$]', fontsize=12)
#     # ax2.set_xlabel('Change rate GI3 - GI5 [% $a^{-1}$]', fontsize=12)

#     ax1 = fig.add_subplot(gs[-1, 0])
#     ax2 = fig.add_subplot(gs[-1, 1])
#     ax3 = fig.add_subplot(gs[-1, 2])
    
#     ax1.hist(GI1GI2['rate'].values, bins=np.arange(-10, 14, 1), histtype='stepfilled', color='lightgrey', alpha=0.8)
#     ax1.hist(GI1GI2['rate'].values, bins=np.arange(-10, 14, 1), histtype='step', color='k', label='GI1-GI2, n='+str(len(GI1GI2['rate'].values)))

#     ax2.hist(GI2GI3['rate'].values, bins=np.arange(-10, 14, 1), histtype='stepfilled', color='lightgrey', alpha=0.8)
#     ax2.hist(GI2GI3['rate'].values, bins=np.arange(-10, 14, 1), histtype='step', color='k', label='GI2-GI3, n='+str(len(GI2GI3['rate'].values)))

#     toplot = all_mrg.loc[~all_mrg['loss_rate'].isnull()]
#     ax3.hist(toplot['loss_rate'].values, bins=np.arange(-10, 14, 1), histtype='stepfilled', color='lightgrey', alpha=0.8)
#     ax3.hist(toplot['loss_rate'].values, bins=np.arange(-10, 14, 1), histtype='step', color='k', label='GI3-GI5, n='+str(len(toplot['id'].unique())))
    
#     ax1.set_ylabel('Nr. of glaciers', fontsize=12)



#     subset = all_mrg.loc[~all_mrg['r1'].isnull()]

#     # ax3.hist(subset['r2'].values, bins=np.arange(-20, 20, 1), histtype='stepfilled', color='magenta', alpha=0.6,label='2017/18-GI5, n='+str(len(subset['r2'].values)))
#     # ax3.hist(subset['r2'].values, bins=np.arange(-20, 20, 1), histtype='step', color='magenta')


#     # ax3.legend(loc='upper right', bbox_to_anchor=(0.8, -0.2))
#     ax1.legend()
#     ax2.legend()
#     ax3.legend()

#     for a in [ax1, ax2, ax3]:
#         a.set_ylim(0, 420)
#         a.set_xlim(-11, 12)
#         a.set_xlabel('Change rate [% $a^{-1}$]', fontsize=12)

#     print(all_mrg.columns)
#     for a, an in zip([ax, ax1, ax2, ax3], ['a', 'b', 'c', 'd']):
#         a.grid('both')
#         a.tick_params(axis='both', which='major', labelsize=12)
#         if an =='a':
#             a.annotate(an,
#             xy=(0.96, 1), xycoords='axes fraction',
#             xytext=(-0.8, -0.5), textcoords='offset fontsize',
#             fontsize=12, verticalalignment='top', #fontfamily='serif',
#             bbox=dict(facecolor='lightgrey', edgecolor='k', pad=3.0))
#         else:
#             a.annotate(an,
#             xy=(0.98, 1), xycoords='axes fraction',
#             xytext=(-0.8, -0.5), textcoords='offset fontsize',
#             fontsize=12, verticalalignment='top', #fontfamily='serif',
#             bbox=dict(facecolor='lightgrey', edgecolor='k', pad=3.0))

#     ax1.legend(loc='upper right', bbox_to_anchor=(0.95, 0.8))

#     pos = all_mrg.loc[all_mrg['loss_rate'] > 0]
#     pos_ins = pos.loc[pos['id'].isin(ins['id'].values)]

#     print(pos_ins)
#     print(GI2GI3.head())
#     print(all_mrg.head())
#     check = all_mrg['id'].loc[~all_mrg['id'].isin(GI2GI3['id'].values)]
#     print(check)
#     # print(all_mrg.loc[all_mrg['id']==5035.0])

#     # plt.show()
#     fig.savefig('figures/loss_stacked_1850_panels.png', dpi=200, bbox_inches='tight')


def loss_stacked_BARS(GI1GI2, GI2GI3, all_mrg, df_prc, df_abs):

    # loss rates - four panels: stacked area since LIA, various versions of glacierwise loss 
    # Define custom colors 
    custom_colors = ["k", "grey", "#003f5c","#2f4b7c","#665191","#a05195","#d45087","#f95d6a","#ff7c43","#ffa600", "silver", "darkgrey"]

    df_abs = df_abs.sort_values(by=1850, ascending=False)
    df_abs = df_abs[[1850, 1969, 1998, 2006, 2023]]

    # stacked area/bar chart:
    fig = plt.figure(figsize=(12, 10), layout="constrained")
    gs = GridSpec(3, 3, figure=fig)
    ax = fig.add_subplot(gs[0:2, :])


    m = {'Oetztaler_Alpen': 'Ötztal Alps', 'Venedigergruppe': 'Venediger Group', 'Zillertaler_Alpen': 'Zillertal Alps',
         'Stubai_Alps': 'Stubai Alps', 'Glocknergruppe': 'Glockner Group', 'Ankogel_Hochalmspitzgruppe': 'Ankogel Group', 
         'Silvrettagruppe': 'Silvretta', 'Sonnblickgruppe': 'Sonnblick', 'Granatspitzgruppe': 'Granatspitzg Group',
          'Salzburger_Kalkalpen': 'Salzburg Limestone Alps', 'Raetikon': 'Rätikon'}

    df_prc.index = ['Ötztal Alps','Venediger Group','Zillertal Alps','Stubai Alps', 'Glockner Group', 
            'Ankogel Group', 'Silvretta', 'Sonnblick','Granatspitz Group', 'Dachstein',
            'Salzburg Limestone Alps', 'Rätikon']
    df_abs.index = ['Ötztal Alps','Venediger Group','Zillertal Alps','Stubai Alps', 'Glockner Group', 
            'Ankogel Group', 'Silvretta Group', 'Sonnblick Group','Granatspitz Group', 'Dachstein',
            'Salzburg Limestone Alps', 'Rätikon']

    # set with as 7 for AGI LIA and AGI1, AGI2: 2002-1996; AGI3: 2012-2004
    # widths = [5, 5, 6, 8, 3]
    widths = [5, 5, 5, 5, 5]
    bottom = [0, 0, 0, 0, 0]
    for i, reg in enumerate(df_abs.index):
        values = df_abs.loc[reg, :].values
        p = ax.bar([1850, 1969, 1998, 2006, 2023], values, width = widths, label=reg, bottom=bottom, color=custom_colors[i])
        bottom += values

    ax.grid('both')
    ax.set_ylim(0, 950)
    ax.set_xlim(1845, 2029)
    # ax.set_xlim(1969, 2025)
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.set_xlabel('Year', fontsize=12)
    # ax.set_ylabel('Percentage of 1850 area [%]', fontsize=12)
    ax.set_ylabel('Glacier area [km$^2$]', fontsize=12)
    ax.legend(ncol=2, loc='lower left', bbox_to_anchor=(0.18, 0.01), fontsize=12)
    # ax.vlines([1850, 1969, 1998, 2006, 2023], 0, 1000, colors='silver')

    ax.annotate('AGI LIA',
            xy=(1858, 620), xycoords='data',
            xytext=(-0.8, -0.5), textcoords='offset fontsize',
            fontsize='medium', verticalalignment='top', #fontfamily='serif',
            bbox=dict(facecolor='white', edgecolor='k', pad=3.0))
    ax.annotate('AGI1',
            xy=(1967, 620), xycoords='data',
            xytext=(-0.8, -0.5), textcoords='offset fontsize',
            fontsize='medium', verticalalignment='top', #fontfamily='serif',
            bbox=dict(facecolor='white', edgecolor='k', pad=3.0))
    ax.annotate('AGI2',
            xy=(1996, 620), xycoords='data',
            xytext=(-0.8, -0.5), textcoords='offset fontsize',
            fontsize='medium', verticalalignment='top', #fontfamily='serif',
            bbox=dict(facecolor='white', edgecolor='k', pad=3.0))
    ax.annotate('AGI3',
            xy=(2004, 620), xycoords='data',
            xytext=(-0.8, -0.5), textcoords='offset fontsize',
            fontsize='medium', verticalalignment='top', #fontfamily='serif',
            bbox=dict(facecolor='white', edgecolor='k', pad=3.0))
    ax.annotate('AGI5',
            xy=(2021, 620), xycoords='data',
            xytext=(-0.8, -0.5), textcoords='offset fontsize',
            fontsize='medium', verticalalignment='top', #fontfamily='serif',
            bbox=dict(facecolor='white', edgecolor='k', pad=3.0))

    
    ax1 = fig.add_subplot(gs[-1, 0])
    ax2 = fig.add_subplot(gs[-1, 1])
    ax3 = fig.add_subplot(gs[-1, 2])
    
    ax1.hist(GI1GI2['rate'].values, bins=np.arange(-10, 14, 1), histtype='stepfilled', color='lightgrey', alpha=0.8)
    ax1.hist(GI1GI2['rate'].values, bins=np.arange(-10, 14, 1), histtype='step', color='k', label='GI1-GI2, n='+str(len(GI1GI2['rate'].values)))

    ax2.hist(GI2GI3['rate'].values, bins=np.arange(-10, 14, 1), histtype='stepfilled', color='lightgrey', alpha=0.8)
    ax2.hist(GI2GI3['rate'].values, bins=np.arange(-10, 14, 1), histtype='step', color='k', label='GI2-GI3, n='+str(len(GI2GI3['rate'].values)))

    toplot = all_mrg.loc[~all_mrg['loss_rate'].isnull()]
    ax3.hist(toplot['loss_rate'].values, bins=np.arange(-10, 14, 1), histtype='stepfilled', color='lightgrey', alpha=0.8)
    ax3.hist(toplot['loss_rate'].values, bins=np.arange(-10, 14, 1), histtype='step', color='k', label='GI3-GI5, n='+str(len(toplot['id'].unique())))
    
    ax1.set_ylabel('Nr. of glaciers', fontsize=12)

    subset = all_mrg.loc[~all_mrg['r1'].isnull()]

    ax1.legend()
    ax2.legend()
    ax3.legend()

    for a in [ax1, ax2, ax3]:
        a.set_ylim(0, 450)
        a.set_xlim(-11, 12)
        a.set_xlabel('Change rate [% $a^{-1}$]', fontsize=12)

    for a, an in zip([ax, ax1, ax2, ax3], ['a', 'b', 'c', 'd']):
        a.grid('both')
        a.tick_params(axis='both', which='major', labelsize=12)
        if an =='a':
            a.annotate(an,
            xy=(0.96, 1), xycoords='axes fraction',
            xytext=(-0.8, -0.5), textcoords='offset fontsize',
            fontsize=12, verticalalignment='top', #fontfamily='serif',
            bbox=dict(facecolor='lightgrey', edgecolor='k', pad=3.0))
        else:
            a.annotate(an,
            xy=(0.98, 1), xycoords='axes fraction',
            xytext=(-0.8, -0.5), textcoords='offset fontsize',
            fontsize=12, verticalalignment='top', #fontfamily='serif',
            bbox=dict(facecolor='lightgrey', edgecolor='k', pad=3.0))

    ax1.legend(loc='upper right', bbox_to_anchor=(0.95, 0.8))


    # add if needed: 
    # load in situ table:
    # ins = pd.read_csv('/Users/leahartl/Desktop/inventare_2025/processing/out/compare_debris.csv')
    # subs = all_mrg.loc[all_mrg['id'].isin(ins['id'].values)]
    # pos = all_mrg.loc[all_mrg['loss_rate'] > 0]
    # pos_ins = pos.loc[pos['id'].isin(ins['id'].values)]

    # print(pos_ins)
    # print(GI2GI3.head())
    # print(all_mrg.head())
    # check = all_mrg['id'].loc[~all_mrg['id'].isin(GI2GI3['id'].values)]
    # print(check)
 
    fig.savefig('figures/loss_stacked_1850_panelsBAR.png', dpi=200, bbox_inches='tight')


def rates_glacierwise1(GI1GI2, GI2GI3, all_mrg, GI3, GI5, goneglaciers):

    fig = plt.figure(figsize=(11, 12))#, layout='constrained')
    gs = GridSpec(4, 4, hspace=0.3)#wspace=0.3)# left=0.05, right=0.48, wspace=0.05)
    # ax0 = fig.add_subplot(gs[:, 0])
    ax2 = fig.add_subplot(gs[0, 0:2])
    ax3 = fig.add_subplot(gs[0, 2:3])
    ax1 = fig.add_subplot(gs[0, 3:])

    ax5 = fig.add_subplot(gs[1, :])

    ax4 = fig.add_subplot(gs[2:, :])


    parts2 = ax2.violinplot([(goneglaciers['area_GI3']*1e-6).values, (GI3['area']*1e-6).values, (GI5['area']*1e-6).values],
                            orientation='horizontal',
                            widths=0.9,
                            showmedians=True,)
    parts3 = ax3.violinplot([(goneglaciers['area_GI3']*1e-6).values, (GI3['area']*1e-6).values, (GI5['area']*1e-6).values],
                            orientation='horizontal',
                            widths=0.9,
                            showmedians=True,)
    ax2.set_xlim(-0.01, 1.)
    ax3.set_xlim(10, 18)

    colors2 = ['tomato', 'grey', 'k']

    # Set the color of the violin patches
    for pc, color in zip(parts2['bodies'], colors2):
        pc.set_facecolor(color)
    parts2['cmedians'].set_colors(colors2)
    parts2['cmins'].set_colors(colors2)
    parts2['cmaxes'].set_colors(colors2)
    parts2['cbars'].set_colors(colors2)
    # Set the color of the violin patches
    for pc, color in zip(parts3['bodies'], colors2):
        pc.set_facecolor(color)
    parts3['cmedians'].set_colors(colors2)
    parts3['cmins'].set_colors(colors2)
    parts3['cmaxes'].set_colors(colors2)
    parts3['cbars'].set_colors(colors2)

    # hide the spines between ax and ax2
    ax2.spines.right.set_visible(False)
    ax3.spines.left.set_visible(False)
    ax2.xaxis.tick_bottom()
    # ax2.tick_params(labeltop=False)  # don't put tick labels at the top
    # ax3.xaxis.tick_left()

    d = .5  # proportion of vertical to horizontal extent of the slanted line
    kwargs = dict(marker=[(-1, -d), (1, d)], markersize=12,
              linestyle="none", color='k', mec='k', mew=1, clip_on=False)
    
    # ax2.plot([1, 1], [1, 1], transform=ax2.transAxes, **kwargs)
    ax2.plot([1, 1], [0, 1], transform=ax2.transAxes, **kwargs)

    ax3.plot([0, 0], [1, 1], transform=ax3.transAxes, **kwargs)
    ax3.plot([0, 0], [0, 0], transform=ax3.transAxes, **kwargs)

    ax2.set_yticks([y + 1 for y in range(len(colors2))]) 
    ax3.set_yticks([y + 1 for y in range(len(colors2))])          
    ax2.set_yticklabels(['van.', 'AGI3', 'AGI5'],
                        fontsize=12)

    ax3.set_yticklabels('')
    ax2.tick_params(axis='both', which='major', labelsize=12)
    ax3.tick_params(axis='both', which='major', labelsize=12)
    ax2.set_xlabel('Glacier area [km$^2$]', fontsize=12)

    #ax2.xaxis.set_label_position("top")
    #ax2.xaxis.tick_top()
    #ax3.xaxis.tick_top()

    parts = ax1.violinplot([goneglaciers['median_elev'].values, GI3['median_elev'].values, GI5['median_elev'].values],
                            # showmeans=False,
                            widths=0.7,
                            showmedians=True,)
                        # tick_labels=['GI3, min. elev.', 'GI5, min. elev.',
                        #    'GI3, median. elev.', 'GI5, median. elev.']


    # Set the color of the violin patches
    for pc, color in zip(parts['bodies'], colors2):
        pc.set_facecolor(color)
    parts['cmedians'].set_colors(colors2)
    parts['cmins'].set_colors(colors2)
    parts['cmaxes'].set_colors(colors2)
    parts['cbars'].set_colors(colors2)
    ax1.yaxis.set_label_position("right")
    ax1.yaxis.tick_right()
    ax1.set_xticks([1, 2, 3])
    ax1.set_xticklabels(['van.', 'AGI3', 'AGI5'])
    ax1.set_ylabel('Median glacier\n elevation [m]', fontsize=12)
    #ax1.xaxis.tick_top()
    #ax1.xaxis.set_label_position("top")
    ax1.tick_params(axis='both', which='major', labelsize=12)


    notgone = all_mrg.loc[all_mrg['area_GI5'] > 0]
    gone = all_mrg.loc[all_mrg['area_GI5'] == 0]
    

    arbins = [0, 0.01, 0.1, 0.5, 1, 5, 18]
    all_mrg['binned'] = pd.cut(all_mrg['area_GI3']*1e-6, arbins)
    notgone['binned'] = pd.cut(notgone['area_GI3']*1e-6, arbins)
    gone['binned'] = pd.cut(gone['area_GI3']*1e-6, arbins)


    gb_all = all_mrg[['loss_rate', 'binned']].groupby(['binned']).median()
    gb_all_count = all_mrg[['loss_rate', 'binned']].groupby(['binned']).count()

    gbnotgone = notgone[['loss_rate', 'binned']].groupby(['binned']).median()
    # gbnotgone = notgone[['loss_rate', 'binned']].groupby(['binned']).median()
    gonegb = gone[['loss_rate', 'binned', 'area_GI3', 'median_elev_GI3']].groupby(['binned']).count()
    print(gb_all)
    print(gb_all_count)
    print(gonegb)

    ax5.bar([0.9, 1.8, 3, 4, 5, 6], gb_all['loss_rate'].values, color='darkgrey', width=0.4)#gb_all_count['loss_rate']/gb_all_count['loss_rate'].sum())
    
    ax55 = ax5.twinx()
    ax55.bar([1.1, 2.2, 3.1, 4, 5, 6], gonegb['loss_rate'].values, color='tomato', width=0.2)
    
    ax5.set_xticks([1, 2, 3, 4, 5, 6])
    ax5.set_xticklabels(['<0.01', '0.01 - 0.1', '0.1-0.5', '0.5-1.0', '1.0-5.0', '>5'])

    ax5.tick_params(axis='both', which='major', labelsize=12)
    ax55.tick_params(axis='both', which='major', labelsize=12)
    ax5.set_xlabel('Glacier area [km$^2$]', fontsize=12)
    ax5.set_ylabel('Median area\n change rate[% a$^{-1}$]', fontsize=12, color='darkgrey')

    ax55.set_ylabel('Vanishing\n glaciers [#]', fontsize=12, color='tomato')
    ax55.tick_params(axis='y', colors='tomato')

    # norm = MidpointNormalize( midpoint =0, vmin=-7, vmax=3)
    # cmap = 'Reds_r'
    max_allowed = 0
    min_allowed = -8
    norm = plt.Normalize(min_allowed, max_allowed)#MidpointNormalize( midpoint =0, vmin=-90, vmax=30)
    palette = plt.cm.get_cmap("Reds_r").copy()
    palette.set_over('grey', 1)
    cmap = 'Reds_r'
    # ax4.semilogx(all_mrg['area_GI3']*1e-6, all_mrg['median_elev_GI3'], c=all_mrg['loss_rate'].values, s=20, markeredgecolor='grey', linestyle='', cmap=cmap, norm=norm)

    sc = ax4.scatter(notgone['area_GI5']*1e-6, notgone['median_elev_GI3'], c=notgone['loss_rate'].values, s=40, edgecolor='grey', linewidth=0.4, cmap=palette, norm=norm)
    
    ax4.scatter(gone['area_GI3']*1e-6, gone['median_elev_GI3'], c='k', s=140, edgecolor='lightgrey', linewidth=0.6, zorder=200, marker='*', label='vanishing glaciers')
    ax4.set_xscale('log')

    # fig.subplots_adjust(right=0.8)
    cbar_ax = inset_axes(ax4,
                    width="2%", 
                    height="90%",
                    loc='center right',
                    borderpad=-5)
    cb = fig.colorbar(sc, extend='max', pad=0.025, aspect=20, cax=cbar_ax)
    cb.set_label('Area change rate [% a$^{-1}$]', fontsize=12)
    ax4.grid()
    ax4.legend(loc='lower right')

    ax4.tick_params(axis='both', which='major', labelsize=12)
    ax4.set_xlabel('Glacier area [km$^2$]', fontsize=12)
    ax4.set_ylabel('Median elevation [m]', fontsize=12)


    ax2.annotate(
            'a',
            xy=(-0.05, 1), xycoords='axes fraction',
            xytext=(-0.8, -0.5), textcoords='offset fontsize',
            fontsize='medium', verticalalignment='top', #fontfamily='serif',
            bbox=dict(facecolor='lightgrey', edgecolor='k', pad=3.0))
    ax1.annotate(
            'b',
            xy=(1.15, 1), xycoords='axes fraction',
            xytext=(-0.8, -0.5), textcoords='offset fontsize',
            fontsize='medium', verticalalignment='top', #fontfamily='serif',
            bbox=dict(facecolor='lightgrey', edgecolor='k', pad=3.0))
    ax5.annotate(
            'c',
            xy=(0.025, 1), xycoords='axes fraction',
            xytext=(-0.8, -0.5), textcoords='offset fontsize',
            fontsize='medium', verticalalignment='top', #fontfamily='serif',
            bbox=dict(facecolor='lightgrey', edgecolor='k', pad=3.0))
    ax4.annotate(
            'd',
            xy=(0.025, 1), xycoords='axes fraction',
            xytext=(-0.8, -0.5), textcoords='offset fontsize',
            fontsize='medium', verticalalignment='top', #fontfamily='serif',
            bbox=dict(facecolor='lightgrey', edgecolor='k', pad=3.0))


    dfStuff = pd.DataFrame(columns=['vanishing', 'GI3', 'GI5'], index=['medianAr', 'sumAr'])
    print(goneglaciers.head())
    print(GI3.head())
    print(GI5.head())
    for subset, col in zip([goneglaciers, GI3, GI5],['vanishing', 'GI3', 'GI5']):
        dfStuff.loc['medianAr', col] = subset.geometry.area.median()*1e-6
        dfStuff.loc['sumAr', col] = subset.geometry.area.sum()*1e-6
        dfStuff.loc['medianEl', col] = subset['median_elev'].median()
        dfStuff.loc['min_MedianEl', col] = subset['median_elev'].min()
        dfStuff.loc['max_MedianEl', col] = subset['median_elev'].max()
    print(dfStuff)
    dfStuff.to_csv('out/summaryGoneGlaciers_area_elevation.csv')


    # for a in [ax1, ax2, ax3, ax4, ax5, ax6]:
    #     a.grid('both')
    #plt.tight_layout()
    fig.savefig('figures/glacierwise_overview.png', dpi=200, bbox_inches='tight')


# def lossrates_plots(df_prc, GI1GI2, GI2GI3, all_mrg):
#     # loss rates - two panels: stacked area since LIA and glacier wise loss as lines
#     # Define custom colors 
#     custom_colors = ["k", "grey", "#003f5c","#2f4b7c","#665191","#a05195","#d45087","#f95d6a","#ff7c43","#ffa600", "silver", "darkgrey"]

#     # stacked area chart:
#     fig, ax = plt.subplots(1, 2, figsize=(12, 7))  # Set the figure size
#     ax=ax.flatten()
#     ax[0].stackplot([1850, 1969, 1998, 2006, 2023], df_prc.values, labels=df_prc.index, colors=custom_colors)
#     ax[0].legend()
#     ax[0].grid('both')
#     ax[0].set_ylim(0, 100)
#     ax[0].set_xlim(1850, 2025)
#     ax[0].tick_params(axis='both', which='major', labelsize=12)
#     ax[0].set_xlabel('Year', fontsize=12)
#     ax[0].set_ylabel('Percentage of 1850 area [%]', fontsize=12)


#     # fig, ax = plt.subplots(1, 1, figsize=(12,6))

#     ax[1].hlines(GI1GI2['rate'], GI1GI2['yearGI1'], GI1GI2['yearGI2'], colors='grey', linewidths=0.4)
#     ax[1].hlines(GI2GI3['rate'], GI2GI3['yearGI2'], GI2GI3['Year'], colors='goldenrod', linewidths=0.4)

#     ax[1].hlines(GI_merge['loss_rate'], GI_merge['Year'], GI_merge['year'], colors='k', linewidths=0.6)

#     ax[1].hlines(all_mrg['r1'], all_mrg['Year'], all_mrg['year_mid'], colors='skyblue', linewidths=0.4)
#     ax[1].hlines(all_mrg['r2'], all_mrg['year_mid'], all_mrg['year'], colors='firebrick', linewidths=0.4)

#     ax[1].grid('both')
#     #ax.set_xticks(np.arange(1965, 2026, 10))
#     ax[1].set_xlim(1965, 2026)
#     ax[1].set_xlabel('Year', fontsize=12)
#     ax[1].set_ylabel('Area change rate [% $a^{-1}$]', fontsize=12)
#     ax[1].tick_params(axis='both', which='major', labelsize=12)

#     lnGI1GI2 = Line2D([0], [0], label='GI1-GI2, n='+str(len(GI1GI2['rate'].values))+', median: '+str(GI1GI2['rate'].median().round(decimals=0).astype(int))+'% $a^{-1}$', color='grey', linewidth=1.2, linestyle='-')
#     lnGI2GI3 = Line2D([0], [0], label='GI2-GI3, n='+str(len(GI2GI3['rate'].values))+', median: '+str(GI2GI3['rate'].median().round(decimals=0).astype(int))+'% $a^{-1}$', color='goldenrod', linewidth=1.2, linestyle='-')
#     lnGI3GI5 = Line2D([0], [0], label='GI3-GI5, n='+str(len(GI_merge['loss_rate'].values))+', median: '+str(GI_merge['loss_rate'].median().round(decimals=0).astype(int))+'% $a^{-1}$', color='k', linewidth=1.4, linestyle='-')
#     lnGI3mid = Line2D([0], [0], label='GI3-2017/18, n='+str(len(all_mrg['r1'].values))+', median: '+str(all_mrg['r1'].median().round(decimals=0).astype(int))+'% $a^{-1}$', color='skyblue', linewidth=1.2, linestyle='-')
#     lnmidGI5 = Line2D([0], [0], label='2017/18-GI5, n='+str(len(all_mrg['r2'].values))+', median: '+str(all_mrg['r2'].median().round(decimals=0).astype(int))+'% $a^{-1}$', color='firebrick', linewidth=1.2, linestyle='-')


#     hls = [lnGI1GI2, lnGI2GI3, lnGI3GI5, lnGI3mid, lnmidGI5]

#     ax[0].annotate(
#             'a',
#             xy=(1, 1), xycoords='axes fraction',
#             xytext=(-0.8, -0.5), textcoords='offset fontsize',
#             fontsize='medium', verticalalignment='top', #fontfamily='serif',
#             bbox=dict(facecolor='lightgrey', edgecolor='k', pad=3.0))
#     ax[1].annotate(
#             'b',
#             xy=(1, 1), xycoords='axes fraction',
#             xytext=(-0.8, -0.5), textcoords='offset fontsize',
#             fontsize='medium', verticalalignment='top', #fontfamily='serif',
#             bbox=dict(facecolor='lightgrey', edgecolor='k', pad=3.0))


#     fig.legend(handles=hls, loc='upper left', bbox_to_anchor=(0.56, 0.82), ncol=1)
#     fig.savefig('figures/changerates_panels.png', dpi=200, bbox_inches='tight')



def make_autopct(values):
    def my_autopct(pct):
        total = values.sum().astype(float).round(decimals=4)#.astype(int)
        #val = values.count()
        return '{p:.0f}%'.format(p=pct)
    return my_autopct


def absolute_value(val):
    total = np.sum(val)
    absolute = int(round(val * 799 / 100))
    return f'{absolute}'

def func(pct, allvals):
        absolute = int(np.round(pct/100.*np.sum(allvals)))
        return f"{pct:.1f}%\n({absolute:d} g)"


def piecharts(GI5):
    fig = plt.figure(figsize=(10, 6))
    gs = GridSpec(2, 2)# left=0.05, right=0.48, wspace=0.05)
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[1, :])

    img_flags =[0, 1, 2, 3] #should only have 0, 1, 2!!
    outline_flags =[0, 1, 2, 3]
    gb1 = GI5[['area', 'img_qf']].groupby('img_qf').sum()#.values

    gb2 = GI5[['area', 'outline_qf']].groupby('outline_qf').sum()#.values
    gb3 = GI5[['area', 'debris']].groupby('debris').sum()#.values
    gb4 = GI5[['area', 'crevs']].groupby('crevs').sum()


    # size bins
    bins = [0, 0.01, 0.1, 0.5, 1, 5, 20]
    # bins = [0, 0.5, 15]
    GI5['binned'] = pd.cut(GI5['area']*1e-6, bins)
    
    gb5 = GI5[['area', 'binned']].groupby('binned').sum()
    gb_count = GI5[['area', 'binned']].groupby('binned').count()

    gb5['lables'] = ['<=0.01', '>0.01-0.1', '>0.1-0.5', '>0.5-1.0', '>1-5', '>5']
    gb_count['lables'] = ['<=0.01 km$^2$', '>0.01-0.1 km$^2$', '>0.1-0.5 km$^2$', '>0.5-1.0 km$^2$', '>1-5 km$^2$', '>5 km$^2$']
    clrs = ['#44AA99', '#CC6677', '#999933', '#88CCEE', '#AA4499', '#DDCC77']
    patches0, texts0, autotexts0 = ax0.pie(gb5['area'], autopct='%1.1f%%',
                                             colors=clrs,#sns.color_palette('Set2'),
                                             explode=[0.13, 0, 0, 0, 0, 0],
                                             pctdistance=1.25,
                                             startangle=90,
                                             counterclock=False)
    patches1, texts1, autotexts1 = ax1.pie(gb_count['area'], autopct=absolute_value,#(gb_count['area']),
                                             colors=clrs, #sns.color_palette('Set2'),
                                             explode=[0, 0, 0, 0, 0, 0.15],#[0, 0, 0, 0.12, 0.12, 0.12],
                                             pctdistance=1.2,
                                             startangle=90,
                                             counterclock=False)

    

    ax1.legend(patches1, gb_count['lables'], title='', loc='center right',
           bbox_to_anchor=(-0.15, 0.5))
    ax0.set_title('Area fraction per size class', y=1.0, pad=+14)
    ax1.set_title('Number of glaciers per size class', y=1.0, pad=+14)


    regBG = GI5[['area', 'region']].groupby('region').sum()
    regBG = regBG.sort_values(by='area', ascending=False)
    # colors = sns.color_palette("pastel6").as_hex()

    for i, region in enumerate(regBG.index):
        dat = GI5.loc[GI5.region==region]
        asymmetric_error = [(dat['median_elev']-dat['min_elev']).values, (dat['max_elev'] - dat['median_elev']).values]
        marker = 'o'
        #label=region, #, c=sns.color_palette())
        ax2.semilogx(dat['area']*1e-6,dat['median_elev'], marker=marker, markersize=4, markeredgecolor='grey', linewidth=0.7, c='k', zorder=50, linestyle='')
        ax2.errorbar(dat['area']*1e-6,dat['median_elev'], yerr=asymmetric_error, markersize=5, fmt='o', linewidth=0.7, c='grey')
        
        # ax2.errorbar(dat['area']*1e-6,dat['median_elev'], yerr=asymmetric_error, markersize=5, fmt='o', linewidth=0.7, c='grey')
        # ax2.scatter(dat['area']*1e-6,dat['median_elev'], marker=marker, s=4, c='k', zorder=50)

    # ax2.set_xlim(-0.01, 15.5)
    ax2.set_xlim(0.0, 15.5)
    

    largest = GI5.sort_values(by='area', ascending=False).head(10)
    largest = largest.sort_values(by='area', ascending=True)
    # print(largest[['id', 'name']])
    largest.loc[largest['id'] == 2125.0, 'name'] = 'Hintereis Ferner'

    for gl in largest['name'].values:
        dat = largest.loc[largest['name']==gl]
        asymmetric_error = [(dat['median_elev']-dat['min_elev']).values, (dat['max_elev'] - dat['median_elev']).values]
        ar = (dat['area'].values[0]*1e-6).astype(float).round(decimals=2)#.astype(str)
        ax2.scatter(dat['area']*1e-6, dat['median_elev'], s=12, label=gl+': '+'{:.2f}'.format(ar)+' km$^2$', zorder=100)
        ax2.errorbar(dat['area']*1e-6, dat['median_elev'], yerr=asymmetric_error,  markersize=4, fmt='o', linewidth=0.7, zorder=200)
    
    
    ax2.grid('both')
    ax2.tick_params(axis='both', which='major', labelsize=12)
    ax2.set_xlabel('Glacier size [km$^2$]', fontsize=12)  
    ax2.set_ylabel('Elevation [m a.s.l.]', fontsize=12)    
    ax2.xaxis.tick_top()
    ax2.xaxis.set_label_position('top')
    ax2.legend(loc='lower right', bbox_to_anchor=(1.04,-0.6), ncol=3, fontsize=10)
    # ax3.tick_params(axis='both', which='major', labelsize=12)

    ax0.annotate(
            'a',
            xy=(0, 1), xycoords='axes fraction',
            xytext=(-0.1, -0.5), textcoords='offset fontsize',
            fontsize='medium', verticalalignment='top', #fontfamily='serif',
            bbox=dict(facecolor='lightgrey', edgecolor='k', pad=3.0))
    ax1.annotate(
            'b',
            xy=(1, 1), xycoords='axes fraction',
            xytext=(-0.8, -0.5), textcoords='offset fontsize',
            fontsize='medium', verticalalignment='top', #fontfamily='serif',
            bbox=dict(facecolor='lightgrey', edgecolor='k', pad=3.0))
    ax2.annotate(
            'c',
            xy=(0, 1), xycoords='axes fraction',
            xytext=(-1.8, 0.5), textcoords='offset fontsize',
            fontsize='medium', verticalalignment='top', #fontfamily='serif',
            bbox=dict(facecolor='lightgrey', edgecolor='k', pad=3.0))
    print(largest)

    fig.savefig('figures/GI5_area_pies_log.png', bbox_inches='tight', dpi=200)



def piecharts_2(GI5):
    fig = plt.figure(figsize=(10, 6))
    # ax = ax.flatten()

    gs = GridSpec(2, 2)# left=0.05, right=0.48, wspace=0.05)
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[1, 1])
    
    # def func(pct, allvals):
    #     absolute = int(np.round(pct/100.*np.sum(allvals)))
    #     return f"{pct:.1f}%\n({absolute:d} g)"

    img_flags =[0, 1, 2, 3] #should only have 0, 1, 2!!
    outline_flags =[0, 1, 2, 3]
    gb1 = GI5[['area', 'img_qf']].groupby('img_qf').sum()
    gb1_count = GI5[['area', 'img_qf']].groupby('img_qf').count()
    print(gb1)


    gb1['lables'] = ['good (0)', 'medium (1)', 'poor (2)']
    clrs = ['#44AA99', '#CC6677', '#999933', '#88CCEE', '#AA4499', '#DDCC77']

    patches0, texts0, autotexts0 = ax0.pie(gb1['area'], autopct=make_autopct(gb1['area']),
                                             colors=clrs,#sns.color_palette('Set2'),
                                             explode=[0, 0, 0.15],
                                             pctdistance=1.2,
                                             startangle=90,
                                             counterclock=False)
    patches1, texts1, autotexts1 = ax1.pie(gb1_count['area'], autopct=absolute_value,#make_autopct(gb1_count['area']),
                                             colors=clrs, #sns.color_palette('Set2'),
                                             # explode=[0, 0, 0, 0, 0, 0.15],
                                             pctdistance=1.2,
                                             startangle=90,
                                             counterclock=False)

    ax1.legend(patches1, gb1['lables'], title='', loc='center right',
           bbox_to_anchor=(-0.15, 0.5))

    ax0.set_title('Area per image quality class', y=1.0, pad=+14)
    ax1.set_title('Number of glaciers per image quality class', y=1.0, pad=+14)

    


    gb2 = GI5[['area', 'outline_qf']].groupby('outline_qf').sum()
    gb2_count = GI5[['area', 'outline_qf']].groupby('outline_qf').count()

    gb2['lables'] = ['good (0)', 'medium (1)', 'poor (2)', 'very uncertain (3)']


    patches2, texts2, autotexts2 = ax2.pie(gb2['area'], autopct=make_autopct(gb2['area']),
                                             colors=clrs, #ns.color_palette('Set2'),
                                             explode=[0, 0, 0, 0.15],
                                             pctdistance=1.2,
                                             startangle=90,
                                             counterclock=False)

    patches3, texts3, autotexts3 = ax3.pie(gb2_count['area'], autopct=absolute_value,#make_autopct(gb2_count['area']),
                                             colors=clrs, #sns.color_palette('Set2'),
                                             # explode=[0, 0, 0, 0, 0, 0.15],
                                             pctdistance=1.2,
                                             startangle=90,
                                             counterclock=False)
    

    ax3.legend(patches3, gb2['lables']
        , title='', loc='center right',
           bbox_to_anchor=(-0.0, 0.2))

    ax2.set_title('Area per outline quality class', y=1.0, pad=+14)
    ax3.set_title('Number of glaciers per outline quality class', y=1.0, pad=+14)

    ax0.annotate(
            'a',
            xy=(0, 1), xycoords='axes fraction',
            xytext=(-0.1, -0.5), textcoords='offset fontsize',
            fontsize='medium', verticalalignment='top', #fontfamily='serif',
            bbox=dict(facecolor='lightgrey', edgecolor='k', pad=3.0))
    ax1.annotate(
            'b',
            xy=(1, 1), xycoords='axes fraction',
            xytext=(-0.8, -0.5), textcoords='offset fontsize',
            fontsize='medium', verticalalignment='top', #fontfamily='serif',
            bbox=dict(facecolor='lightgrey', edgecolor='k', pad=3.0))

    ax2.annotate(
            'c',
            xy=(0, 1), xycoords='axes fraction',
            xytext=(-0.1, -0.5), textcoords='offset fontsize',
            fontsize='medium', verticalalignment='top', #fontfamily='serif',
            bbox=dict(facecolor='lightgrey', edgecolor='k', pad=3.0))
    ax3.annotate(
            'd',
            xy=(1, 1), xycoords='axes fraction',
            xytext=(-0.8, -0.5), textcoords='offset fontsize',
            fontsize='medium', verticalalignment='top', #fontfamily='serif',
            bbox=dict(facecolor='lightgrey', edgecolor='k', pad=3.0))

    fig.savefig('figures/GI5_QF_pies.png', bbox_inches='tight', dpi=200)


def piecharts_3(GI5):
    fig = plt.figure(figsize=(10, 6))
    # ax = ax.flatten()

    gs = GridSpec(2, 2)# left=0.05, right=0.48, wspace=0.05)
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[1, 1])
    
    # def func(pct, allvals):
    #     absolute = int(np.round(pct/100.*np.sum(allvals)))
    #     return f"{pct:.1f}%\n({absolute:d} g)"


    gb1 = GI5[['area', 'debris']].groupby('debris').sum()
    gb1_count = GI5[['area', 'debris']].groupby('debris').count()


    gb1['lables'] = ['no debris (0)', 'partial debris (1)', 'mostly debris (2)', 'full debris (3)', 'unsure (4)']
    clrs = ['#44AA99', '#CC6677', '#999933', '#88CCEE', '#AA4499', '#DDCC77']

    patches0, texts0, autotexts0 = ax0.pie(gb1['area'], autopct=make_autopct(gb1['area']),#autopct='%1.1d%%',
                                             colors=clrs,#sns.color_palette('Set2'),
                                             explode=[0, 0, 0.15, 0, 0.15],
                                             pctdistance=1.2,
                                             startangle=90,
                                             counterclock=False)
    patches1, texts1, autotexts1 = ax1.pie(gb1_count['area'], autopct=absolute_value,#make_autopct(gb1_count['area']),
                                             colors=clrs, #sns.color_palette('Set2'),
                                             # explode=[0, 0, 0, 0, 0, 0.15],
                                             pctdistance=1.2,
                                             startangle=90,
                                             counterclock=False)

    ax1.legend(patches1, gb1['lables'], title='', loc='center right',
           bbox_to_anchor=(-0.15, 0.5))

    ax0.set_title('Area per "debris" class', y=1.0, pad=+14)
    ax1.set_title('Number of glaciers per "debris" class', y=1.0, pad=+14)


    gb2 = GI5[['area', 'crevs']].groupby('crevs').sum()
    gb2_count = GI5[['area', 'crevs']].groupby('crevs').count()

    print(gb2_count)
 

    gb2['lables'] = ['no visible crevasses (0)', 'visible crevasses (1)', 'unsure (2)']


    patches2, texts2, autotexts2 = ax2.pie(gb2['area'], autopct=make_autopct(gb2['area']),
                                             colors=clrs, #sns.color_palette('Set2'),
                                             explode=[0, 0, 0.15],
                                             pctdistance=1.2,
                                             startangle=90,
                                             counterclock=False)
    patches3, texts3, autotexts3 = ax3.pie(gb2_count['area'], autopct=absolute_value,#make_autopct(gb2_count['area']),
                                             colors=clrs, #sns.color_palette('Set2'),
                                             # explode=[0, 0, 0, 0, 0, 0.15],
                                             pctdistance=1.2,
                                             startangle=90,
                                             counterclock=False)
    

    ax3.legend(patches3, gb2['lables'], title='', loc='center right',
           bbox_to_anchor=(-0.0, 0.7))

    ax2.set_title('Area per "crevasses" class', y=1.0, pad=+14)
    ax3.set_title('Number of glaciers per "crevasses" class', y=1.0, pad=+14)



    ax0.annotate(
            'a',
            xy=(0, 1), xycoords='axes fraction',
            xytext=(-0.1, -0.5), textcoords='offset fontsize',
            fontsize='medium', verticalalignment='top', #fontfamily='serif',
            bbox=dict(facecolor='lightgrey', edgecolor='k', pad=3.0))
    ax1.annotate(
            'b',
            xy=(1, 1), xycoords='axes fraction',
            xytext=(-0.8, -0.5), textcoords='offset fontsize',
            fontsize='medium', verticalalignment='top', #fontfamily='serif',
            bbox=dict(facecolor='lightgrey', edgecolor='k', pad=3.0))

    ax2.annotate(
            'c',
            xy=(0, 1), xycoords='axes fraction',
            xytext=(-0.1, -0.5), textcoords='offset fontsize',
            fontsize='medium', verticalalignment='top', #fontfamily='serif',
            bbox=dict(facecolor='lightgrey', edgecolor='k', pad=3.0))
    ax3.annotate(
            'd',
            xy=(1, 1), xycoords='axes fraction',
            xytext=(-0.8, -0.5), textcoords='offset fontsize',
            fontsize='medium', verticalalignment='top', #fontfamily='serif',
            bbox=dict(facecolor='lightgrey', edgecolor='k', pad=3.0))

    fig.savefig('figures/GI5_DebrisCrevs_pies.png', bbox_inches='tight', dpi=200)



def fig_hist(GI5):
    GI5['area_km'] = GI5.geometry.area *1e-6
    fig, ax = plt.subplots(1, 1, figsize=(9,6))
    ax.hist(GI5['area_km'], bins=np.arange(0, 14, 0.2))
    ax.grid('both')
    fig.savefig('figures/hist_allregions.png', bbox_inches='tight', dpi=200)



 