import numpy as np
import pandas as pd
import glob
import geopandas as gpd
import matplotlib.pyplot as plt
from statistics import mode


# function to set size dependent relative uncertainties
def add_reluncertainties(ol, arcol):
    # currently not using exploded version!
    ol['area_unc_r'] = np.nan
    ol.loc[ol.area > 1e6, 'area_unc_r'] = 0.015
    ol.loc[(ol.area < 1e6) & (ol.area >= 0.1e6), 'area_unc_r'] = 0.05
    ol.loc[(ol.area < 0.1e6) & (ol.area >= 0.05e6), 'area_unc_r'] = 0.10
    ol.loc[(ol.area < 0.05e6), 'area_unc_r'] = 0.25

    # if outline QF is 2 or 3, apply uncertainties independent of size (overwrite the others)
    ol.loc[ol['outline_qf'] == 2, 'area_unc_r'] = 0.25
    ol.loc[ol['outline_qf'] == 3, 'area_unc_r'] = 0.50

    # absolute uncertainties
    ol['unc_abs'] = ol.area * ol['area_unc_r']
    return(ol)

def circles(dat):
    dat['id'] = dat.index
    dat.index.name = 'ix'
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
    dat_all = dat_nocircles.dissolve(by='id')
    return(dat_all)


# load files, add area column and sort by index
def load(fn):
    gdf = gpd.read_file(fn)
    gdf['area'] = gdf.geometry.area
    gdf.index = gdf['id']
    gdf.drop(columns=['id'], inplace=True)
    gdf = gdf.sort_index(ascending=True)
    return gdf


# fix bad data type for flightstr string lists
def fixtype(merged):
        merged['flightstr'] = merged['flightstr'].astype(str)
        merged['flightstr'] = merged['flightstr'].str.replace('[', '')
        merged['flightstr'] = merged['flightstr'].str.replace(']', '')
        merged['flightstr'] = merged['flightstr'].str.replace('\'', '')
        merged['flightstr'] = merged['flightstr'].str.replace('\'', '')
        return (merged)


# -------######-----------
# load some files: 
# table of glaciers with positive changes between intermediate 2017/18 inventory and 2023:
pos_changeOtzMid = pd.read_csv('/Users/leahartl/Desktop/inventare_2025/processing/inventories/out/pos/poschange_otz_2017_2023.csv')
pos_changeGI3GI5 = pd.read_csv('/Users/leahartl/Desktop/inventare_2025/processing/inventories/out/pos/poschange_GI3GI5.csv')
# load main inventory file of Ötztal Alps (contains "in situ" versions of the outlines)
oetztal_MS = load('/Users/leahartl/Desktop/inventare_2025/mergedfiles/Oetztaler_Alpen_GI5_proc.geojson')
# remove circles
oetztal_MS = circles(oetztal_MS)
oetztal_MS = fixtype(oetztal_MS)
# add uncertainties
oetztal_MS = add_reluncertainties(oetztal_MS, 'area')
# load alternative file of Ötztal Alps outlines (only mapped from aerial imagery)
oetztal_IGF = load('/Users/leahartl/Desktop/inventare_2025/Data/Oetztaler_Alpen_2023_versionAnneetal.geojson')
oetztal_IGF = circles(oetztal_IGF)
#oetztal_IGF = fixtype(oetztal_IGF)


print('area km2, All Ötztal v. Markus', oetztal_MS.area.sum()*1e-6)
print(oetztal_MS.shape)
print('area km2, All Ötztal v. IGF', oetztal_IGF.area.sum()*1e-6)
print(oetztal_IGF.shape)
print('dif=', oetztal_MS.area.sum()*1e-6 - oetztal_IGF.area.sum()*1e-6)

# extract glaciers for which there are two versions
# assumption: all such glaciers can be identified by filtering the analyst column for M. Strudl
# or by filtering for "in situ survey" in data type 2 (this would include vernagtferner)

subset_MS = oetztal_MS.loc[oetztal_MS['data_type2'].str.contains('in situ survey')]
subset_IGF = oetztal_IGF.loc[oetztal_IGF.index.isin(subset_MS.index.unique())]
subset_IGF.drop(columns=['index'], inplace=True)

print('area km2, subset MS', subset_MS.area.sum()*1e-6)
print(subset_MS.shape)
print('area km2, subset IGF', subset_IGF.area.sum()*1e-6)
print(subset_IGF.shape)
print('dif=', subset_MS.area.sum()*1e-6 - subset_IGF.area.sum()*1e-6)
print(subset_MS.sort_values(by='name'))

print('glaciers in subset:', len(subset_MS.index))
print('poschange in subset MS GImid-GI5:', subset_MS.loc[subset_MS.index.isin(pos_changeOtzMid['id'].values)])
print('poschange in subset MS GI3-GI5:', subset_MS.loc[subset_MS.index.isin(pos_changeGI3GI5['id'].values)])

stop
# subset_MS.to_file('out/subset_oetztal_MS.geojson')
# subset_IGF.to_file('out/subset_oetztal_IGF.geojson')


compare = subset_MS[['name', 'area',  'debris', 'outline_qf', 'img_qf', 'analyst', 'area_unc_r']].merge(subset_IGF[['area']], left_index=True, right_index=True, suffixes=['_ins', '_rs'], how='outer')
compare['area_rs'] = compare['area_rs'].fillna(0)
compare['perc_dif'] = (100*(compare['area_ins']-compare['area_rs']) / compare['area_ins']).round(decimals=1)

compare['area_unc_r'] = compare['area_unc_r']*100

compare = compare.sort_values(by='perc_dif')
# write to csv file: 
compare.to_csv('out/compare_debris.csv')


def fig_flags(df):

    fig, ax = plt.subplots(2, 2, figsize=(8,6))
    ax = ax.flatten()
    ax[0].scatter(df.area*1e-6, df.debris)
    ax[0].set_xlabel('glacier area (km2)')
    ax[0].set_ylabel('debris')
    # ax[0].set_xlim(0, 1.8)
    # ax[0].set_ylim(0, 100)


    ax[1].scatter(df.img_qf, df.debris,)
    ax[1].set_xlabel('image QF')
    ax[1].set_ylabel('debris')

    ax[2].scatter(df.img_qf, df.outline_qf,)
    ax[2].set_xlabel('image QF')
    ax[2].set_ylabel('outline QF')

    ax[3].scatter(df.outline_qf, df.debris)
    ax[3].set_xlabel('outline QF')
    ax[3].set_ylabel('debris')


    ax[0].set_title('Area vs debris')
    ax[1].set_title('QF image vs QF debris')
    ax[2].set_title('QF image vs QF outline')
    ax[3].set_title('QF outline vs QF debris')


    tocheck = df.loc[(df.debris==3) & (df.outline_qf<2)]
    print(tocheck)

    plt.tight_layout()


def fig_compare2(compare):
    fig, ax = plt.subplots(1, 2, figsize=(9,6), sharey=True)
    ax = ax.flatten()

    clrs = ['skyblue', 'grey', 'brown', 'k']
    lbl = ['no debris', 'partial debris', 'mostly debris', 'full debris',]

    unc_high = compare.loc[compare['perc_dif'] > compare['area_unc_r']]
    print('difference greater than uncertainty, number of cases: ', len(unc_high.index))
    unc_OK = compare.loc[compare['perc_dif'] <= compare['area_unc_r']]
    print('difference <= uncertainty, number of cases: ', len(unc_OK.index))
    
    for i, deb in enumerate([0, 1, 2, 3]):
        temp1 = unc_high.loc[unc_high.debris == deb]
        temp2 = unc_OK.loc[unc_OK.debris == deb]
        ax[0].scatter(temp1.area_ins*1e-6, temp1.perc_dif, color=clrs[i], label=lbl[i], s=60)
        ax[0].scatter(temp2.area_ins*1e-6, temp2.perc_dif, facecolors='none', edgecolors=clrs[i], s=60)

    firm = compare.loc[compare.index==2099]

    ax[0].annotate('Firmisan Ferner', xy=(firm.area_ins*1e-6, firm.perc_dif), textcoords='offset points',
                arrowprops=dict(arrowstyle='-|>', color='k'),
                bbox=dict(boxstyle="square,pad=0.1",
                fc="white", ec="k", lw=0.5))

    ax[0].set_xlabel('Glacier area ($km^2$)', fontsize=12)
    ax[0].set_ylabel('Area difference:\n local knowledge vs. aerial imagery only (%)', fontsize=12)
    ax[0].set_xlim(0, 1.5)
    ax[0].set_ylim(-1, 104)
    ax[1].set_ylim(0, 104)
    ax[0].legend()

    nodeb = compare.loc[compare.debris==0]
    somedeb = compare.loc[compare.debris==1]
    mostlydeb = compare.loc[compare.debris==2]
    fulldeb = compare.loc[compare.debris==3]
    unclear = compare.loc[compare.debris==4]

    print('no debris: ', nodeb)


    bplot = ax[1].boxplot([nodeb['perc_dif'].values, somedeb['perc_dif'].values, 
                        mostlydeb['perc_dif'].values, fulldeb['perc_dif'].values],
                        patch_artist=True,# notch=True,
                        tick_labels=['no debris, n='+str(len(nodeb['perc_dif'].values)), 'partial debris, n='+str(len(somedeb['perc_dif'].values)),
                           'mostly debris, n='+str(len(mostlydeb['perc_dif'].values)), 'full debris, n='+str(len(fulldeb['perc_dif'].values))])#, 'unclear'])

    for patch, color in zip(bplot['boxes'], clrs):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)

    df = pd.DataFrame(columns=['nodeb', 'somedeb', 'mostlydeb', 'fulldeb'], index=['median', 'min', 'max'])
    for c, val in zip(['nodeb', 'somedeb', 'mostlydeb', 'fulldeb'],[nodeb, somedeb, mostlydeb, fulldeb]):
        df.loc['median', c] = val['area_ins'].median()
        df.loc['min', c] = val['area_ins'].min()
        df.loc['max', c] = val['area_ins'].max()
        df.loc['med_prcdif', c] = val['perc_dif'].median()
        df.loc['min_prcdif', c] = val['perc_dif'].min()
        df.loc['max_prcdif', c] = val['perc_dif'].max()

    print(df)

    tocheck = compare.loc[(compare.debris==0) & (compare.perc_dif>50)]

    for label in ax[1].get_xticklabels():
        label.set_rotation(45)

    ax[0].grid('both')
    ax[1].grid('y')

    ax[0].annotate("a",
             xy=(0.08, 0.95), xycoords='axes fraction', fontsize=14,
             ha="center", va="center",
             bbox=dict(boxstyle="square,pad=0.2",
             fc="silver", ec="k", lw=2))
    ax[1].annotate("b",
             xy=(0.05, 0.95), xycoords='axes fraction', fontsize=14,
             ha="center", va="center",
             bbox=dict(boxstyle="square,pad=0.2",
             fc="silver", ec="k", lw=2))

    fig.savefig('figures/debriscover_comparisons.png', dpi=200, bbox_inches='tight')






# fig_compare(compare)

fig_compare2(compare)
# fig_flags(oetztal_MS)


plt.show()