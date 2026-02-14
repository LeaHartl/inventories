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
# import helpers_plots as hlpplots


## ------ load or extract data -----------
# load GI data from folders:
def getGI(fls, strnr):
    # list of all regions
    GIlist = []
    for f in fls:
        # extract region name from the file path:
        reg = f.split('/')[-1]
        r = reg[:strnr]
        # load regional GI file:
        GIreg = gpd.read_file(f)
        GIreg.to_crs(epsg=31287, inplace=True)
        GIreg['region'] = r

        # ensure consistent column names (needed for older GI)
        if 'Jahr' in GIreg.columns:
            GIreg.rename(columns={'Jahr': 'year'}, inplace=True)
        if 'Year' in GIreg.columns:
            GIreg.rename(columns={'Year': 'year'}, inplace=True)
        if 'nr' in GIreg.columns:
            GIreg.rename(columns={'nr': 'id'}, inplace=True)
        if 'Gletschern' in GIreg.columns:
            GIreg.rename(columns={'Gletschern': 'name'}, inplace=True)

        if 'year' in GIreg.columns:
            GIreg['year'] = GIreg['year'].astype(int)
        GIlist.append(GIreg)

    GI = pd.concat(GIlist)

    # add lat lon centroids
    GI['x'] = GI.centroid.to_crs(epsg=4236).x
    GI['y'] = GI.centroid.to_crs(epsg=4236).y

    # sort by region
    GI.sort_values(by='region')

    return(GI)


# get regional inventories
def subregions(GI5):
    # Salzburg
    Salzburg = '/Users/leahartl/Desktop/inventare_2025/GI_Salzburg_Bertolotti_2018/GI5_Sbg_Orthofotos_new_lea/'
    # get files:
    fls_Salzburg = glob.glob(Salzburg+'*.shp')
    # 5033 und 5034 vertauscht!
    salzb = getGI(fls_Salzburg, -4)
    salzb['area'] = salzb.geometry.area
    salzb = salzb.rename(columns={'year':'year_mid', 'area':'area_mid'})
    salzb = salzb.dissolve(by='id')
    salzb['id'] = salzb.index

    # Stubai
    stb = gpd.read_file('/Users/leahartl/Desktop/inventare_2025/GI_Salzburg_Bertolotti_2018/Stubai_GI5/Stubai_GI5_pangaea.shp')
    stb.to_crs(GI5.crs, inplace=True)
    stb['area_mid'] = stb.geometry.area
    stb = stb.rename(columns={'nr':'id', 'Gletschern':'name', 'Year': 'year_mid'})
    stb = stb.drop(columns=['Area'])

    # Ötztal - change HEF and Toteis to HEF with Toteis, ID 2125
    otz = gpd.read_file('/Users/leahartl/Desktop/inventare_2025/GI_Salzburg_Bertolotti_2018/Oetztaler_Alpen_GI5/Oetztaler_Alpen_GI5_pangaea.shp')
    otz.to_crs(GI5.crs, inplace=True)
    otz['area_mid'] = otz.geometry.area
    otz = otz.rename(columns={'nr':'id', 'Gletschern':'name', 'Year': 'year_mid'})
    otz = otz.drop(columns=['Area'])

    # Silvretta Tirol
    silvT = gpd.read_file('/Users/leahartl/Desktop/inventare_2025/GI_Salzburg_Bertolotti_2018/Gl_silv_T_2018/Gl_silv_T_2018.shp')
    silvT.to_crs(GI5.crs, inplace=True)
    silvT['area_mid'] = silvT.geometry.area
    silvT = silvT.rename(columns={'nr':'id', 'Name':'name', 'year': 'year_mid'})

    # Silvretta Vorarlberg
    VBGfolder = '/Users/leahartl/Desktop/svenja/pangaea_ueberarbeitet/Vorarlberg_inventar/'
    VBG17 = gpd.read_file(VBGfolder+'outlines2017.shp')
    VBG17.to_crs(GI5.crs, inplace=True)
    VBG17['area_mid'] = VBG17.geometry.area
    
    # merge all subregions:
    allregs = pd.concat([salzb, stb, otz, silvT, VBG17])
    allregs = allregs[['id', 'name', 'year_mid', 'area_mid', 'geometry']]
    allregs['area_mid'] = allregs.geometry.area
    return(allregs)


# load files of vanished and vanishing glaciers for all regions,
# combine both, get centroids, save to file.
def getGoneGlaciers(fls_vanishing, fls_vanished):
    # list of all regions
    vanishing_list = []
    vanished_list = []

    # vanishing
    for f in fls_vanishing:
        # extract region name from the file path:
        reg = f.split('/')[-1]
        r = reg[:-38]
        # load regional file:
        vanishing_reg = gpd.read_file(f)
        if vanishing_reg.shape[0] > 0:
            vanishing_reg.to_crs(epsg=31287, inplace=True)
            vanishing_reg['region'] = r
            vanishing_reg['status'] = 'vanishing'
            if 'Jahr' in vanishing_reg.columns:
                vanishing_reg.rename(columns={'Jahr': 'Year'}, inplace=True)
            vanishing_reg['area_GI3_km2'] = vanishing_reg['area_GI3']*1e-6
            vanishing_list.append(vanishing_reg)

    vanishing_df = pd.concat(vanishing_list)

    # vanished
    for f in fls_vanished:
        # extract region name from the file path:
        reg = f.split('/')[-1]
        r = reg[:-37]
    
        # load regional file:
        vanished_reg = gpd.read_file(f)
        if vanished_reg.shape[0] > 0:
            vanished_reg.to_crs(epsg=31287, inplace=True)
            vanished_reg['region'] = r
            vanished_reg['status'] = 'vanished'
            if 'Jahr' in vanished_reg.columns:
                vanished_reg.rename(columns={'Jahr': 'Year'}, inplace=True)

            vanished_reg['area_GI3_km2'] = vanished_reg['area_GI3']*1e-6
            vanished_list.append(vanished_reg)

    
    vanished_df = pd.concat(vanished_list)

    # combine vanishing and vanished
    gone_df = pd.concat([vanishing_df, vanished_df])
    
    # get centroids and save to file.
    goneglaciers_cntr = gone_df[['id', 'name', 'Year', 'area_GI3', 'region', 'status', 'geometry']]
    goneglaciers_cntr.rename(columns={'Year':'year_GI3', 'status': 'status_GI5'}, inplace=True)
    goneglaciers_cntr.geometry = goneglaciers_cntr.geometry.centroid
    goneglaciers_cntr.to_file('out/vanishing_glaciers.geojson')

    # write some data to summary table.
    df_summary = pd.DataFrame(columns=['all', 'vanishing', 'vanished'], index=['count'])

    for subset, col in zip([gone_df, vanishing_df, vanished_df], ['all', 'vanishing', 'vanished']):
        df_summary.loc['count', col] = len(subset['status'].values)
        df_summary.loc['count >=0.01 in GI3', col] = len(subset.loc[subset['area_GI3_km2'] >= 0.01].values)
        df_summary.loc['count <0.01 in GI3', col] = len(subset.loc[subset['area_GI3_km2'] < 0.01].values)
        df_summary.loc['total area gone glaciers(km2) in GI3', col] = subset['area_GI3'].sum()*1e-6

    df_summary.to_csv('out/goneglaciers_summary.csv')
    return(gone_df)


# extract GI3 and GI5 outlines of glaciers for round robin experiment, save to file
def forMappingExp(GI3, GI5):
    idsGI3 = [13029, 3028, 6013, 4038, 14033, 4027]
    idsGI5 = [3028, 6013, 4038, 14033, 4027]

    examplesGI5 = GI5.loc[GI5['id'].isin(idsGI5)]
    examplesGI5['area_GI5_km2'] = examplesGI5['area']*1e-6
    examplesGI5 = examplesGI5.sort_values(by='area_GI5_km2')
    examplesGI5.index = examplesGI5['id']
    examplesGI5.drop(columns=['id'], inplace=True)

    examplesGI3 = GI3.loc[GI3['id'].isin(idsGI3)]
    examplesGI3['area_GI3_km2'] = examplesGI3['area']*1e-6
    examplesGI3 = examplesGI3.sort_values(by='area_GI3_km2')
    examplesGI3.index = examplesGI3['id']
    examplesGI3.drop(columns=['min_elev', 'median_elev', 'mean_elev', 'max_elev', 'id', 'index'], inplace=True)

    examplesGI3.to_file('out/examples_RR_GI3.geojson')
    examplesGI5.to_file('out/examples_RR_GI5.geojson')


## ------ process and produce output -----------

# function to set size dependent relative uncertainties with extra AGI5 categories for outline QF
def add_reluncertainties(ol, arcol):
    # currently not using exploded version!
    ol['area_unc_r'] = np.nan
    ol['area'] = ol[arcol]
    ol.loc[ol.area > 1e6, 'area_unc_r'] = 0.015
    ol.loc[(ol.area < 1e6) & (ol.area >= 0.1e6), 'area_unc_r'] = 0.05
    ol.loc[(ol.area < 0.1e6) & (ol.area >= 0.05e6), 'area_unc_r'] = 0.10
    ol.loc[(ol.area < 0.05e6), 'area_unc_r'] = 0.25

    # if outline QF is 2 or 3, apply uncertainties independent of size (overwrite the others)
    ol.loc[ol['outline_qf'] == 2, 'area_unc_r'] = 0.25
    ol.loc[ol['outline_qf'] == 3, 'area_unc_r'] = 0.50

    # absolute uncertainties
    ol['unc_abs_GI5'] = ol.area * ol['area_unc_r']
    ol.drop(columns=['area'], inplace=True)
    return(ol)


# function to set size dependent relative uncertainties as per AGI3
def add_reluncertaintiesGI3(ol, arcol):
    # currently not using exploded version!
    ol['area_unc_r'] = np.nan
    ol['area'] = ol[arcol]
    ol.loc[ol.area > 1e6, 'area_unc_r'] = 0.015
    ol.loc[ol.area <= 1e6, 'area_unc_r'] = 0.05

    ol['unc_abs_GI3'] = ol.area * ol['area_unc_r']
    ol.drop(columns=['area'], inplace=True)
    return(ol)





# group by region and compute regional summary stats:
def prepareTable(GI_merge, GI3, GI5, goneglaciers):
    grouped = GI_merge[['region_GI3', 'area_GI5', 'area_GI3']].groupby('region_GI3').sum()
    grouped_regGI5 = GI5[['id', 'region']].groupby('region').count()
    grouped_regGI3 = GI3[['id', 'region']].groupby('region').count()

    grouped['area_GI5_km2'] = (grouped['area_GI5']*1e-6).astype(float).round(decimals=3)

    grouped = grouped.merge(grouped_regGI5, left_index=True, right_index=True, suffixes=('', '_GI5'))
    grouped = grouped.merge(grouped_regGI3, left_index=True, right_index=True, suffixes=('', '_GI3'))

    meanlossrates = GI_merge[['loss_rate', 'region_GI3']].groupby('region_GI3').median().astype(float).round(decimals=1)
    grouped = grouped.merge(meanlossrates, left_index=True, right_index=True)


    grouped_vanished = goneglaciers.loc[goneglaciers.status=='vanished',['name', 'region']].groupby(['region']).count()
    grouped_vanished.rename(columns={'name': 'count_vanished'}, inplace=True)
    grouped_vanishing = goneglaciers.loc[goneglaciers.status=='vanishing',['name', 'region']].groupby(['region']).count()
    grouped_vanishing.rename(columns={'name': 'count_vanishing'}, inplace=True)


    # merge the dataframes:
    grouped = grouped.merge(grouped_vanishing, left_index=True, right_index=True, how='outer')
    grouped = grouped.merge(grouped_vanished, left_index=True, right_index=True, how='outer')
    grouped['perc_of_total'] = (100*grouped['area_GI5'] / grouped['area_GI5'].sum()).astype(float).round(decimals=2)
    grouped['loss_km'] = ((grouped['area_GI5'] - grouped['area_GI3'])*1e-6).astype(float).round(decimals=2)
    grouped['perc_loss'] = (100*(grouped['area_GI5'] - grouped['area_GI3'])/grouped['area_GI3']).astype(float).round(decimals=1)

    # USE THIS FOR ROOT SUM SQUARE UNCERTAINTY:
    # unc = GI_merge[['region_GI3', 'unc_abs']].groupby('region_GI3')['unc_abs'].apply(sumsquare)

    # USE THIS FOR SIMPLE SUM:
    unc = GI_merge[['region_GI3', 'unc_abs_GI5']].groupby('region_GI3')['unc_abs_GI5'].sum()
    unc_change = GI_merge[['region_GI3', 'unc_change']].groupby('region_GI3')['unc_change'].sum()

    grouped = grouped.merge(unc, left_index=True, right_index=True, how='outer')
    grouped = grouped.merge(unc_change, left_index=True, right_index=True, how='outer')

    grouped[['unc_abs_GI5', 'unc_change']] = grouped[['unc_abs_GI5', 'unc_change']]*1e-6
    grouped[['unc_abs_GI5', 'unc_change']] = grouped[['unc_abs_GI5', 'unc_change']].astype(float).round(decimals=3)

    return(grouped)


# format grouped dataframe to make and export a table:
def makeTable(grouped, GI_merge, GI3, GI5):
    fortable = grouped.sort_values(by='area_GI5_km2', ascending=False)
    fortable['lostglaciers'] = (fortable['count_vanishing'].fillna(0)+fortable['count_vanished'].fillna(0)).astype(int)
    fortable.rename(columns={'id': 'count_glaciers'}, inplace=True)
    fortable = fortable[['area_GI5_km2', 'unc_abs_GI5', 'perc_of_total', 'loss_km', 'unc_change', 'perc_loss', 'loss_rate','count_glaciers',
                     'lostglaciers', 'count_vanishing', 'count_vanished']]

    total = fortable.sum()
    print(total)

    # get min and max years of GI3 and GI5
    fortable['yearGI5_min'] = GI_merge[['year_GI5', 'region_GI5']].groupby('region_GI5').min()
    fortable['yearGI5_max'] = GI_merge[['year_GI5', 'region_GI5']].groupby('region_GI5').max()

    fortable['yearGI3_min'] = GI_merge[['year_GI3', 'region_GI3']].groupby('region_GI3').min()
    fortable['yearGI3_max'] = GI_merge[['year_GI3', 'region_GI3']].groupby('region_GI3').max()

    # format year range for output:
    fortable['yearsGI5'] = fortable['yearGI5_min'].astype(int).astype(str)+'-'+fortable['yearGI5_max'].astype(int).astype(str)
    fortable.loc[fortable['yearGI5_min'] == fortable['yearGI5_max'], 'yearsGI5'] = fortable['yearGI5_min'].astype(int).astype(str)
    fortable['yearsGI3'] = fortable['yearGI3_min'].astype(int).astype(str)+'-'+fortable['yearGI3_max'].astype(int).astype(str)
    fortable.loc[fortable['yearGI3_min'] == fortable['yearGI3_max'], 'yearsGI3'] = fortable['yearGI3_min'].astype(int).astype(str)


    # adjust some dtypes
    fortable['count_glaciers'] = fortable['count_glaciers'].astype(int)
    fortable['lostglaciers'] = fortable['lostglaciers'].astype(int)

    # get total values for study region:
    # fortable.loc['total2',:] = total.values
    fortable.loc['total2','loss_km'] = ((GI_merge['area_GI5'].sum()-GI_merge['area_GI3'].sum())*1e-6).astype(float).round(decimals=4)
    fortable.loc['total2','perc_loss'] = (100*(GI_merge['area_GI5'].sum()-GI_merge['area_GI3'].sum())/GI_merge['area_GI3'].sum()).astype(float).round(decimals=2)
    fortable.loc['total2','loss_rate'] = GI_merge['loss_rate'].median().astype(float).round(decimals=1)
    fortable.loc['total2','count_glaciersGI3'] = len(GI3['id'].unique())
    fortable.loc['total2','count_glaciers'] = len(GI5['id'].unique())
    fortable.loc['total2','lostglaciers'] = len(GI3['id'].unique())-len(GI5['id'].unique())

    # total region-wide uncertainty:
    # use this for root sum of squares:
    # fortable.loc['total2', 'unc_abs'] = (np.sqrt((GI_merge['unc_abs']**2).sum())*1e-6).astype(float).round(decimals=3)
    # use this for simple sum:
    fortable.loc['total2', 'unc_abs'] = (GI_merge['unc_abs_GI5'].sum()*1e-6).astype(float).round(decimals=3)

    # some extras:
    fortable.loc['total2', 'perc_of_total'] = total['perc_of_total']
    fortable.loc['total2', 'yearsGI3'] = '2004-2012'
    fortable.loc['total2', 'yearsGI5'] = '2021-2023'
    fortable.loc['total2', 'area_GI5_km2'] = total['area_GI5_km2'].astype(float).round(decimals=3)

    # format for output:
    fortable['arkm_str'] = fortable['area_GI5_km2'].astype(str)+'±'+fortable['unc_abs_GI5'].astype(str)
    fortable['losskm_str'] = fortable['loss_km'].astype(str)+'±'+fortable['unc_change'].astype(str)


    fortable['total2', 'losskm_str'] = fortable.loc['total2','loss_km'].astype(str)+'±'+total['unc_change'].astype(str)

    # keep only required columns:
    fortable = fortable[['arkm_str', 'perc_of_total', 'losskm_str', 'perc_loss', 'loss_rate', 'count_glaciers', 'lostglaciers', 'yearsGI3', 'yearsGI5']]

    print('uniqueID GI5', len(GI5['id'].unique()))
    print('uniqueID GI3', len(GI3['id'].unique()))

    fortable.to_csv('out/summary_area_changesGI3GI5.csv')


# print values area and number of glaciers per year of data coverage
def datayears(GI5):
    
    arbyyear = GI5[['id', 'year', 'area']].groupby('year').sum()
    totalar = GI5['area'].sum()
    percar = 100*arbyyear['area']/totalar

    countbyyear = GI5[['id', 'year']].groupby('year').count()
    totalcount = GI5['id'].count()
    percount = 100*countbyyear['id']/totalcount

    df = pd.DataFrame(columns=['area', 'area_perc', 'count', 'count_perc'], index=percar.index)
    df['area'] = arbyyear.values
    df['area_perc'] = percar.values
    df['count'] = countbyyear.values
    df['count_perc'] = percount.values
    df.to_csv('out/data_years.csv')

    # print data type values: 
    countbydatatype = GI5[['id', 'data_type']].groupby('data_type').count()
    arbydatatype = GI5[['data_type', 'area']].groupby('data_type').sum()

    perdata_count = 100*countbydatatype['id']/totalcount
    perdata_ar = 100*arbydatatype['area']/totalar

    # uncomment to count by data type 2:
    countbydatatype2 = GI5[['id', 'data_type2']].groupby('data_type2').count()

    # make another DF for different data types. requires a manual check in the table to pick out
    # required types due to varying names for same data type
    df2 = pd.DataFrame(columns=['area', 'area_perc', 'count', 'count_perc'], index=countbydatatype.index)
    df2['area'] = arbydatatype.values
    df2['area_perc'] = perdata_ar.values
    df2['count'] = countbydatatype.values
    df2['count_perc'] = perdata_count.values
    df2.to_csv('out/datatype_area_count.csv')


# compute loss rates across the GI and write to output tables. Also find lost glaciers GI1GI2
# and GI2GI3 and export tables with positive change rates
def lossrates(GI_merge, allregs, GILIA, GI1, GI2, GI3, GI5, goneglaciers):
    # changed HEF to have ID 2125 in all GI and include Toteis in outline. deleted "Toteis" / HEF without Toteis variations.
    GI1['area_GI1'] = GI1.geometry.area
    GI2['area_GI2'] = GI2.geometry.area

    # get time periods between GI3 and GI5: 
    GI_merge['yearsGI3GI5'] = GI_merge['year_GI3'].astype(str) +'-'+ GI_merge['year_GI5'].astype(str)
    #print(GI_merge['yearsGI3GI5'].unique())

    # add GI3 to GI5 change rate in KM2
    GI_merge['KMrate'] = 1e-6*(GI_merge['area_GI3']-GI_merge['area_GI5'])/(GI_merge['year_GI3']-GI_merge['year_GI5'])
    
    # make a df with loss rates between GI3 and intermediate data (r1) and int. dat. and GI5 (r2)
    all_mrg = allregs[['id', 'name', 'year_mid', 'area_mid']].merge(GI_merge, left_on='id', right_on='id', how='outer')
    # GI3 - intermediate
    all_mrg['r1'] = np.nan
    all_mrg['KMr1'] = np.nan
    # get change rate GI3-intermediate where the intermediate year is not null (i.e., where intermediate data exists)
    all_mrg.loc[~all_mrg['year_mid'].isnull(), 'r1'] = 100*((all_mrg['area_GI3']-all_mrg['area_mid'])/(all_mrg['year_GI3']-all_mrg['year_mid']) / all_mrg['area_GI3'])
    all_mrg.loc[~all_mrg['year_mid'].isnull(), 'KMr1'] = 1e-6*(all_mrg['area_GI3']-all_mrg['area_mid'])/(all_mrg['year_GI3']-all_mrg['year_mid']) 
    
    # intermdiate - GI5
    # get change rate GIintermediate-GI5 where the intermediate year is not null (i.e., where intermediate data exists)
    all_mrg['r2'] = np.nan
    all_mrg['KMr2'] = np.nan
    all_mrg.loc[~all_mrg['year_mid'].isnull(), 'r2'] = 100*((all_mrg['area_mid']-all_mrg['area_GI5'])/(all_mrg['year_mid']-all_mrg['year_GI5']) /  all_mrg['area_mid'])
    all_mrg.loc[~all_mrg['year_mid'].isnull(), 'KMr2'] = 1e-6*(all_mrg['area_mid']-all_mrg['area_GI5'])/(all_mrg['year_mid']-all_mrg['year_GI5']) 
    
    # write years as strings:
    all_mrg['years_r1'] = all_mrg['year_GI3'].astype(str) +'-'+ all_mrg['year_mid'].astype(str)
    all_mrg['years_r2'] = all_mrg['year_mid'].astype(str) +'-'+ all_mrg['year_GI5'].astype(str)

    # GI2 - GI3
    # make a df with loss rates between GI2 and GI3:
    GI2GI3 = GI2.merge(GI_merge[['id', 'area_GI3', 'year_GI3']], left_on='id', right_on='id', how='outer')
    # set area of "lost" glaciers to zero, i.e., fill empty values in GI3 area column with zero
    GI2GI3['area_GI3'] = GI2GI3['area_GI3'].fillna(0)
    # compute change rates
    GI2GI3['rate'] = 100*((GI2GI3['area_GI2']-GI2GI3['area_GI3'])/(GI2GI3['year_GI2']-GI2GI3['year_GI3']) / GI2GI3['area_GI2'])
    GI2GI3['KMrate'] = 1e-6*(GI2GI3['area_GI2']-GI2GI3['area_GI3'])/(GI2GI3['year_GI2']-GI2GI3['year_GI3'])
    # extract glaciers lost between GI2 ad GI3 and write to geojson file:
    GI2GI3lost = GI2GI3.loc[GI2GI3['area_GI3'] == 0]
    GI2GI3lost.to_file('out/GI2GI3lost.geojson')
    GI2GI3lostCntrs = GI2GI3lost.copy()
    GI2GI3lostCntrs.geometry = GI2GI3lost.centroid.to_crs(epsg=4326).geometry
    GI2GI3lostCntrs.to_file('out/GI2GI3lost_centroids.geojson')

    # GI1 - GI2
    # make a df with loss rates between GI1 and GI2:
    GI1GI2_1 = GI1.merge(GI2[['id', 'area_GI2', 'year_GI2']], left_on='id', right_on='id', how='outer')#, suffixes=('_GI1', '_GI2'))
    # keep only regions that are present in both GI1 and GI2: 
    regsGI1 = GI1.region.unique()
    GI1GI2 = GI1GI2_1.loc[GI1GI2_1['region'].isin(regsGI1)]
    GI1GI2['area_GI2'] = GI1GI2['area_GI2'].fillna(0)
    # compute change rates
    GI1GI2['rate'] = 100*((GI1GI2['area_GI1']-GI1GI2['area_GI2'])/(GI1GI2['year_GI1']-GI1GI2['year_GI2']) / GI1GI2['area_GI1'])
    GI1GI2['KMrate'] = 1e-6*(GI1GI2['area_GI1']-GI1GI2['area_GI2'])/(GI1GI2['year_GI1']-GI1GI2['year_GI2'])
    ## check lost gl between GI1 and GI2.
    GI1GI2lost = GI1GI2.loc[GI1GI2['area_GI2'] == 0]

    if len(GI1GI2lost.index)>0:
        GI1GI2lost.to_file('out/GI1GI2lost.geojson')
        GI1GI2lostCntrs = GI1GI2lost.copy()
        GI1GI2lostCntrs.geometry = GI1GI2lost.centroid.to_crs(epsg=4326).geometry
        GI1GI2lostCntrs.to_file('out/GI1GI2lost_centroids.geojson')

    # prepare a df to make an output table with loss rates in different periods:
    # homogenize names a bit to loop:
    GI3GI5 = GI_merge.rename(columns={'loss_rate': 'rate'})

    GI3Mid = all_mrg.loc[~all_mrg['r1'].isnull()]
    GI3Mid.drop(columns='KMrate', inplace=True)
    GI3Mid = GI3Mid.rename(columns={'r1': 'rate', 'KMr1': 'KMrate'})
    MidGI5 = all_mrg.loc[~all_mrg['r2'].isnull()]
    MidGI5.drop(columns='KMrate', inplace=True)
    MidGI5 = MidGI5.rename(columns={'r2': 'rate', 'KMr2': 'KMrate'})

    dfRates = pd.DataFrame(columns=['GI1-GI2', 'GI2-GI3', 'GI3-GI5', 'GI3-mid', 'mid-GI5'], index=['mean', 'median'])

    for data, label in zip([GI1GI2, GI2GI3, GI3GI5, GI3Mid, MidGI5], ['GI1-GI2', 'GI2-GI3', 'GI3-GI5', 'GI3-mid', 'mid-GI5']):
        dfRates.loc['mean', label] = data['rate'].mean()
        dfRates.loc['median', label] = data['rate'].median()
        dfRates.loc['SD', label] = data['rate'].std()
        dfRates.loc['medianKM', label] = data['KMrate'].median()
        dfRates.loc['sumKM', label] = data['KMrate'].sum()
        dfRates.loc['countpos', label] = data.loc[data['rate'] > 0, 'rate'].count()
        dfRates.loc['countAll', label] = len(data['id'].unique())
 
    dfRates.loc['countpos_ar_med', 'GI1-GI2'] = GI1GI2.loc[GI1GI2['rate'] > 0, 'area_GI2'].median()*1e-6
    dfRates.loc['countpos_ar_med', 'GI2-GI3'] = GI2GI3.loc[GI2GI3['rate'] > 0, 'area_GI3'].median()*1e-6
    dfRates.loc['countpos_ar_med', 'GI3-GI5'] = GI3GI5.loc[GI3GI5['rate'] > 0, 'area_GI5'].median()*1e-6
    dfRates.loc['countpos_ar_med', 'GI3-mid'] = GI3Mid.loc[GI3Mid['rate'] > 0, 'area_mid'].median()*1e-6
    dfRates.loc['countpos_ar_med', 'mid-GI5'] = MidGI5.loc[MidGI5['rate'] > 0, 'area_GI5'].median()*1e-6

    dfRates.loc['countpos_ar_sum', 'GI1-GI2'] = GI1GI2.loc[GI1GI2['rate'] > 0, 'area_GI2'].sum()*1e-6
    dfRates.loc['countpos_ar_sum', 'GI2-GI3'] = GI2GI3.loc[GI2GI3['rate'] > 0, 'area_GI3'].sum()*1e-6
    dfRates.loc['countpos_ar_sum', 'GI3-GI5'] = GI3GI5.loc[GI3GI5['rate'] > 0, 'area_GI5'].sum()*1e-6
    dfRates.loc['countpos_ar_sum', 'GI3-mid'] = GI3Mid.loc[GI3Mid['rate'] > 0, 'area_mid'].sum()*1e-6
    dfRates.loc['countpos_ar_sum', 'mid-GI5'] = MidGI5.loc[MidGI5['rate'] > 0, 'area_GI5'].sum()*1e-6

    # write output tables: 
    dfRates = dfRates.astype(float).round(decimals=3)
    dfRates.to_csv('out/rates_1.csv')
    dfRates.T[['median', 'medianKM', 'sumKM', 'countpos', 'countAll']].to_csv('out/rates_T.csv')

    # to check positive change rates, also write those to output tables    
    GI2GI3.loc[GI2GI3['rate']>0].sort_values(by='rate', ascending=False).to_csv('out/pos/poschange_GI2GI3.csv')
    GI1GI2.loc[GI1GI2['rate']>0].sort_values(by='rate', ascending=False).to_csv('out/pos/poschange_GI1GI2.csv')
    GI3GI5.loc[GI3GI5['rate']>0].sort_values(by='rate', ascending=False).to_csv('out/pos/poschange_GI3GI5.csv')
    MidGI5.loc[(MidGI5['rate']>0) & (MidGI5['region_GI5']=='Oetztaler_Alpen')].to_csv('out/pos/poschange_otz_2017_2023.csv')
    GI2GI3.loc[GI2GI3['rate']>0].sort_values(by='rate', ascending=False).to_file('out/pos/poschange_GI2GI3.geojson')
    GI1GI2.loc[GI1GI2['rate']>0].sort_values(by='rate', ascending=False).to_file('out/pos/poschange_GI1GI2.geojson')
    #GI3GI5.loc[GI3GI5['rate']>0].sort_values(by='rate', ascending=False).to_file('out/pos/poschange_GI3GI5.geojson')
    #MidGI5.loc[(MidGI5['rate']>0) & (MidGI5['region_GI5']=='Oetztaler_Alpen')].to_file('out/pos/poschange_otz_2017_2023.geojson')

    return(GI1GI2, GI2GI3, all_mrg)


# alternative buffer uncertainties, writes to csv (does not return in the df)
def get_bufferUnc(ol):
    ol['area'] = ol.geometry.area
    ol['area_min2'] = ol.geometry.buffer(-2).area
    ol['area_pls2'] = ol.geometry.buffer(2).area

    ol['area_min20'] = ol.geometry.buffer(-20).area
    ol['area_pls20'] = ol.geometry.buffer(20).area

    ol['area_min40'] = ol.geometry.buffer(-40).area
    ol['area_pls40'] = ol.geometry.buffer(40).area

    highdebr = ol.loc[ol.debris > 1]
    lowdebr = ol.loc[ol.debris <= 1]

    df_buffer = pd.DataFrame(columns=['nobuff', '-2m', '+2m'], index=['lowdebr', 'highdebr', 'all'])
    for ix, gls in zip(['all', 'highdebr', 'lowdebr'], [ol, highdebr, lowdebr]):
        df_buffer.loc[ix, 'nobuff'] = gls['area'].sum()*1e-6
        df_buffer.loc[ix, '-2m'] = gls['area_min2'].sum()*1e-6
        df_buffer.loc[ix, '+2m'] = gls['area_pls2'].sum()*1e-6
        df_buffer.loc[ix, '-20m'] = gls['area_min20'].sum()*1e-6
        df_buffer.loc[ix, '+20m'] = gls['area_pls20'].sum()*1e-6
        df_buffer.loc[ix, '-40m'] = gls['area_min40'].sum()*1e-6
        df_buffer.loc[ix, '+40m'] = gls['area_pls40'].sum()*1e-6

    df_buffer.T.to_csv('out/buffertable.csv')
    # return(ol)


# makes a boxplot figure of the category scores and writes data to table
def fig_box(GI5):
    GI5['area_km'] = GI5.geometry.area *1e-6
    tocheck = GI5.loc[(GI5.area_km>0.25) & (GI5.outline_qf==3)]

    # debris qf
    nodeb = GI5.loc[GI5.debris==0]
    somedeb = GI5.loc[GI5.debris==1]
    mostlydeb = GI5.loc[GI5.debris==2]
    fulldeb = GI5.loc[GI5.debris==3]
    unclear = GI5.loc[GI5.debris==4]

    # outline qf:
    good = GI5.loc[GI5.outline_qf==0]
    medium = GI5.loc[GI5.outline_qf==1]
    poor = GI5.loc[GI5.outline_qf==2]
    vanishing = GI5.loc[GI5.outline_qf==3]

    # crevasses:
    yes = GI5.loc[GI5.crevs==1]
    no = GI5.loc[GI5.crevs==0]
    unclear = GI5.loc[GI5.crevs==2]

    fig, ax = plt.subplots(1, 3, figsize=(9,6), sharey=True)
    ax = ax.flatten()

    bplot_qf = ax[0].boxplot([good['area_km'].values, medium['area_km'].values, 
                        poor['area_km'].values, vanishing['area_km'].values],
                           tick_labels=['good, n='+str(len(good['area_km'].values)), 'medium, n='+str(len(medium['area_km'].values)),
                           'poor, n='+str(len(poor['area_km'].values)), 'vanishing, n='+str(len(vanishing['area_km'].values))])#, 'unclear'])

    ax[0].set_ylabel('Area [km$^2$]')
    ax[0].set_title('Outline quality')
    ax[0].set_ylim(0, 1.5)

    bplot_deb = ax[1].boxplot([nodeb['area_km'].values, somedeb['area_km'].values, 
                        mostlydeb['area_km'].values, fulldeb['area_km'].values],
                           tick_labels=['no debris, n='+str(len(nodeb['area_km'].values)), 'some debris, n='+str(len(somedeb['area_km'].values)),
                           'mostly debris, n='+str(len(mostlydeb['area_km'].values)), 'full debris, n='+str(len(fulldeb['area_km'].values))])#, 'unclear'])

    

    bplot_crev = ax[2].boxplot([yes['area_km'].values, no['area_km'].values, 
                        unclear['area_km'].values],
                           tick_labels=['visible crevasses, n='+str(len(yes['area_km'].values)), 'no vis. crev., n='+str(len(no['area_km'].values)),
                           'unsure, n='+str(len(unclear['area_km'].values))])

    ax[1].set_title('Debris categories')
    ax[2].set_title('Crevasses')

    for a in ax: 
        for label in a.get_xticklabels():
            label.set_rotation(45)
            a.grid('y')

    fig.savefig('figures/boxplots_Flags_Area.png', bbox_inches='tight', dpi=200)

    dfMedian = pd.DataFrame(columns=['all', 'good', 'medium', 'poor', 'almost gone', 'nodebris', 'somedebris',
                                 'muchdebris', 'fulldebris', 'crevasses', 'nocrevs'])
    subsets = [GI5, good, medium, poor, vanishing, nodeb, somedeb, mostlydeb, fulldeb, yes, no]

    for i, s in enumerate(dfMedian.columns):
        dfMedian.loc['count', s] = subsets[i]['area'].count()
        dfMedian.loc['medianElev', s] = subsets[i]['median_elev'].mean()
        dfMedian.loc['medianAr', s] = subsets[i]['area'].median()*1e-6
        dfMedian.loc['minimumAr', s] = subsets[i]['area'].min()*1e-6
        dfMedian.loc['maximumAr', s] = subsets[i]['area'].max()*1e-6
        dfMedian.loc['perc_totalAR', s] = 100*subsets[i]['area'].sum() /GI5['area'].sum()

    print(dfMedian.T)

    forTab = dfMedian.T
    forTab[['medianAr', 'minimumAr', 'maximumAr']] = forTab[['medianAr', 'minimumAr', 'maximumAr']].astype(float).round(decimals=4)
    forTab['medianElev'] = forTab['medianElev'].astype(float).round(decimals=0).astype(int)
    forTab['perc_totalAR'] = forTab['perc_totalAR'].astype(float).round(decimals=2)

    forTab.to_csv('out/table_categories.csv')

    print(GI5[['id', 'name', 'region', 'analyst', 'area_km', 'img_qf']].loc[GI5.img_qf==3])
    vanishing.to_file('out/vanishing_qf3.geojson')

    rogi = GI5.loc[GI5['ROGI']==2]
    rogi_1 = GI5.loc[GI5['ROGI']==1]
    rogi.to_file('out/rogi_qf2.geojson')
    rogi_1.to_file('out/rogi_qf1.geojson')


# prepare data frame for stacked area/bar charts. Area since LIA
def sinceLIA(GILIA, GI1, GI2, GI3, GI5):
    yrs = [1850, 1969, 1998, 2006, 2023]
    df = pd.DataFrame(index=[GI5.region.unique()], columns=[00])
    df[00] = 0
    df['region']=df.index
    for i, gi in enumerate([GILIA, GI1, GI2, GI3, GI5]):
        gi['area'] = gi.geometry.area
        gb = gi[['region', 'area']].groupby('region').sum()
        gb.rename(columns={'area': yrs[i]}, inplace=True)
        df = df.merge(gb, left_on='region', right_on='region', how='outer')

    # remove empty columns (only keep regions present in all GI)
    df = df.loc[df[0] != 0]
    df.drop(columns=[0], inplace=True)

    df = df.sort_values(by=1850)
    df.dropna(axis=0, inplace=True)

    df['1850prc'] = 100*df[1850]/df[1850].sum()
    df['1969prc'] = 100*df[1969]/df[1850].sum()
    df['1998prc'] = 100*df[1998]/df[1850].sum()
    df['2006prc'] = 100*df[2006]/df[1850].sum()
    df['2023prc'] = 100*df[2023]/df[1850].sum()

    df['1969prc_relGI1'] = 100*df[1969]/df[1969].sum()
    df['1998prc_relGI1'] = 100*df[1998]/df[1969].sum()
    df['2006prc_relGI1'] = 100*df[2006]/df[1969].sum()
    df['2023prc_relGI1'] = 100*df[2023]/df[1969].sum()

    df_abs = df[['region', 1850, 1969, 1998, 2006, 2023]]
    df_prc = df[['region', '1850prc', '1969prc', '1998prc', '2006prc', '2023prc']]
    df_prc69 = df[['region', '1969prc_relGI1', '1998prc_relGI1', '2006prc_relGI1', '2023prc_relGI1']]
    df_abs.index = df_abs['region']
    df_abs.drop(columns='region', inplace=True)
    df_prc.index = df_prc['region']
    df_prc69.index = df_prc69['region']
    df_prc.drop(columns='region', inplace=True)
    df_prc69.drop(columns='region', inplace=True)
    df_abs = (df_abs.astype(float)*1e-6).sort_values(by=1850, ascending=True)
    df_prc = df_prc.astype(float).sort_values(by='1850prc', ascending=False)
    df_prc69 = df_prc69.astype(float).sort_values(by='1969prc_relGI1', ascending=False)
    return(df_prc, df_abs)

## ------ misc -----------

# get area weighted year from "year" column
def arWeightYear(inv):
    inv['year'] = inv['year'].astype(int)
    df = pd.DataFrame(columns=inv['year'].astype(int).unique())
    arTot = inv.geometry.area.sum()

    for yr in inv['year'].astype(int).unique():
        arPrc = inv.loc[inv['year']==yr].geometry.area.sum() / arTot
        df.loc['val', yr] = arPrc
    df = df.T
    df['calc'] = df.index*df.val
    wY = df['calc'].sum() / df['val'].sum()
    return (wY)



# DEM processing (hypsometry)
def getHyps(dem, sufx, GI3, GI5, arealost, what):
    with rasterio.open(dem, 'r') as src:
        GI3 = GI3.to_crs(src.crs)
        GI5 = GI5.to_crs(src.crs)
        arealost = arealost.to_crs(src.crs)

        clippedGI3, out_transformGI3 = rasterio.mask.mask(src, GI3.geometry, crop=True)
        clippedGI5, out_transformGI5 = rasterio.mask.mask(src, GI5.geometry, crop=True)
        clippedlost, out_transformlost = rasterio.mask.mask(src, arealost.geometry, crop=True)
    
    clippedGI3[clippedGI3 < 0] = np.nan
    clippedGI5[clippedGI5 < 0] = np.nan
    clippedlost[clippedlost < 0] = np.nan

    print(what)
    print ('median '+what+' GI5 all: ', np.nanmedian(clippedGI5))
    print('20 and 80 quant, GI5:', np.nanquantile(clippedGI5, [0.20, 0.80]))
    print ('median '+what+' GI3 all: ', np.nanmedian(clippedGI3))
    # print ('median elev. Gone Gl. all: ',np.nanmedian(clippedGone))
    print ('median '+what+' area lost: ',np.nanmedian(clippedlost))

    if what == 'elevation':
        levels = np.arange(1800, 3900, 50)
    if what == 'aspect':
        levels = np.arange(0, 370, 10)
    countsGI3, binsGI3 = np.histogram(clippedGI3.flatten(), levels)
    countsGI5, binsGI5 = np.histogram(clippedGI5.flatten(), levels)
    countsLost, binsLost = np.histogram(clippedlost.flatten(), levels)

    df_area_elevation = pd.DataFrame(index=levels[:-1])
    df_area_elevation['GI3'] = countsGI3
    df_area_elevation['GI5'] = countsGI5
    df_area_elevation['lostAR'] = countsLost

    df_area_elevation['lossprc'] = 100*df_area_elevation['lostAR']/df_area_elevation['lostAR'].sum()
    df_area_elevation.to_csv('out/df_area_'+what+sufx+'.csv')


def getHypsRegions(dem, GI5):
    # get median elevation of subregions - calls "getHyps" functions above
    dfReg = pd.DataFrame(index=GI5.region.unique(), columns=['medianElev'])
    for r in GI5.region.unique():
        # g3 = GI3.loc[GI3.region == r]
        g5 = GI5.loc[GI5.region == r]
        g5 = g5.dissolve()

        # # make DF with mediean elev per regions!
        print(r)
        with rasterio.open(dem, 'r') as src:
            clippedg5, out_transformg5 = rasterio.mask.mask(src, g5.geometry, crop=True)

        clippedg5[clippedg5 < 0] = np.nan
        dfReg.loc[r, 'medianElev'] = np.nanmedian(clippedg5)
        dfReg.loc[r, 'Areakm'] = g5.geometry.area.sum()*1e-6
        dfReg.loc[r, 'Lat'] = g5.geometry.centroid.to_crs(epsg=4236).y.values[0]
        dfReg.loc[r, 'Lon'] = g5.geometry.centroid.to_crs(epsg=4236).x.values[0]

    dfReg.to_csv('out/regions_medianElevation.csv')



#root sum of the squares
def sumsquare(ar):
    val = np.sqrt((ar**2).sum())
    return(val)






# def procVanishing(goneglaciers):
#     goneglaciers.index = goneglaciers['id']
#     goneglaciers = goneglaciers.sort_values(by='area_GI3', ascending=False)
#     goneglaciers['area_km'] = goneglaciers['area_GI3']*1e-6
#     print(goneglaciers.head(20))
#     print(goneglaciers['area_GI3'].max()*1e-6)
#     ixmax = goneglaciers['area_GI3'].idxmax()
#     print(goneglaciers.loc[goneglaciers.index == ixmax])

#     print(goneglaciers['min_elev'].min())
#     print(goneglaciers['max_elev'].max())


#     bins = [0, 0.001, 0.01, 0.05, 0.1, 0.5]
#     # bins = [0, 0.5, 15]
#     goneglaciers['binned'] = pd.cut(goneglaciers['area_km'], bins)

#     print(goneglaciers[['area_GI3', 'binned']].groupby('binned').count())
#     print(goneglaciers['area_km'].median())

#     return()




