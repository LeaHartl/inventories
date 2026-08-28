# Scripts to process AGI glacier outline files		
This repo contains various scripts that handle glacier outline files produced for the 5th Austrian glacier inventory (AGI5) and older inventory data sets. 	
AGI5 data publication: https://doi.org/10.1594/PANGAEA.991106
AGI5 paper (accepted preprint): https://doi.org/10.5194/egusphere-2026-1241

Other relevant data publications:
AGI1, 2, 3, and LIA (data publication series): https://doi.org/10.1594/PANGAEA.844988
AGI4: https://doi.org/10.1594/PANGAEA.887415 
Glacier inventory Austrian Silvretta 2017/2018: https://doi.org/10.1594/PANGAEA.936109
Glacier inventory Ötztal Alps 2017: https://doi.org/10.1594/PANGAEA.965798
Glacier inventory Stubai Alps 2017/2018: https://doi.org/10.1594/PANGAEA.965791
Area and volume change of glaciers in the Salzburg region, Austria: a new inventory (2008-2018) (dataset bundled publication): https://doi.org/10.1594/PANGAEA.984878
Vorarlberg (Austria) Glacier outlines, 2017-2020-2022-2023: https://doi.org/10.1594/PANGAEA.984116


### process_GI5_step3.py		
- loads pre-processed AGI5 files and compiles larger scale statistics over all AGI subregions
- loads older AGI (GI Lia, GI1, GI2, GI3) and regional intermediate GI (Salzburg, Vorarlberg, Stubai, Ötztal, Silvretta)

- imports and calls helper functions in **proc_helper_functions.py** and **helpers_plots.py** to produce output figures and tables

### process_inv_comparisons.py		
- loads AGI2,3,4,5; RGI, C3s inventories	
- extracts glacier outlines for AT from RGI and C3s by selecting all outlines with centroids in Austria.	
- compute some comparisons and makes a figure 	

### compare_oetztal.py		
- compare two alternative sets of outlines for a subset of the Ötztal Alps (mapped with local knowledge and without)	
- produces some statistics (output to csv) and makes a figure	

## plotting functions:		
### Fig_data_example.py 	
- makes a figure showing examples of different data types (hillshades, orthophotos)

## Fig_RR_example.py 	
- makes a figure showing Round Robin outlines of Seekarles Ferner	

## Fig_Skiresorts.py 	
- makes a figure showing examples of glacier coverings in ski resorts  	

## additional_plots.py	
- makes extra figures added during revision, loads "digitization uncertainty" outlines to compute statistics 