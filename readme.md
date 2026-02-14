## scripts to process AGI files		




# process_GI5_step3.py		
- imports proc_helper_functions.py and helpers_plots.py
- loads pre-processed AGI5 files and compiles larger scale stats over all regions
- loads older GI (GI Lia, GI1, GI2, GI3) and regional intermediate GI (Salzburg, Vorarlberg, Stubai, Ötztal, Silvretta)
- calls helper functions to produce various output tables and figures


# proc_helper_functions.py			
- various helpers for "process_GI5_step3.py"		

# helpers_plots.py	
- functions to make plots




# process_inv_comparisons.py		
- load AGI2,3,4,5; RGI, C3s inventories	
- extract outlines for AT from RGI and C3s (centroid in Austria)	
- compute some comparisons and make a figure 	

# compare_oetztal.py		
- compare two alternative sets of outlines for a subset of the Ötztal Alps (mapped with local knowledge and without)	
- produces some statistics (output to csv) and makes a figure	