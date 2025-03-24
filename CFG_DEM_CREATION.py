import os

# main directory
workdir=fr'F:\DEM_GLAMM\Git_DEM_GLAM'

# Version of DEM to be created
res_folder_name='V4_4'

# DEM tiles to compute
#tiles=list(range(78, 492))
tiles=[217]
tiles_to_remove=[160, 161, 138, 125, 85, 79, 80, 81, 333]
tiles=[x for x in tiles if x not in tiles_to_remove]

# Resolution of final DEM (limited to one value for both variables)
final_resolutions=[1]
resamp_res=10

# path to gdal (needs to be inslatted locally)
gdal_scripts=r'C:\Anaconda3\envs\geo_env2\Lib\site-packages\GDAL-3.2.1-py3.9-win-amd64.egg-info\scripts'
gdal_path=r'C:\Anaconda3\envs\geo_env2\Library\bin'

# Source data location
dataset_dir = fr'{workdir}\DATA'

# Datasets specification and priority
specs_file = os.path.join(workdir, f'list_main_datasets.csv')
specs_file_details = os.path.join(workdir, f'list_main_datasets_details.csv')

# Overlapping analysis between tile and datasets location
tile_overlap_dir= os.path.join(workdir, "dataset_tile_overlap")
# Results of previous tile and datasets analysis (to avoid unnecessary computation)
previous_version = os.path.join(workdir, "tiles_dataset.csv")

# ISEE tile files
tiles_file_overview=fr"{workdir}\GLAM_ISEE_Tiles\Tuile_final_w_conversions.shp"
tiles_file=os.path.join(workdir, 'GLAM_ISEE_Tiles', 'extent_tiles_GLAM.csv')

# Data gaps files are stored
nodata_folder=fr"{workdir}\data_gaps"

# Dump for intermediate products (needs a few Gb of storage space)
dump_folder = os.path.join(workdir, 'dump')
# Useful to set to "False" when debugging
clean_dump=True

# Area of Interest shapefile
AOI= os.path.join(workdir, 'AOI',  r"AOI_distance_5km_LKO_USL_SLR.shp")

# In LKO CD to IGLD85 conversion is fixed at 74.2, in the river there is a slope, hence conversion varies and is obtained from cd_file
cd_file = os.path.join(workdir, 'CD_IGLD85_modified.csv')
slope_tiles=list(range(1, 239))
flat_CD_conversion=74.2

# Tiles that we know have only bathymetric data, avoids unncessary computation
only_bathy_tiles=[220,221,231,232,233,235,241,242,243,244,245,246,247,253,254,255,256,257,258,259,263,264,265,266,267,268,270,274,275,276,277,278,279,286,287,288,289,290,291,298,299,300,301,302,310,311,312,313,314,315,322,323,324,325,326,327,335,336,337,338,339,340,341,347,348,349,350,351,352,353,358,359,360,361,362,363,364,367,368,369,370,371,372,376,377,378,379,380,381,384,385,386,387,388,389,392,393,394,395,396,397,401,402,403,404,405,408,409,410,411,412,415,416,417,418,419,423,424,425,426,427,431,432,433,434,435,442,443,444,445,446,452,453,454,455,460,461,462,463,467,468,469,470,474,475,476,480,481]

# Specify if want to write raw data in .csv format or datasets with NONNA already interpolated with gdal_grid, which are ready for interpolation in regular grid
write_raw_data=False
write_interpolated_data=False

# set if there is a blend or feathering at the juction of bathy and topo DEMs 
final_feathering=True

##Wetland correction algorithm parameters
# If we want to apply wetland correction algorithm
apply_correction=True
# Wetland extent file, indicate where to apply wetland correction algorithm
wetland_extent=fr"{workdir}\wetlands\wetlands_LKO_USL_SLR.shp"
dump_corr=os.path.join(workdir, 'dump_corr')
# tiles for which we want to apply correction (only available >=169
tiles_w_corr=list(range(169, 492))
path_img_sentinel2 = fr'{workdir}\correction_model\NDVI'
model_file = fr"{workdir}\correction_model\GLAM_LKO_DemCorrection_RF_model_gridsearch_v3_20230329.joblib"
dict_sentinel_years = {'4WMH':2018, '5LCM':2018, '6JSM':2018, 'CLOCA_extent':2018, 'GTA_extent':2016,
                   'Peterborough_extent':'2016-2017', 'Hamilton_Niagara_extent':2021, 'HALTON_extent':2018, '1HIE':2018, '2ACM':2018, '3SBM':2018, 'SCOOP_2013_H_extent':2016, 'SCOOP_2013_G_extent':2016, 'DRAPE_2014_G_extent':2016,
                   'DRAPE_2014_H_extent':2016, 'DRAPE_2014_I_extent':2016, 'DRAPE_2014_F_extent':2016, 'DRAPE_2014_D_extent':2016, 'SNC_extent':'2018-2019',
                    'LEAP_extent':2017, 'CE_M1-1__mosaic_post_treated_IGLD85':2016, 'CE_M1-2__mosaic_post_treated_IGLD85':2016,
                    'CE_M1-3__mosaic_post_treated_IGLD85':2016, 'CE_M2-A__mosaic_post_treated_IGLD85':2017, 'CE_M2-B__mosaic_post_treated_IGLD85':2018,
                     'CE_M2-C__mosaic_post_treated_IGLD85':2018, 'CE_M2-D__mosaic_post_treated_IGLD85':2018, 'CE_M2-E__mosaic_post_treated_IGLD85':2018,
                      'CE_M2-F':2021 }