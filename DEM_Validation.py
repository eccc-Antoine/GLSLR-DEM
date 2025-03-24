import geopandas as gpd
import rasterio
import pandas as pd
import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# todo: modifie les paths comme tu as besoin selon tes données..

''' 
Script to compare elevation between DEM values of ground truth points
Make sure ground truth elevation value are in IGLD85. If not, you need to make elevation conversion
(conversion values with CGVD28 and CGVD2013 can be found in ISEE tiles shapefile) 

Authors: Antoine Maranda (antoine.maranda@ec.gc.ca) and Dominic Thériault (dominic.theriault@ec.gc.ca)
Environment and Climate Change Canada, National Hydrologic Services, Hydrodynamic and Ecohydraulic Section
'''

# Ground truth points shapefile
file_shp = r"C:\Users\MarandaA\Downloads\Buildings\Buildings.shp"
# DEM tiles
tile_shp = r"F:\DEM_GLAMM\DEM_CREATION_FINAL\GLAM_ISEE_TILES\Tuile_final_w_conversions.shp"
# DEM folder
dem_folder = r'F:\DEM_GLAMM\DEM_CREATION_FINAL\results'

gdf_cosine = gpd.read_file(file_shp)
gdf_tile = gpd.read_file(tile_shp)

crs = 32618
gdf_tile = gdf_tile.to_crs(crs)
gdf_cosine = gdf_cosine.to_crs(crs)

gdf_cosine = gpd.sjoin(gdf_cosine, gdf_tile)

list_gdf_tile = []

for tile, gdf_cosine_tile in gdf_cosine.groupby('tile'):

    # Make sure ground truth elevation value are in IGLD85, if not you need to make the converion
    gdf_cosine_tile['Z_GROUND_IGLD85'] = gdf_cosine_tile['Elev']

    raster_file = os.path.join(dem_folder, *[f"{tile}_V4_3", f"{tile}_1m_DEM_idw_filtered.tif"])

    with rasterio.open(raster_file) as src:
        # Reproject points to raster CRS if needed
        band = src.read(1)

        if gdf_cosine_tile.crs != src.crs:
            gdf_cosine_tile = gdf_cosine_tile.to_crs(src.crs)

        # Extract raster values at point locations
        coords = [(geom.x, geom.y) for geom in gdf_cosine_tile.geometry]
        values = list(src.sample(coords))

    # Add the raster values to the GeoDataFrame
    gdf_cosine_tile["Z_DEM_IGLD85"] = [val[0] if val else np.nan for val in values]

    list_gdf_tile.append(gdf_cosine_tile)

gdf_cosine_tile_merged= pd.concat(list_gdf_tile)

gdf_cosine_tile_merged['DEM_error']=gdf_cosine_tile_merged["Z_DEM_IGLD85"]-gdf_cosine_tile_merged['Z_GROUND_IGLD85']
gdf_cosine_tile_merged['DEM_a_err']=abs(gdf_cosine_tile_merged['DEM_error'])

Q1 = gdf_cosine_tile_merged['DEM_a_err'].quantile(0.25)
Q3 = gdf_cosine_tile_merged['DEM_a_err'].quantile(0.75)
IQR = Q3 - Q1

# Define outlier bounds
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Remove outliers
gdf_cosine_tile_merged = gdf_cosine_tile_merged[(gdf_cosine_tile_merged['DEM_a_err'] >= lower_bound) & (gdf_cosine_tile_merged['DEM_a_err'] <= upper_bound)]

sns.histplot(gdf_cosine_tile_merged['DEM_a_err'], bins=10, kde=True)

plt.show()

#compute absolute mean error
mean=gdf_cosine_tile_merged['DEM_a_err'].mean()
print(mean)

#compute root mean square error
rmse = np.sqrt(np.mean(gdf_cosine_tile_merged['DEM_a_err'] ** 2))
print(rmse)

output_path = r"F:\DEM_GLAMM\accuracy\donnees_khalid_1per_build_w_dem_values.shp"

gdf_cosine_tile_merged.to_file(output_path)

quit()