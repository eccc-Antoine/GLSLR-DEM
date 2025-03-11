import geopandas as gpd
import rasterio
import pandas as pd
import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# todo: modifie les paths comme tu as besoin selon tes données..

#file_shp = r"F:\DEM_GLAMM\accuracy\donnees_khalid_1per_build.shp"
file_shp = r"C:\Users\MarandaA\Downloads\Buildings\Buildings.shp"
tile_shp = r"F:\DEM_GLAMM\DEM_CREATION_FINAL\GLAM_ISEE_TILES\Tuile_final_w_conversions.shp"
dem_folder = r'F:\DEM_GLAMM\DEM_CREATION_FINAL\results'

gdf_cosine = gpd.read_file(file_shp)
gdf_tile = gpd.read_file(tile_shp)

crs = 32618

# projection des couches dans le même systeme
gdf_tile = gdf_tile.to_crs(crs)
gdf_cosine = gdf_cosine.to_crs(crs)


print(len(gdf_cosine))


# jointure spatiale
gdf_cosine = gpd.sjoin(gdf_cosine, gdf_tile)

print(len(gdf_cosine))

# pour chaque tuile, on va chercher la tuile du DEM.

list_gdf_tile = []

for tile, gdf_cosine_tile in gdf_cosine.groupby('tile'):

    # IGLD85 - CGVD2013 = conversion
    # IGLD85 = conversion + CGVD2013

    print(tile, gdf_cosine_tile)

    # # convertir les valeurs de ground truth en IGLD85
    corr_value_cgvd2013 = gdf_cosine_tile['H85-H13'].values[0]      # première valeur de la colonne H85-H13
    # corr_value_cgvd28 = gdf_cosine_tile['H85-H28'].values[0]        # première valeur de la colonne H85-H28
    #
    # # nouvelle colonne créée pour tout convertir en IGLD85
    #gdf_cosine_tile['Z_GROUND_IGLD85'] = np.nan
    #
    # # condition pour sélectionner les points 2013
    # condition_2013 = (gdf_cosine_tile['DATUM'] == 'CGVD2013')
    #
    # # condition pour sélectionner les points 28
    # condition_28 = (gdf_cosine_tile['DATUM'] == 'CGVD28')
    #
    # # valeurs d'élévation ...
    # z_cgvd2013 = gdf_cosine_tile.loc[condition_2013, 'ELEVATION']
    # z_cgvd28 = gdf_cosine_tile.loc[condition_28, 'ELEVATION']
    #
    # # applique la correction
    # gdf_cosine_tile.loc[condition_2013, 'Z_GROUND_IGLD85'] = z_cgvd2013 + corr_value_cgvd2013
    # gdf_cosine_tile.loc[condition_28, 'Z_GROUND_IGLD85'] = z_cgvd28 + corr_value_cgvd28

    gdf_cosine_tile['Z_GROUND_IGLD85'] = gdf_cosine_tile['Elev']+gdf_cosine_tile['H85-H13']

    raster_file = os.path.join(dem_folder, *[f"{tile}_V4_3", f"{tile}_1m_DEM_idw_filtered.tif"])

    with rasterio.open(raster_file) as src:
        # Reproject points to raster CRS if needed
        band = src.read(1)

        # print(gdf_cosine_tile.geometry)
        # print(len(gdf_cosine_tile.geometry))
        # quit()
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

mean=gdf_cosine_tile_merged['DEM_a_err'].mean()

print(mean)

rmse = np.sqrt(np.mean(gdf_cosine_tile_merged['DEM_a_err'] ** 2))
print(rmse)

rmse2 = np.sqrt(np.mean(gdf_cosine_tile_merged['DEM_error'] ** 2))
print(rmse2)


# TODO: RENOMME LE FICHIER OUTPUT COMME TU LE VEUX.. tu peux aussi écrire un fichier csv si tu préfères... ensuite calculer les erreurs et statistiques...
#output_path = './output_file.shp'

output_path = r"F:\DEM_GLAMM\accuracy\donnees_khalid_1per_build_w_dem_values.shp"

gdf_cosine_tile_merged.to_file(output_path)

quit()