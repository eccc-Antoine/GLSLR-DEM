import os
import geopandas as gpd
import numpy as np


liste=os.listdir(fr'F:\DEM_GLAMM\DEM_CREATION_FINAL\results')

gdf_tile=gpd.read_file(fr"F:\DEM_GLAMM\DEM_CREATION_FINAL\GLAM_ISEE_TILES\Tuile_final_w_conversions.shp")

gdf_tile['tile_score']=np.nan

for l in liste:
    print(l)



    tile=l.split('_')[0]

    try:
        integer=int(tile)

        print(tile)

        path=os.path.join(fr'F:\DEM_GLAMM\DEM_CREATION_FINAL\results', l, f'datasets_extent_{tile}_details.shp')

        print(path)

        gdf=gpd.read_file(path)

        score=np.round(gdf['tile_score'][0], 2)

        print(score)

        gdf_tile.loc[gdf_tile['tile']==int(tile), 'tile_score']=score

    except:
        print(tile, 'NOT POSSIBLE')

print(gdf_tile['tile_score'])

gdf_tile.to_file(fr'F:\DEM_GLAMM\DEM_CREATION_FINAL\GLAM_ISEE_TILES\Tiles_w_scores.shp')

quit()