import os
import pandas as pd
import numpy as np
import rasterio
import laspy

def raster_to_XYZ_numpy(raster):
    with rasterio.open(raster) as src:
        image= src.read()
        bands,rows,cols = np.shape(image)
        image1 = image.reshape (rows*cols,bands)
        height = image.shape[1]
        width = image.shape[2]
        cols, rows = np.meshgrid(np.arange(width), np.arange(height))
        xs, ys = rasterio.transform.xy(src.transform, rows, cols)
        x = np.array(xs)
        y = np.array(ys)
        newX = np.array(x.flatten())
        newY = np.array(y.flatten())
        export_fct = np.column_stack((newX, newY, image1))
        export_fct=np.round(export_fct, 3)
    return export_fct

raster=fr"P:\Freshwater\FightingIsland\DEM\AOI_large\new_bathy\DEM_w_new_bathy_IGLD85_w_GLPI.tif"

arr=raster_to_XYZ_numpy(raster)

header = laspy.LasHeader(point_format=3, version="1.2")  # LAS 1.6 uses formats 6-10
las = laspy.LasData(header)

# Assign XYZ coordinates
las.x = arr[:, 0]
las.y = arr[:, 1]
las.z = arr[:, 2]

# Save to a .las file
las.write(fr'P:\Freshwater\FightingIsland\DEM\AOI_large\new_bathy\DEM_w_new_bathy_IGLD85_w_GLPI.las')

quit()

df=pd.DataFrame(arr, columns=['X', 'Y', 'Z'])

print(df.head())

df.to_csv(fr'P:\Freshwater\FightingIsland\DEM\AOI_large\new_bathy\DEM_w_new_bathy_IGLD85_w_GLPI.csv', sep=';')

quit()

src=fr"F:\DEM_GLAMM\Dom\correction_MH\GroundTruthElevationPoints_GLAM_LKO_USL_WetlandCorrection_valuesExtracted.csv"

df=pd.read_csv(src, sep=';')

print(len(df['SITE'].unique()))

quit()


moi=fr'H:\Projets\GLAM\DATA_ISEE\Grille_ISEE\GLAM_ISEE_Tiles\DEM_4.2_regular_grid_10m_20240314\feather'

dom=fr'H:\Projets\GLAM\DATA_ISEE\Grille_ISEE\GLAM_ISEE_Tiles\test_AM\feather'

tiles=[78, 104, 165]

for t in tiles:
    print(t)
    path_moi=os.path.join(moi, f'{t}_regular_grid.feather')
    df_moi=pd.read_feather(path_moi)
    
    path_dom=os.path.join(dom, f'{t}_regular_grid.feather')
    df_dom=pd.read_feather(path_dom)
    
    if df_moi.equals(df_dom):
        print('no_prob')
    else:
        print('isshhh')
    
     
    dfs=[df_moi, df_dom]
     
    for df in dfs:
        print(len(df))
        print(list(df))
        print(len(df.loc[df['Mask']==True]))
        
quit()