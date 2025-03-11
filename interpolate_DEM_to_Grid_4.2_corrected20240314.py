import os
import glob
import rasterio
import pandas as pd
import numpy as np
from scipy.spatial import cKDTree
import geopandas as gpd

def raster_to_XYZ_numpy(raster):
    """

    :param raster: fichier raster du DEM (format .tif)
    :return: une grille [x, y, z] et le CRS du raster.
    """

    with rasterio.open(raster) as src:
        crs = src.crs
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
        export = np.column_stack((newX, newY, image1))
        export=np.round(export, 3)

    return export, crs

def nearestNeighbor(source_array, dest_array, distance_upper_bound=30, k=1):
    """

    :param source_array:   grille ISEE (x, y, z=Id)
    :param dest_array:     grille DEM (x, y)
    :param distance_upper_bound:   distance maximale pour chercher les voisins (30m)
    :param k: nombre de voisin = 1
    :return: retourne le ID de la grille associé au PIXEL du Raster
    """
    x = source_array[:, 0]
    y = source_array[:, 1]
    z = source_array[:, 2]
    xys = np.c_[x, y]
    tree = cKDTree(xys)

    xi = dest_array[:, 0]
    yi = dest_array[:, 1]
    grid = np.c_[xi, yi]
    dist, idx = tree.query(grid, distance_upper_bound=distance_upper_bound, k=k)

    idx = np.where(idx==len(x), 0, idx)

    z_pred = z.flatten()[idx].reshape(idx.shape)
    # dans le cas où il ne trouve pas de voisin, il retourne l'indice = len(x).
    # on filtre donc les indice len(x) et on assigne la valeur de Z à NoData.
    z_pred = np.where(idx==len(x), np.nan, z_pred)

    return z_pred


if __name__ == '__main__':

    input_folder_grid = r'H:\Projets\GLAM\DATA_ISEE\Grille_ISEE\GLAM_ISEE_Tiles\regular_grid_10m'
    #output_folder_grid = r'H:\Projets\GLAM\DATA_ISEE\Grille_ISEE\GLAM_ISEE_Tiles\DEM_3.10_regular_grid_10m'
    
    #output_folder_grid = r'H:\Projets\GLAM\DATA_ISEE\Grille_ISEE\GLAM_ISEE_Tiles\DEM_4.2_regular_grid_10m_20240314'
    
    output_folder_grid = r'H:\Projets\GLAM\DATA_ISEE\Grille_ISEE\GLAM_ISEE_Tiles\test_AM'
    
    os.makedirs(output_folder_grid, exist_ok=True)

    #folder_dem = r'T:\DEM_CREATION_CLEAN\results'
    #folder_dem_43 = r'T:\DEM_CREATION_FINAL\results'

    #file_info_tile = r'T:\DEM_CREATION_CLEAN\GLAM_ISEE_Tiles\Tuile_final_w_conversions.shp'
    
    
    folder_dem = r'F:\DEM_GLAMM\DEM_CREATION_CLEAN\results'

    file_info_tile = r'F:\DEM_GLAMM\DEM_CREATION_CLEAN\GLAM_ISEE_Tiles\Tuile_final_w_conversions.shp'
    
    
    gdf_info_tile = gpd.read_file(file_info_tile)

    #list_tile = [249, 283, 261, 237]
    #list_tile = [456, 460, 464, 465, 471, 472, 473, 477, 478, 480, 481, 482, 487, 488, 489, 491]

    outfile_metadata = os.path.join(output_folder_grid, 'metadata_regular_grid_10m_20240314.csv')

    list_files = glob.glob(input_folder_grid+f'\*.csv')

    print(list_files)
    list_df_meta = []
    #for tile_file in list_files:
    
    tiles=[78, 104, 165]

    for tile in tiles:
        tile_file = glob.glob(os.path.join(input_folder_grid, f'{tile}*.csv'))[0]


        basename = os.path.basename(tile_file)
        print(basename)
        tile_id = int(basename.split('_')[0])

        gdf_tile = gdf_info_tile[gdf_info_tile['tile']==tile_id]
        utm = gdf_tile['UTM'].values[0]

        print(tile_id)

        if utm==19:
            crs=32619
        elif utm==18:
            crs=32618
        elif utm==17:
            crs=32617
        else:
            crs=''

        tile_folder = os.path.join(folder_dem, f'{tile_id}_V4_2')
        if os.path.isdir(tile_folder):

            if tile_id == 486:
                version = '4.3_on_demand'
                dem_file_corr = os.path.join(folder_dem_43, f'{tile_id}_on_demand', f'{tile_id}_10m_dem_wetland_corrected.tif')
                dem_file_nocorr = os.path.join(folder_dem_43, f'{tile_id}_on_demand', f'{tile_id}_10m_DEM_idw_filtered.tif')
                dem_file_masked = os.path.join(folder_dem_43, f'{tile_id}_on_demand', f'{tile_id}_10m_DEM_idw_filtered_masked.tif')
            else:
                version = '4.2'

                dem_file_corr = os.path.join(folder_dem, f'{tile_id}_V4_2', f'{tile_id}_10m_dem_wetland_corrected.tif')
                dem_file_nocorr = os.path.join(folder_dem, f'{tile_id}_V4_2', f'{tile_id}_10m_DEM_idw_filtered.tif')
                dem_file_masked = os.path.join(folder_dem, f'{tile_id}_V4_2', f'{tile_id}_10m_DEM_idw_filtered_masked.tif')

            output_folder_grid_csv = os.path.join(output_folder_grid, 'csv')
            os.makedirs(output_folder_grid_csv, exist_ok=True)

            output_folder_grid_feather = os.path.join(output_folder_grid, 'feather')
            os.makedirs(output_folder_grid_feather, exist_ok=True)

            output_tile_isee_csv = os.path.join(output_folder_grid_csv, basename)
            output_tile_isee_feather = os.path.join(output_folder_grid_feather, basename[:-4] + '.feather')

            list_dems = [dem_file_nocorr, dem_file_corr, dem_file_masked]
            list_outputcols = ['ZVAL', 'ZVAL_corr', 'Mask']

            df_grid = pd.read_csv(tile_file, sep=';', header=0)
            gdf_grid = gpd.GeoDataFrame(df_grid, geometry=gpd.points_from_xy(df_grid['XVAL'], df_grid['YVAL'], crs=crs))

            gdf_grid = gdf_grid.rename({'ID':'PT_ID'}, axis=1)
            print(gdf_grid.columns)
            gdf_grid = gdf_grid.to_crs(4326)

            gdf_grid['LAT'] = gdf_grid.geometry.y
            gdf_grid['LON'] = gdf_grid.geometry.x
            z_cols = []
            df_meta = pd.DataFrame({'TILE_ID': [tile_id], 'VERSION': [version]})

            for dem_file in list_dems:
                alias = list_outputcols[list_dems.index(dem_file)]

                if os.path.exists(dem_file):

                    z_cols.append(alias)
                    print(alias)
                    df_meta[f'{alias}_path'] = dem_file

                    with rasterio.open(dem_file) as src:
                        # Get the NoData value
                        nodata_value = src.nodata

                    source_array = gdf_grid[['XVAL', 'YVAL', 'PT_ID']].to_numpy()

                    dest_array, crs = raster_to_XYZ_numpy(dem_file)
                    pt_id_pred = nearestNeighbor(source_array, dest_array)      # retourne le ID de la grille ISEE associé au PIXEL
                    z = dest_array[:, 2]     # valeur Z de la grille
                    print(nodata_value)
                    if alias == 'Mask':
                        z = z == nodata_value
                    else:
                        z = np.where(z == nodata_value, np.nan, z)

                    df_z_pred = pd.DataFrame(data={'PT_ID': pt_id_pred, alias: z})   # on crée un dataframe associant ID et Z

                    gdf_grid = gdf_grid.merge(df_z_pred, on='PT_ID', how='left')     # on merge le ID et le Z avec la Grille ISEE

                    gdf_grid = gdf_grid.drop_duplicates(subset=['PT_ID'])


            list_df_meta.append(df_meta)

            gdf_grid = gdf_grid.reset_index()

            output_cols = ['PT_ID', 'XVAL', 'YVAL', 'LAT', 'LON'] + z_cols
            print(output_cols)
            #output_cols_final = [col for col in output_cols if col in df_grid.columns]

            gdf_grid = gdf_grid[output_cols].copy()


            mask = gdf_grid['Mask'].to_numpy()

            for col in z_cols[:-1]:
                mask = np.where(np.isnan(gdf_grid[col].to_numpy()), True, mask)

            gdf_grid['Mask'] = mask
            gdf_grid['Mask'] = gdf_grid['Mask'].fillna(True)

            #df_grid.to_csv(output_tile_isee_csv, sep=';', index=False)
            gdf_grid.to_feather(output_tile_isee_feather)

        df_meta_merged = pd.concat(list_df_meta)

        df_meta_merged.to_csv(outfile_metadata, sep=';', index=False)