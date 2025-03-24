
import pandas as pd
import geopandas as gpd
import os
import shapely
import shapely.vectorized as vect
from shapely.validation import make_valid
from shapely.geometry import Polygon, MultiPolygon, Point, MultiPoint, MultiLineString
from rasterio.plot import show
from rasterio.merge import merge
import rasterio as rio
import rasterio.mask
from rasterio.warp import reproject, Resampling, calculate_default_transform
from rasterio.transform import Affine
import fiona
from pyproj import transform, Proj
import pyproj
from pyproj import Transformer
import numpy as np
from scipy.spatial import cKDTree
import joblib
import shutil


''' 
Script to apply pre-trained ground filtering algorithm (cfg.model_file) on parts of dataset that falls within 
wetlands footprint shapefile (cfg.wetland_extent)
Applies seperately on each dataset of each tile
Applies only on dataset where "ApplyCorrection" is set to "True" in cfg.specs_file
Applies only if cfg.apply_correction is set to "True"
Uses Sentinel2 mosaics stored in cfg.path_img_sentinel2  

Ground filtering algorithm development: Dominic Thériault (dominic.theriault@ec.gc.ca) and Antoine Maranda (antoine.maranda@ec.gc.ca)
Integration in DEM creation workflow: Antoine Maranda (antoine.maranda@ec.gc.ca) and Dominic Thériault (dominic.theriault@ec.gc.ca)

Environment and Climate Change Canada, National Hydrologic Services, Hydrodynamic and Ecohydraulic Section
'''

def merging_rasters(list_tif_file, dst, extent, crs_tile, dump_folder):
    raster_to_mosiac=[]
    list_clipped_proj=[]
    count_raster=0
    for f in list_tif_file:
        count_raster+=1
        with rasterio.open(f) as src:
            crs=src.crs    
        if count_raster==1:
            crs_base=src.crs
        if crs!=crs_base:
            print('need to reproject raster')
            dist_fin=fr'{dump_folder}\{count_raster}.tif'
            dist=reproj_raster(f, crs_base, dist_fin)
            list_clipped_proj.append(dist)
        else:
            list_clipped_proj.append(f)            
    for p in list_clipped_proj:
        raster = rio.open(p)
        nodat=raster.nodata
        raster_to_mosiac.append(raster)  
    extent=extent.to_crs(crs_base)
    bounds=tuple(extent.bounds.values[0])
    mosaic, output = merge(raster_to_mosiac, bounds=bounds)
    output_meta = raster.meta.copy()
    output_meta.update(
                       {"driver": "GTiff",
                        "height": mosaic.shape[1],
                        "width": mosaic.shape[2],
                        "transform": output, 
                        "nodata":nodat})                                            
    with rio.open(dst, "w", **output_meta) as m:
        m.write(mosaic)

def clean_gdf(gdf):
    extent=gdf.explode(ignore_index=True, index_parts=True)
    extent.reset_index(drop=True)
    list_2_drop=[]
    for f in range(len(extent)):
        extent.geometry.iloc[f] = make_valid(extent.geometry.iloc[f])
        if str(extent.geometry.iloc[f]).split(' ')[0]!='POLYGON' and str(extent.geometry.iloc[f]).split(' ')[0]!='MULTIPOLYGON':
            list_2_drop.append(f)       
    extent=extent.drop(labels=list_2_drop, axis=0)
    return extent

def reproj_raster(raster, dst_crs, dist_fin):
    with rasterio.open(raster) as src:
        dst_transform, width, height = calculate_default_transform(
        src.crs,    # source CRS
        dst_crs,    # destination CRS
        src.width,    # column count
        src.height,  # row count
        *src.bounds,  # unpacks outer boundaries (left, bottom, right, top)
                    )    
        dst_kwargs = src.meta.copy()
        dst_kwargs.update(
            {
                "crs": dst_crs,
                "transform": dst_transform,
                "width": width,
                "height": height,
                "nodata": 0, 
            })
        with rasterio.open(dist_fin, "w", **dst_kwargs) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=dst_transform,
                    dst_crs=dst_crs,
                    resampling=Resampling.nearest,
                )
    return dist_fin

def clipping_raster(extent_file, reproj, dst):
    with fiona.open(extent_file, "r") as shapefile:
        shapes = [feature["geometry"] for feature in shapefile]
    
    with rasterio.open(reproj) as src:
        out_image, out_transform = rasterio.mask.mask(src, shapes, crop=True, all_touched=True, pad=True)
        out_meta=src.meta
        out_meta.update({"driver": "GTiff",
             "height": out_image.shape[1],
             "width": out_image.shape[2],
             "transform": out_transform,"nodata":0}
        )
    with rasterio.open(dst, "w", **out_meta) as dest:
            dest.write(out_image)

def merge_rasters_ndvi(list_ndvi_raster, output_mosaic, nodata_value=-9999):
    list_ndvi_obj = [rio.open(ras) for ras in list_ndvi_raster]
    output_meta = list_ndvi_obj[0].meta.copy()
    mosaic, output = merge(list_ndvi_raster)
    output_meta.update(
        {"driver": "GTiff",
         "height": mosaic.shape[1],
         "width": mosaic.shape[2],
         "transform": output,
         "nodata": nodata_value})
    with rio.open(output_mosaic, "w", **output_meta) as m:
        m.write(mosaic)
        
def CD_to_IGLD85(cd_file, cd_crs, dataset, crs):   

    df_cd=pd.read_csv(cd_file, sep=';')
    x_pts=df_cd['XVAL'].to_numpy()
    y_pts=df_cd['YVAL'].to_numpy()
    conv_pts=df_cd['CD'].to_numpy()

    transformer = Transformer.from_crs(cd_crs, crs)

    x_proj, y_proj=transformer.transform(y_pts, x_pts)
    xys=np.c_[x_proj, y_proj]
    tree=cKDTree(xys)

    with rasterio.open(dataset) as src:
        image= src.read()
        bands,rows,cols = np.shape(image)
        image1 = image.reshape (rows*cols,bands)
        indices_0=np.where(image1==0)[0]
        height = image.shape[1]
        width = image.shape[2]
        cols, rows = np.meshgrid(np.arange(width), np.arange(height))
        xs, ys = rasterio.transform.xy(src.transform, rows, cols)
        x= np.array(xs)
        y = np.array(ys)
        newX = np.array(x.flatten())
        newY = np.array(y.flatten())
    
    tiles_center=np.c_[newX, newY]
    dist, idx = tree.query(tiles_center, distance_upper_bound=100000000, k=3)
    w = 1.0 / dist ** 2
    w_sum = np.sum(w, axis=1)
    conv= np.sum(w * conv_pts.flatten()[idx], axis=1) / w_sum
    indices_0 = indices_0[indices_0 < conv.shape[0]]
    conv[indices_0]=0
    return conv

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

def apply_DEM_correction(df, model, img_file, ndvi_field, z_field, crs_dtm,  dataset_name, lower=0, upper=2, nodata_value=-9999, xval='XVAL', yval='YVAL', min_NDVI=0.25):
    input_columns = df.columns
    condition1 = (df[z_field]>lower) & (df[z_field]<=upper) & (~df[z_field].isnull()) & (df[z_field]!=np.nan) 
    df_no_corr = df[~condition1].copy()
    df_to_corr = df[condition1].copy()

    print("CONVERTING DATAFRAME TO GEODATAFRAME BEFORE EXTRACTING NDVI RASTER VALUES")
    gdf_to_corr = xy_to_geodataframe(df_to_corr, xval, yval, crs_dtm)

    print("EXTRACTING NDVI RASTER VALUES TO POINTS")
    gdf_to_corr = extract_raster_values_to_point(img_file, gdf_to_corr, ndvi_field)
    
    condition2 = (~gdf_to_corr[ndvi_field].isnull()) & (gdf_to_corr[ndvi_field]!=np.nan) & (gdf_to_corr[ndvi_field]!=nodata_value) & (gdf_to_corr[ndvi_field]>min_NDVI)
    df_to_corr_nonan = gdf_to_corr[condition2].copy()
    df_to_corr_withnan = gdf_to_corr[~condition2].copy()

    val_dem, val_ndvi = df_to_corr_nonan[z_field].values, df_to_corr_nonan[ndvi_field].values
    
    ##nouveau bloc
    dem2 = np.round(val_dem ** 2, 6)
    ndvi2 = np.round(val_ndvi ** 2, 6)
    dem_ndvi = np.round(val_dem * val_ndvi, 6)
    dem_ndvi2 = np.round(val_dem * ndvi2, 6)
    dem2_ndvi = np.round(dem2 * val_ndvi, 6)
    dem2_ndvi2 = np.round(dem2 * ndvi2, 6)
    predictors = np.asarray([val_dem, dem2, val_ndvi, ndvi2, dem_ndvi, dem_ndvi2, dem2_ndvi, dem2_ndvi2]).T
    
    
    print(f"PREDICTING ELEVATION ERROR IN WETLANDS FOR {dataset_name}")
    if predictors.shape[0]>0:  
        error_pred = model.predict(predictors)
    else:
        print(f"NO PIXEL TO APPLY CORRECTION ON")
        error_pred = 0
    df_to_corr_nonan[z_field] = val_dem - error_pred
    df_full = pd.concat([df_no_corr, df_to_corr_nonan, df_to_corr_withnan], axis=0).sort_index(axis=0)
    df_full = df_full[input_columns].copy()
    return df_full

def xy_to_geodataframe(df, x, y, crs):
    gdf_point = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df[x], df[y]))
    gdf_point.set_crs(epsg=crs, inplace=True)
    return gdf_point

def extract_raster_values_to_point(img_file, pointData, field='NDVI'):
    raster = rasterio.open(img_file)
    if pointData.crs != raster.crs:
        print("REPROJECTING POINT DATA TO EXTRACT NDVI VALUES")
        pointData = pointData.to_crs(raster.crs)
    else:
        print("DATASET CRS = SENTINEL2 IMAGE CRS")
    source_array = raster_to_XYZ_numpy(img_file)
    pointData['XVAL'] = pointData.geometry.x
    pointData['YVAL'] = pointData.geometry.y
    destination_array = pointData[['XVAL', 'YVAL']].to_numpy()
    pointData[field] = interpolationIDW(source_array, destination_array, distance_upper_bound=100, k=1)
    return pointData

def interpolationIDW(source_array, dest_array, distance_upper_bound=10000, k=3):
    x = source_array[:, 0]
    y = source_array[:, 1]
    z = source_array[:, 2]
    xys = np.c_[x, y]
    tree = cKDTree(xys)
    xi = dest_array[:, 0]
    yi = dest_array[:, 1]
    grid = np.c_[xi, yi]
    dist, idx = tree.query(grid, distance_upper_bound=distance_upper_bound, k=k)
    if k>1:
        ## evite le probleme de division par 0
        dist = np.where(dist == 0, 0.000001, dist)
        w = 1.0 / dist ** 2
        w_sum = np.sum(w, axis=1)
        idw = np.sum(w * z.flatten()[idx], axis=1) / w_sum
    else:       
        idx_corr = np.where(idx==len(idx), 0, idx)
        if len(idx_corr) > 0:
            idx_corr2 = np.where(idx==np.max(idx_corr), 0, idx)
        else:
            idx_corr2=idx
        idw = z.flatten()[idx_corr2].reshape(idx.shape)
        idw = np.where(idx==len(idx), np.nan, idw)
    return idw

def change_nodata(src, dst, src_nodat, dst_nodat):  
    with rio.open(src) as src:
        nodat=src.nodata
        out_image= src.read()
        out_image=np.where(out_image==np.min(out_image), dst_nodat, out_image)
        out_meta=src.meta
        out_meta.update({"driver": "GTiff",
             "height": out_image.shape[1],
             "width": out_image.shape[2],
             "nodata":dst_nodat}
        )
    with rio.open(dst, "w", **out_meta) as dest:
            dest.write(out_image)

def execute(t, cfg, res_folder):
    dump_folder = cfg.dump_corr
    res_folder_name=cfg.res_folder_name
    workdir = cfg.workdir
    slope_tiles=cfg.slope_tiles
    flat_CD_conversion=cfg.flat_CD_conversion
    cd_file = cfg.cd_file
    model = joblib.load(cfg.model_file)
    if t not in cfg.only_bathy_tiles :
        dataset_extent=gpd.read_file(fr"{res_folder}\datasets_extent_{t}.shp")
        gdf_tile=gpd.read_file(cfg.tiles_file_overview)
        gdf_tile=gdf_tile.loc[gdf_tile['tile']==t]
        crs=dataset_extent.crs
        crs=str(crs).split(':')
        crs=int(crs[-1])
        gdf_tile=gdf_tile.to_crs(crs)    
        sets=dataset_extent['dataset'].unique()
        dtm=fr"{res_folder}\{t}_1m_DEM_idw_filtered.tif"        
        list_tif=[dtm]
        datasets_corrected=[]
        
        for s in sets:
            specs_file = cfg.specs_file
            df_specs=pd.read_csv(specs_file, sep=';')
            specs=df_specs.loc[df_specs['DATASET']==s]
            data_type=specs['type'].values[0]
            dataset_name = specs['DATASET'].values[0]
            apply_corr_to_dataset = specs['ApplyCorrection'].values[0] # if dataset need to be corrected or not
            gdf_extent_set=dataset_extent.loc[dataset_extent['dataset']==s]
            gdf_extent_set=gdf_extent_set.to_crs(crs)
            gdf_extent_set=gdf_extent_set.dissolve()    
            extent_set_file=fr'{dump_folder}\{s}_extent.shp'
            gdf_extent_set.to_file(extent_set_file)
            clipped_dem=fr'{dump_folder}\dtm_clipped_{s}.tif'
            clipping_raster(extent_set_file, dtm, clipped_dem)
             
             
            if apply_corr_to_dataset and t not in cfg.only_bathy_tiles :
                print('applying wetland correction algorithm...')
                dict_years_ndvi = {}    
                output_mosaic_ndvi = os.path.join(dump_folder, f'{dataset_name}_{t}_{crs}_NDVI.tif')
                if not os.path.exists(output_mosaic_ndvi):
                    path_extent = os.path.join(workdir, specs['Path_extent_file'].values[0])
                    gdf_tiles=gpd.read_file(path_extent)
                    gdf_tiles=gdf_tiles.to_crs(crs)
                    gdf_extent=gdf_tiles.clip(gdf_extent_set)
                    if specs['tiled'].values[0]==True:
                        tiles_set=gdf_extent['tile'].unique()
                        tiles_set=list(tiles_set)                 
                        for tt in tiles_set:
                            tile_id = tt.split('.')[0]
                            gdf_extent['tile'] = gdf_extent['tile'].astype(str)
                            gdf_extent['tile'] = [i[0] for i in gdf_extent['tile'].str.split('.').to_list()]                            
                            gdf_wetland=gpd.read_file(cfg.wetland_extent)
                            gdf_wetland=gdf_wetland.to_crs(crs)
                            gdf_wetland=clean_gdf(gdf_wetland)
                            gdf_extent=clean_gdf(gdf_extent)                               
                            gdf_wetland=gdf_wetland.clip(gdf_tile)                     
                            gdf_extent= gdf_extent.clip(gdf_wetland)
                            gdf_extent_tile = gdf_extent[gdf_extent['tile'] == tile_id]
         
                            if dataset_name == 'Peterborough_extent':
                                year = tile_id.split('LPETERBOROUGH')[0][-4:]
                            elif dataset_name == 'SNC_extent':
                                year = tile_id.split('LSNC')[0][-4:]
                            else:
                                year = specs['Sentinel2_year'].values[0]
         
                            if year not in dict_years_ndvi:
                                dict_years_ndvi[year] = []
                                 
                            if gdf_extent_tile.area.sum() > 100:
                                dict_years_ndvi[year].append(gdf_extent_tile)  
                                     
                    else:
                        print('not tiled')
                        year = specs['Sentinel2_year'].values[0]
                        if year not in dict_years_ndvi:
                                dict_years_ndvi[year] = []
                        gdf_extent=gdf_extent.dissolve()
                        gdf_extent=gdf_extent.to_crs(crs)
                        gdf_extent=gdf_extent.clip(gdf_tile)
                         
                        gdf_wetland=gpd.read_file(cfg.wetland_extent)
                        gdf_wetland=gdf_wetland.to_crs(crs)   
                        gdf_wetland=gdf_wetland.clip(gdf_tile)                      
                        gdf_extent= gdf_extent.clip(gdf_wetland)
                        if gdf_extent.area.sum() > 100:
                            dict_years_ndvi[year].append(gdf_extent)
                             
                        else:
                            print('no wetland portion in this dataset')
                            continue

                    if len(dict_years_ndvi[year])>0:
                                             
                        list_ndvi_raster = []
                        count_ndvi=0
                        for year, list_gdf in dict_years_ndvi.items():
                            count_ndvi+=1
                            if len(dict_years_ndvi[year])>1:
                                gdf_merge = pd.concat(list_gdf, axis=0)
                            else:
                                gdf_merge=gdf_extent
                                 
                            if gdf_merge.area.sum() < 100:
                                print('no wetland portion in this dataset')
                                continue
                             
                            gdf_merge=clean_gdf(gdf_merge)
                            gdf_merge=gdf_merge.dissolve()
                            gdf_merge_file=os.path.join(dump_folder, f'gdf_merge.shp')
                            gdf_merge.to_file(gdf_merge_file)

                            ndvi_raster = os.path.join(cfg.path_img_sentinel2, year, f'Tile_{t}_4326_{year}_S2_NDVI_BOA1.tif')
                            ndvi_proj=fr'{dump_folder}\Tile_{t}_4326_{year}_S2_NDVI_BOA1_proj.tif'
                            reproj_raster(ndvi_raster, crs, ndvi_proj)
                             
                            ndvi_raster_clipped = os.path.join(dump_folder, f'{dataset_name}_{t}_{crs}_{year}_NDVI_{count_ndvi}.tif')
                            clipping_raster(gdf_merge_file, ndvi_proj, ndvi_raster_clipped)
                            list_ndvi_raster.append(ndvi_raster_clipped)
                             
                        if len(list_ndvi_raster)>0:
                            merge_rasters_ndvi(list_ndvi_raster, output_mosaic_ndvi)
                             
                    else:
                        print('no wetland portion in this dataset')
                        continue
                             
                if os.path.exists(output_mosaic_ndvi):
                    ## in LKO CD to IGLD85 is fixed to 74.2 which avoids computation
                    if t not in slope_tiles:
                        conv_igld85_to_cd=flat_CD_conversion
                    else:
                        conv_igld85_to_cd = CD_to_IGLD85(cd_file, 4326, clipped_dem, crs)
                     
                     
                    export=raster_to_XYZ_numpy(clipped_dem)
                    arr=export[:,2]

                    # remove outliers
                    arr=np.where(arr>1000, 0, arr)
                    arr=np.where(arr<-1000, 0, arr)
                     
                    # convert igld85 to cd to apply dem correction
                    arr_cd = arr-conv_igld85_to_cd
         
                    df_coords = pd.DataFrame(data={'XVAL':export[:,0], 'YVAL':export[:,1], 'ZVAL':arr, 'ZVAL_CD':arr_cd})
                    # extract raster values to points
                    ndvi_field = 'NDVI'
                    print('applying correction.....')
                    gdf_coords = apply_DEM_correction(df_coords, model, output_mosaic_ndvi, ndvi_field, 'ZVAL_CD', crs, dataset_name)  # apply correction
                    # reconvert to igld85
                    arr = gdf_coords['ZVAL_CD'].values + conv_igld85_to_cd                     
         
                    print(f"DEM CORRECTION SUCCESFULLY APPLIED TO {dataset_name}")
                    datasets_corrected.append(dataset_name)
            else:
                export=raster_to_XYZ_numpy(clipped_dem)
                arr=export[:,2]
                print(f'ground filter correction for {s} is not needed')

            clipped_dem_corr = os.path.join(dump_folder, f'{t}_{s}_clipped_dem_corr.tif')
            with rio.open(clipped_dem) as src:
                ras_data = src.read()
                ras_meta = src.profile
                ras_meta.update({"driver": "GTiff", "nodata":0}
            )
            arr=arr.reshape(ras_data.shape)
            with rio.open(clipped_dem_corr, 'w', **ras_meta) as dst:
                dst.write(arr)
        
            clipped = os.path.join(dump_folder, f'{t}_{s}_corr_clipped.tif')
        
            clipping_raster(extent_set_file, clipped_dem_corr, clipped)
            
            list_tif.insert(0, clipped)

        dst=os.path.join(dump_folder, f'{t}_1m_dem_wetland_corrected.tif')
        dst2=os.path.join(res_folder, f'{t}_1m_dem_wetland_corrected.tif')
        merging_rasters(list_tif, dst, gdf_tile, crs, dump_folder)
        
        #change_nodata(dst, dst2, 0, -32768)
        change_nodata(dst, dst2, -32768, 0)
        
        dem_10m=fr'{res_folder}\{t}_10m_dem_wetland_corrected.tif'   
        cmd_resamp_10=fr'{cfg.gdal_path}\gdalwarp.exe -tr 10 10 -r cubicspline -overwrite {dst2} {dem_10m}'
        os.system(cmd_resamp_10)
        
        for filename in os.listdir(dump_folder):
            file_path = os.path.join(dump_folder, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f'Failed to delete {file_path}. Reason: {e}')
        
    else:
        datasets_corrected=[]    

    return datasets_corrected

if __name__ == '__main__':
    
    lot_of_tiles=[436]
    
    for tiles in lot_of_tiles:
        tiles=[tiles]
        res_folder=fr'F:\DEM_GLAMM\Git_DEM_GLAM\results\{tiles[0]}_V4_4'
        dump_folder=r'F:\DEM_GLAMM\Git_DEM_GLAM\dump_corr'
        final_res=1
        import CFG_DEM_CREATION as cfg
        execute(tiles[0], cfg, res_folder)



                
                