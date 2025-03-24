
'''
Main code to create seamless topobathy DEM in LKO, USL and SLR from various topo and bathy datasources 
Parameters are set in CFG_DEM_CREATION.py file 
Developped for GLAM expedited review in 2023

Author: Antoine Maranda (antoine.maranda@ec.gc.ca)
Environment and Climate Change Canada, National Hydrologic Services, Hydrodynamic and Ecohydraulic Section
'''

##PYTHON MODULES to import##
import os
import numpy as np
import pandas as pd
import geopandas as gpd
import time, datetime 
import glob
import json
from glob import iglob
import shapely
from shapely.validation import make_valid
from shapely.geometry import Polygon, MultiPolygon
from rasterio.merge import merge
import rasterio as rio
import fiona
import rasterio.mask
from rasterio.warp import reproject, Resampling, calculate_default_transform
from osgeo import ogr
from osgeo import osr
from scipy.spatial import cKDTree
from pyproj import transform, Proj
import pyproj
from pyproj import Transformer
import shutil
import matplotlib.pyplot as plt
from datetime import date
import logging
from rasterio.transform import Affine
from pyproj import CRS
import earthpy.spatial as es
from osgeo import gdal
from WBT.whitebox_tools import WhiteboxTools
wbt = WhiteboxTools()
wbt.set_verbose_mode(False)
from ast import literal_eval
from rasterio.features import shapes
from shapely.geometry import shape

##OTHER CODES TO IMPORT 
from dataset_tiles_intersect import *
import dataset_extent_modif
import Wetland_correction_seperate
import MASK_DEM_EXEC
import CFG_DEM_CREATION as cfg

def reclass_and_polygonize(src, dst):
    with rasterio.open(src) as raster_file:
        raster = raster_file.read()
        raster[raster == 0] = -32768
        out_meta = raster_file.meta
        out_meta.update({"driver": "GTiff",
                         "nodata": -32768,
                         "height": raster.shape[1],
                         "width": raster.shape[2],
                         "dtype": 'float32'})
        raster[raster > -999] = 1
        mask = raster != -32768
        raster=raster.astype(np.float32)
        shapes_generator = shapes(raster, mask=mask, transform=raster_file.transform)
        polygons = []
        values = []

        for geom, value in shapes_generator:
            polygons.append(shape(geom))
            values.append(value)

        gdf = gpd.GeoDataFrame({'geometry': polygons, 'value': values}, crs=raster_file.crs)
    gdf.to_file(dst)


def clean_gdf(gdf):
    extent=gdf.explode(ignore_index=True, index_parts=True)
    extent.reset_index(drop=True)    
    list_2_drop=[]
    for f in range(len(extent)):
        extent.geometry.iloc[f] = make_valid(extent.geometry.iloc[f])
        if str(extent.geometry.iloc[f]).split(' ')[0]=='GEOMETRYCOLLECTION':
            count_geom=-1
            geoms_list=[]
            for geom in extent.geometry.iloc[f].geoms:
                count_geom+=1
                if str(geom).split(' ')[0]=='POLYGON' and str(geom).split(' ')[0]=='MULTIPOLYGON':
                    geoms_list.append(geom)
                extent.geometry.iloc[f]=geom         
        elif str(extent.geometry.iloc[f]).split(' ')[0]!='POLYGON' and str(extent.geometry.iloc[f]).split(' ')[0]!='MULTIPOLYGON':
            list_2_drop.append(f)       
    extent=extent.drop(labels=list_2_drop, axis=0)
    return extent

def csv_to_geotiff(outFileName, Xfield, Yfield, field, dst_tif, resolution, final_res, crs):
    xval = Xfield
    yval = Yfield
    vrt_fn =  outFileName.replace('.csv', '.vrt')
    lyr_name =  outFileName.replace('.csv', '')
    lyr_name = os.path.basename(lyr_name)
    dst_tif =  dst_tif   
    df = pd.read_csv(outFileName, sep=';')
    xmax = (df[xval].max() + (resolution / 2))
    xmin = (df[xval].min() - (resolution / 2))
    ymax = (df[yval].max() + (resolution / 2))
    ymin = df[yval].min() - (resolution / 2)
   
    with open(vrt_fn, 'w+') as fn_vrt:
        fn_vrt.write('<OGRVRTDataSource>\n')
        fn_vrt.write('\t<OGRVRTLayer name="%s">\n' % lyr_name)
        fn_vrt.write('\t\t<SrcDataSource>%s</SrcDataSource>\n' % outFileName)
        fn_vrt.write('\t\t<GeometryType>wkbPoint</GeometryType>\n')
        fn_vrt.write(f'\t\t<GeometryField encoding="PointFromColumns" x="{xval}" y="{yval}" z="{field}"/>\n')
        fn_vrt.write('\t</OGRVRTLayer>\n')
        fn_vrt.write('</OGRVRTDataSource>\n')

    cmd=fr'{gdal_path}\gdal_grid.exe -a linear:radius=100 -a_srs epsg:{crs} -txe {xmin} {xmax} -tye {ymin} {ymax} -tr {final_res} {final_res} {vrt_fn} {dst_tif} --config GDAL_NUM_THREADS ALL_CPUS'
    os.system(cmd)
    return

def hillshadeoverview(dtm, nodat):
    with rio.open(dtm) as src:
        image=src.read(1)
        elevation=image
        elevation[elevation == nodat] = np.nan
        elevation=elevation*10
    fig, ax = plt.subplots(figsize=(15, 9))
    hillshade= es.hillshade(elevation)
    ax.imshow(hillshade*-1, cmap="Greys")
    ax.imshow(elevation, cmap="turbo", alpha=0.6)
    png=dtm.replace('.tif', '_hillshade.png')
    plt.savefig(png)
    plt.close()

def remove_small_parts_and_holes(gdf_raster, tresh):       
    gdf_raster=gdf_raster.explode(ignore_index=True, index_parts=True)
    gdf_raster['area']=gdf_raster.area
    gdf_raster= gdf_raster.loc[gdf_raster['area']>tresh]
    for l in range(len(gdf_raster)):
        polygon=gdf_raster['geometry'].iloc[l]
        if str(polygon).split(' ')[0]=='POLYGON':
            list_interiors = []
            eps = tresh
            for interior in polygon.interiors:
                i = Polygon(interior)    
                if i.area > eps:
                    list_interiors.append(interior)
            new_polygon = Polygon(polygon.exterior.coords, holes=list_interiors)
            gdf_raster['geometry'].iloc[l]=new_polygon
        elif str(polygon).split(' ')[0]=='MULTIPOLYGON':
            list_parts = []
            eps = tresh
            for poly in polygon.geoms:
                list_interiors = [] 
                for interior in poly.interiors:
                    j = Polygon(interior)
                    if j.area > eps:
                        list_interiors.append(interior)
                temp_pol = Polygon(poly.exterior.coords, holes=list_interiors)
                list_parts.append(temp_pol)
            new_multipolygon = MultiPolygon(list_parts)               
            gdf_raster['geometry'].iloc[l]=new_multipolygon               
        else:
            print('STRANGE GEOMETRY!!')
    gdf_raster=clean_gdf(gdf_raster)
    return gdf_raster

def overlappingstats(path_actual, path_prev, dump_folder, res_folder, tif_actual, tif_prev, count, set_nb):
    if (os.path.exists(path_actual)) and (os.path.exists(path_prev)):
        gdf_prev=gpd.read_file(path_prev)
        for f in range(len(gdf_prev)):
            gdf_prev.geometry.values[0] = make_valid(gdf_prev.geometry.values[0])
        gdf_actual=gpd.read_file(path_actual)
        for f in range(len(gdf_actual)):
            gdf_actual.geometry.values[0] = make_valid(gdf_actual.geometry.values[0])
        overlap= gdf_prev.clip(gdf_actual)
        overlap=overlap.explode(index_parts=True)
        overlap=overlap.loc[(overlap.geom_type=='Polygon')|(overlap.geom_type=='MutliPolygon')]
        for f in range(len(overlap)):
            overlap.geometry.values[0] = make_valid(overlap.geometry.values[0])
        if overlap.empty:
            print('geoDataFrame is empty, skipping...')
            med_error=0
        else:
            over_area=overlap.area[0].sum()
            if over_area<1000:
                print('not much of overlap between those 2 don t bother...')
                med_error=0
            else:
                over_file=fr'{dump_folder}\overlap_{set_nb}_{count}.shp'
                overlap.to_file(fr'{dump_folder}\overlap_{set_nb}_{count}.shp')
                sets=[count, set_nb]
                for s in sets:
                    if s == count:
                        previous=tif_actual
                        clipped_over=fr'{dump_folder}\{s}_clipped_by_overlap_{set_nb}_{count}.tif'
                        clipping_raster(over_file, previous, clipped_over)
                        array_g=raster_to_XYZ_numpy(clipped_over)
                        array_g=array_g[array_g[:,2] != 0]
                        xi=array_g[:, 0]
                        yi=array_g[:, 1]
                        zi=array_g[:, 2]
                        grid=np.c_[xi, yi]
                    else:
                        previous=tif_prev
                        clipped_over=fr'{dump_folder}\{s}_clipped_by_overlap_{set_nb}_{count}.tif'
                        clipping_raster(over_file, previous, clipped_over)
                        array=raster_to_XYZ_numpy(clipped_over)
                        array=array[array[:, 2] != 0]
                        x=array[:, 0]
                        y=array[:, 1]
                        z=array[:, 2]
                        xys=np.c_[x, y]
                        tree=cKDTree(xys)
                dist, idx = tree.query(grid,  k=1)
                not_to_far=np.where(dist<=10)[0]
                idx=idx[not_to_far]
                new_z=z.flatten()[idx].reshape(idx.shape)
                zi=zi[not_to_far]
                ### Difference is actual dataset minus previous datasets
                diff=zi-new_z
                if len(diff) > 0:
                    med_error = np.median(diff)
                else:
                    med_error = 0
                plt.hist(diff, bins=100)
                plt.suptitle(fr"Elevation difference between overlaping points of {datasets_sorted[count-1]} and {datasets_sorted[set_nb-1]} (m IGLD85)", fontsize='small')
                plt.title(f'Overlap area: {over_area.round(0)}; N overlaping points:{pd.DataFrame(diff).describe()[0][0].round(0)}; mean_distance: {np.round(np.mean(dist), 2)};\nmean_diff: {pd.DataFrame(diff).describe()[0][1].round(2)}; std:{pd.DataFrame(diff).describe()[0][2].round(2)}, 25%:{pd.DataFrame(diff).describe()[0][4].round(2)}, 75%:{pd.DataFrame(diff).describe()[0][6].round(2)}', fontsize='small')
                png_file=fr'{res_folder}\overlap_stats_{datasets_sorted[count-1]}_{datasets_sorted[set_nb-1]}.png'
                plt.savefig(png_file)
                plt.close()
                logger.info(f'\Statistics of elevation differences with overlapping points calculated and provided in {png_file} \n')
    else:
        med_error=0
        pass
    return med_error
     
def to_IGLD85(datum, gdf_grid, cd_file, raster, t):
    print(f'dataset {name} is in {datum} vertical datum')
    ## TODO pour le moment une valeur de converion par tuile base sur le IDW par rapport au bechmarks vs le centroid de la tuile, voir si on veut faire qqch de plus fancy
    if datum=='CGVD2013':
        conv=gdf_grid['H85-H13'].loc[gdf_grid['tile']==t].iloc[0]
        print(f'conversion to IGLD85 is {conv}')
    elif datum=='CGVD28':
        conv=gdf_grid['H85-H28'].loc[gdf_grid['tile']==t].iloc[0]
        print(f'conversion to IGLD85 is {conv}')
    elif datum=='IGLD85':
        conv=0
        print(f'conversion to IGLD85 is {conv}')
    elif datum=='CD':
        if t not in slope_tiles:
            conv=flat_CD_conversion
            print(f'conversion to IGLD85 is between {conv}')
        else:
            conv=CD_to_IGLD85(cd_file, 4326, raster)
            print(f'conversion to IGLD85 is between {conv.min()} to {conv.max()}')
            logger.info(f'conversion to IGLD85 is between {conv.min()} to {conv.max()}')
    else:
        print('strange datum!! quitting')
        quit()   
    return conv

def resample_raster(res, final_res, raster, dst):
    upscale_factor = res/final_res
    with rasterio.open(raster) as dataset:
        # resample data to target shape
        data = dataset.read(
            out_shape=(
                dataset.count,
                int(dataset.height * upscale_factor),
                int(dataset.width * upscale_factor)
            ),
            resampling=Resampling.bilinear
        )    
        # scale image transform
        transform = dataset.transform * dataset.transform.scale(
            (dataset.width / data.shape[-1]),
            (dataset.height / data.shape[-2])
        )         
        out_meta=dataset.meta
        out_meta.update({"driver": "GTiff",
             "height": data.shape[1],
             "width": data.shape[2],
             "transform": transform, "nodata":0})

    with rasterio.open(dst, "w", **out_meta) as dest:
        dest.write(data)

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

def CD_to_IGLD85(cd_file, cd_crs, dataset):   

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

def merging_rasters(list_tif_file, dst, extent, crs_tile):
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
          
def buffer_w_ogr(gdf, crs, buffer_size, fname):
        gdf=gdf.dissolve()
        wkt = [geom.wkt for geom in gdf.geometry]
        pt = ogr.CreateGeometryFromWkt(wkt[0])
        poly = pt.Buffer(buffer_size)
        driver = ogr.GetDriverByName("ESRI Shapefile")
        ds = driver.CreateDataSource(fname)
        srs = osr.SpatialReference()
        srs.ImportFromEPSG(crs)
        layer = ds.CreateLayer("poly", srs, ogr.wkbMultiPolygon)
        idField = ogr.FieldDefn("id", ogr.OFTInteger)
        layer.CreateField(idField)
        featureDefn = layer.GetLayerDefn()
        feature = ogr.Feature(featureDefn)
        feature.SetGeometry(poly)
        feature.SetField("id", 1)
        layer.CreateFeature(feature)
        feature = None
        ds = None

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

def dem_no_data_and_clip(crs, src, dst, dump_folder, res_folder, final_res, gdal_scripts, gdal_path, tile_xtent):
    final=f'{dump_folder}\{t}_{final_res}m_DEM_idw_filtered.tif'
    cmd_final=fr'{gdal_path}\gdalwarp.exe -s_srs EPSG:{crs} -t_srs EPSG:{crs} -ot Float32 -srcnodata -32768 -dstnodata -32768 -overwrite {src} {final}'
    os.system(cmd_final)
    final_baton=f'{dump_folder}\{t}_{final_res}m_DEM_idw_filtered2.tif'
    cmd_fill=fr'python {gdal_scripts}\gdal_fillnodata.py -md 1000 {final} {final_baton}'
    os.system(cmd_fill)
    final_baton_clipped=f'{dump_folder}\{t}_{final_res}m_DEM_idw_filtered_clipped.tif'
    clipping_raster(tile_xtent, final_baton, final_baton_clipped)
    cmd_final2=fr'{gdal_path}\gdalwarp.exe -s_srs EPSG:{crs} -t_srs EPSG:{crs} -ot Float32 -srcnodata -32768 -dstnodata -32768 -overwrite {final_baton_clipped} {dst}'
    os.system(cmd_final2)
    files=[final, final_baton, final_baton_clipped]
    for f in files:
        os.remove(f)
            
   
if __name__ == '__main__':
       
    ###main variable
    gdal_scripts = cfg.gdal_scripts
    gdal_path = cfg.gdal_path
    dataset_dir = cfg.dataset_dir
    workdir = cfg.workdir
    dump_folder = cfg.dump_folder
    os.makedirs(dump_folder, exist_ok=True) 
    clean_dump=cfg.clean_dump
    tile_overlap_dir = cfg.tile_overlap_dir
    tiles_file_overview = cfg.tiles_file_overview
    tiles_file = cfg.tiles_file
    AOI = cfg.AOI
    specs_file = cfg.specs_file
    tiles = cfg.tiles
    final_resolutions = cfg.final_resolutions
    previous_version = cfg.previous_version
    cd_file = cfg.cd_file  
    gdf_grid_overview=gpd.read_file(tiles_file_overview)
    df_grid=pd.read_csv(tiles_file, sep=';')
    res_folder_name=cfg.res_folder_name
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    log_format = "|%(levelname)s| : %(message)s"
    formatter = logging.Formatter(log_format)
    list_worked=[]
    list_bugged=[]
    slope_tiles=cfg.slope_tiles
    flat_CD_conversion=cfg.flat_CD_conversion
    
    ## idenitfy which datasets intersects with each tile of the AOI 
    tile_dataset, affected_tiles=main_intersect(tiles_file_overview, workdir, previous_version, cfg.tile_overlap_dir)
    if len(affected_tiles)>0:
        print(fr'WARNING: tiles: {affected_tiles} are affected by an updated or new dataset, consider to rerun those tiles. If you decide not to do so, you can simply ignore this message and rerun')
        logger.info(fr'WARNING: tiles: {affected_tiles} are affected by an updated or new dataset, consider to rerun those tiles. If you decide not to do, so you can simply ignore this message and rerun')
        quit()
    df_dataset=pd.read_csv(tile_dataset, sep=';')
    
    ## iterate trough each tile
    for t in tiles:
    
        med_error=0
        try:
            ## create folders and initiate log
            print(f'processing tile {t}')
            res_folder = os.path.join(workdir, 'results', f'{t}_{res_folder_name}')
            if not os.path.exists(res_folder):
                os.makedirs(res_folder)
            log_file = os.path.join(res_folder, f'{t}_log.log')
            if os.path.exists(log_file):
                os.remove(log_file)

            for handler in logger.handlers[:]:
                logger.removeHandler(handler)

            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
            logger.info(f'********** LOG FILE FOR DTM CREATION OF TILE {t} **********\n')
            logger.info('Author: Antoine Maranda antoine.maranda@ec.gc.ca\n')
            logger.info('Contributors: Dominic Theriault and Patrice Fortin \n')
            logger.info('Hydrodynamic and Ecohydraulic Section, NHS-QC, Environment and Climate Change Canada\n')
            logger.info(f'Date of creation: {date.today()}\n')
            logger.info(f'**********PROCESS Started at : {datetime.datetime.now()}***************\n')
            start=datetime.datetime.now()
     
            ## determine tile native CRS which will be used throughout the process
            dct_crs={17:'epsg:32617', 18:'espg:32618', 19:'epsg:32619'}
            zone=df_grid['UTM'].loc[df_grid['TILE']==t].values[0]
            crs_long=dct_crs[zone]
            crs=int(crs_long.split(':')[-1])
            df_grid_utm=df_grid.loc[df_grid['UTM']==zone]
            gdf_grid=gpd.GeoDataFrame(df_grid_utm, crs=crs, geometry=gpd.GeoSeries.from_wkt(df_grid_utm['GEOMETRY']))            
            logger.info(f'CRS: EPSG:{crs}')
             
            aoi=gpd.read_file(AOI)
            aoi=aoi.to_crs(crs)
     
            ##list datasets that will be used to create tile's dem and sort by priority
            datasets=[]     
            for d in df_dataset['dataset_extent'].unique():
                good_set=d.split('\\')[-1]
                good_set=good_set.split('.')[0]
                tiles_grid=df_dataset['tiles'].loc[df_dataset['dataset_extent']==d].values[0]
                tiles_grid= literal_eval(tiles_grid)
                tiles_grid= [int(s) for s in tiles_grid]
                if t in tiles_grid:
                    datasets.append(good_set)
            df_specs=pd.read_csv(specs_file, sep=';')
            df_specs=df_specs.loc[df_specs['DATASET'].isin(datasets)]
            df_specs=df_specs.sort_values(by='priority')
            datasets_sorted=df_specs['DATASET'].values
            datasets_name=[]
            for d in datasets_sorted:
                dd=d.replace('_extent','')
                datasets_name.append(dd)           
            logger.info(f'Datasets used to create DEM (in priority order): {datasets_name}')
            topo=df_specs.loc[(df_specs['type']=='topo') | (df_specs['type']=='topo_bathy')]
            topo=topo.sort_values(by='priority')
            logger.info('\n********************************************************\n')
     
            ##gdf of tile we are processing and clip with aoi
            gdf_tile=gdf_grid.loc[gdf_grid['TILE']==t]
            gdf_tile=gdf_tile.to_crs(crs)         
            gdf_aoi=gpd.read_file(AOI)
            gdf_aoi=gdf_aoi.to_crs(crs)
            gdf_tile=gdf_tile.clip(gdf_aoi)
            gdf_tile=gdf_tile.dissolve()
            bounds=tuple(gdf_tile.bounds.values[0])
            logger.info(f'Tile bounds: {bounds}')
             
            ## intiate various empty lists and counts
            dfs=[]        
            dfs_real=[]
            gdfs=[]
            gdfs_topo=[]        
            count=0
            count_bathy=0
            count_topo=0
            count_feather=0
            previous_data_type='NA'
            extent_topo_buff20_file=fr'{dump_folder}\extent_topo_buff20_bidon.shp'
             
             
            ## iterate trough each dataset
            for name in datasets_sorted[:]:
                 
                count+=1
                logger.info(f'Processing : {name.replace("_extent", "")}')
                print(fr'processing {name}....')
                res=df_specs['resolution'].loc[df_specs['DATASET']==name].iloc[0]
                if res >=1:
                    res=int(res)
                     
                ##get dataset specs
                specs=df_specs.loc[df_specs['DATASET']==name]
                data_type=specs['type'].values[0]
                dataset_name = specs['DATASET'].values[0]
                 
                if data_type=='bathy':
                    count_bathy+=1
                else:
                    count_topo+=1
     
                ## keep only the part of the dataset that intersects with the tile we are processing
                path=fr'{tile_overlap_dir}\{name}_tile_intersect.shp'
                gdf=gpd.read_file(path)
                gdf=gdf.to_crs(crs)
                gdf=gdf.loc[gdf['tile_grid']==t]
                logger.info(f'dataset specs: {specs.iloc[0]}')
                if 'tile' in list(gdf):
                    gdf=gdf[['tile_grid', 'tile', 'geometry']]
                else:
                    gdf['tile']=1
                    gdf=gdf[['tile_grid', 'tile', 'geometry']]
                for f in range(len(gdf)):
                    gdf.geometry.iloc[f] = make_valid(gdf.geometry.iloc[f])
     
                gdf=gdf.clip(gdf_tile)
                extent=gdf
     
                for f in range(len(extent)):
                    extent.geometry.iloc[f] = make_valid(extent.geometry.iloc[f])
                if len(extent)==0:
                    logger.info(fr'STRANGE dataset {name} do not overlap this tile, while tile_intersects was stating otherwise...')
                    logger.info('\n********************************************************\n')
                    print(fr'STRANGE dataset {name} do not overlap this tile, while tile_intersects was stating otherwise...')
                    continue
                else:
                    pass
     
                ### PRIORITIZING ###
                ### Keep only part of the dataset that is not already covered by datsets of higher priority while keeping an overlap of 20m between each set
                extent_file=fr'{dump_folder}\extent_{t}_dataset_{count}_clipped.shp'
                extent_base_file=fr'{dump_folder}\extent_{t}_dataset_{count}_clipped_sans_buff.shp'
                extent_full=fr'{dump_folder}\extent_{t}_dataset_{count}_full.shp'
                extent_all_file=fr'{dump_folder}\extent_{t}_dataset_ALL_sans_buff.shp'
     
                if count ==1:
                    buff=2
                    logger.info(f'buffer size used to create overlap with adjacent dataset: {buff}')
                    if count != len(datasets_sorted):
                        extent_buff=extent
                        extent_buff=remove_small_parts_and_holes(extent_buff, res**2)
                    else:
                        extent_buff=gdf_tile
                         
                    extent_buff=extent_buff.dissolve()
                    extent_buff['geometry']=extent_buff['geometry'].simplify(1)
                    extent_base=extent_buff
                    extent_base.to_file(extent_base_file)
                    buffer_w_ogr(extent_buff, crs, buff, extent_file)
                    extent_buff=gpd.read_file(extent_file)
                    extent_buff.to_file(extent_file)
                    extent_buff.to_file(extent_full)
                     
                    ## little workaround for a bug in Hamilton_Niagara extent file
                    if name=='Hamilton_Niagara_extent':
                        extent_buff.to_file(extent_base_file)
                        extent_all=extent_buff
                        extent_all=extent_all.dissolve()
                    else:                      
                        extent_all=extent_base
                        extent_all=extent_all.dissolve()
                     
                else :  
                    if count != len(datasets_sorted):
                        extent_buff=extent    
                                         
                    #for nonna dataset (last priority) fills the rest of the tile
                     
                    elif 'NONNA' in name:
                        extent_buff['geometry']=gdf_tile.buffer(200)
                     
                    else:    
                        extent_buff=gdf_tile
                                              
                    if data_type == previous_data_type and data_type == 'bathy':
                        buff=20
                         
                    if count == len(datasets_sorted):
                        buff=0

                    logger.info(f'buffer size used to create overlap with adjacent dataset: {buff}')
                         
                    if 'NONNA' not in name:
                        extent_buff=remove_small_parts_and_holes(extent_buff, 100)                    
                     
                     
                    extent_buff=extent_buff.dissolve()
                    extent_buff.to_file(extent_full)
     
                    print('....identify areas that are not already use by other datasets...')
                    extent_buff['geometry']=extent_buff['geometry'].difference(extent_all['geometry'])
                    extent_buff=extent_buff.dissolve()
     
                    if extent_buff['geometry'].iloc[0] == 'GEOMETRYCOLLECTION EMPTY' or extent_buff['geometry'].area.sum() < 1000:
                        print('skipping')
                        logger.info(fr'nothing left of this dataset after prioritization... skipping')
                        logger.info('\n********************************************************\n')
                        print('< than 1000m2 left of this dataset... skipping')
                        continue
                     
                    ## cleaning the resulting geometry
                    extent_buff=extent_buff.explode(ignore_index=True, index_parts=True)
                    for f in range(len(extent_buff)):
                        extent_buff.geometry.iloc[f] = make_valid(extent_buff.geometry.iloc[f])
                    extent_buff=extent_buff.dissolve()
                    extent_buff=clean_gdf(extent_buff)
                    extent=clean_gdf(extent)
                    extent=extent.dissolve()
                     
                    extent_buff.to_file(fr'{dump_folder}\extent_buff.shp')
                    extent.to_file(fr'{dump_folder}\extent.shp')
                     
                    ## double check
                    if count == len(datasets_sorted):
                        gdf_check=extent
                        gdf_check=gdf_check.dissolve()
                        check=extent_buff.clip(gdf_check)
                        check=clean_gdf(check)
                        check.to_file(fr'{dump_folder}\check.shp')
                        check=check.dissolve()
                        if len(check)==0 or check['geometry'].iloc[0] == 'GEOMETRYCOLLECTION EMPTY':
                            logger.info(fr'nothing left of this dataset after prioritization... skipping')
                            logger.info('\n********************************************************\n')
                            print('nothing left of this dataset... skipping')
                            continue

                    extent_buff['geometry']=extent_buff['geometry'].simplify(1)
                    extent_base=extent_buff
                    extent_base=extent_base.explode()
     
                    ## dont bother to fill in the rest of the tile with low priority datasets if it is only sparse small parts 
                    if 'NONNA' not in name:
                        extent_base=remove_small_parts_and_holes(extent_base, 1000)                       
                    if count == len(datasets_sorted):
                            extent_base=remove_small_parts_and_holes(extent_base, 10000)
                    if 'OLD_USACE' in name:
                            extent_base=remove_small_parts_and_holes(extent_base, 1000000)
                    extent_base=extent_base.dissolve()
                    extent_base.to_file(extent_base_file)
     
                    total_area=extent_base['geometry'].area.sum()
                    if data_type=='bathy' and total_area<100000 and 'NONNA' not in name:
                        logger.info(fr'not much left of this dataset after prioritization... skipping')
                        logger.info('\n********************************************************\n')
                        print('not much left of this dataset... skipping')
                        continue
                     
                    if len(extent_base)==0:
                        logger.info(fr'nothing left of this dataset after prioritization... skipping')
                        logger.info('\n********************************************************\n')
                        print('not much left of this dataset... skipping')
                        continue
                     
                    buffer_w_ogr(extent_base, crs, buff, extent_file)
                    extent_buff=gpd.read_file(extent_file)
                     
                    extent_buff.to_file(fr'{dump_folder}\test.shp')
                     
                    extent_buff=extent_buff.explode()
     
                    if 'NONNA' not in name:
                        extent_buff=remove_small_parts_and_holes(extent_buff, 1000)   
                    if count == len(datasets_sorted) and data_type=='bathy':
                        extent_buff=remove_small_parts_and_holes(extent_buff, 200000)
                    elif count == len(datasets_sorted): 
                        extent_buff=remove_small_parts_and_holes(extent_buff, 10000)
                    extent_buff=extent_buff.dissolve()
                     
                    if len(extent_base)==0 or extent_base['geometry'].iloc[0] == 'GEOMETRYCOLLECTION EMPTY':
                        logger.info(fr'nothing much left of this dataset after prioritization... skipping')
                        logger.info('\n********************************************************\n')
                        print('not much left of this dataset... skipping')
                        continue
                     
                    extent_buff.to_file(fr'{dump_folder}\test2.shp')     
                    extent_buff.to_file(extent_file)
                     
                    ## little workaround for a bug in Hamilton_Niagara extent file
                    if name=='Hamilton_Niagara_extent':
                        buffer_w_ogr(extent_base, crs, 1, extent_base_file)
                        extent_small_buff=gpd.read_file(extent_base_file)
                        extent_base=extent_small_buff
                     
                    total_area=extent_buff['geometry'].area.sum()
                    if data_type=='bathy' and total_area<100000 and 'NONNA' not in name:
                        logger.info(fr'not much left of this dataset after prioritization... skipping')
                        logger.info('\n********************************************************\n')
                        print('not much left of this dataset... skipping')
                        continue
     
                    if count != len(datasets_sorted):
                        #merge processed dataset to clip with the next one
                        extent_all['geometry']=extent_all['geometry'].union(extent_base['geometry'])
                        extent_all=extent_all.dissolve()
                        extent_all=clean_gdf(extent_all)
                        extent_all=extent_all.dissolve()
                        extent_all.to_file(extent_all_file)
                         
                        ### PRIORITIZATION DONE ###
                    
                ##indentify dataset vertical datum
                datum=specs['datum'].values[0]
                                
                tile_xtent = os.path.join(fr'{res_folder}', f'{t}_extent.shp')
                gdf_tile.to_file(tile_xtent)
                 
                ## for nonna and NOAA we include 6000m buffer around the tile to properly interpolate at the edge of the tile
                if 'NONNA' in name or 'NOAA' in name:
                    tile_buff=fr'{dump_folder}\tile_buff_6000.shp'
                    buffer_w_ogr(gdf_tile, crs, 6000, tile_buff)
                    gdf_tile_buff=gpd.read_file(tile_buff)
                    bounds_buff=tuple(gdf_tile_buff.bounds.values[0])
                 
                ## merging tiled DATASET in one mosaic
                if specs['tiled'].values[0]==True:
                    dst = os.path.join(dump_folder, f'{t}_{count}_mosaic.tif')
                    tiles_set=gdf['tile'].unique()
                    tiles_set=list(tiles_set)
                    logger.info(f'tiled dataset, using tiles: {tiles_set}')
                    print(f'mosaicing {len(tiles_set)} tiles together...')
                    list_tif_file=[]
                    for tt in tiles_set:
                        tt=tt.split('.')[0]
                        tt=tt+specs['extentions'].values[0]
                        path_tt= os.path.join(dataset_dir+os.path.sep, fr"{specs['dataset_Path'].values[0]}", tt)
                        list_tif_file.append(path_tt)
                    ### used the buffered verison of tile extent for NONNA
                    if not os.path.exists(dst):
                        if 'NONNA' in name or 'NOAA' in name:                     
                            item=os.path.join(dataset_dir+os.path.sep, fr"{specs['dataset_Path'].values[0]}", 'NOAA_GRID_extent.tiff')
                            if item in list_tif_file:
                                list_tif_file.remove(item)
                                list_tif_file.insert(0, item)
                            merging_rasters(list_tif_file, dst, gdf_tile_buff, crs)
                        else:
                            merging_rasters(list_tif_file, dst, gdf_tile, crs)
                                                
                else:
                    logger.info(f'dataset not tiled, use full dataset')
                    print('dataset not tiled, use full dataset')
                    tt=name+specs['extentions'].values[0]
                    dst = os.path.join(dataset_dir, f"{specs['dataset_Path'].values[0]}", tt)
                     
                if specs['extentions'].iloc[0]=='.csv':
                    logger.info(f'dataset {name} not in tif format converting using gdal_grid...')
                    desti=dst.replace('.csv', '.tif')
                    ##apply elevation filter to remove off terrain objects

                    ## TODO HARDCODED value of 78 but such datasets are only in LKO and downstream for the moment would need to change if going upstream (LKE, etc. 
                    shoal_df=pd.read_csv(dst, sep=';')
                    shoal_df=shoal_df.loc[shoal_df['Z']<=78]
                    shoal=fr'{dump_folder}\shoal.csv'
                    shoal_df.to_csv(shoal, sep=';', index=None)
                    csv_to_geotiff(shoal, 'X', 'Y', 'Z', desti, 5, 5, 32618)
                    dst=desti
                        
                ## reproject dataset tif and clip with tile extent               
                reproj_file = os.path.join(dump_folder, f'{t}_reproj.tif')
                reproj = reproj_raster(dst, crs, reproj_file)
                clipped_grid = os.path.join(dump_folder, f'{t}_clipped_{count}_grid.tif')
                clipping_raster(tile_xtent, reproj, clipped_grid)
                 
               
                if 'NONNA' in name or 'NOAA' in name:
                    ## But first NONNA needs to be clipped away from shoreline 
                    NONNA_extent=gpd.read_file(tile_buff)
                    if os.path.exists(extent_topo_buff20_file):                
                        extent_topo_buff20=gpd.read_file(extent_topo_buff20_file)
                        extent_topo_buff20=extent_topo_buff20.to_crs(crs)
                        print('....clip NONNA 20m away from topo datasets')
                        NONNA_extent['geometry']=NONNA_extent['geometry'].difference(extent_topo_buff20['geometry'])
                    NONNA_extent=NONNA_extent.dissolve()
                    NONNA_extent_file=fr'{dump_folder}\extent_NONNA_{t}_clipped20.shp'
                    NONNA_extent.to_file(NONNA_extent_file)
                    clipping_raster(NONNA_extent_file, reproj, clipped_grid)
                     
                # resample <1m res dataset to avoid unescesseray computation at the interpolation
                if res<1:
                    resample_raster(res, 1, clipped_grid, clipped_grid)
                    res=1
     
                # find conversion value to IGLD85
                conv=to_IGLD85(datum, gdf_grid_overview, cd_file, clipped_grid, t)
     
                #change dataset to array
                export=raster_to_XYZ_numpy(clipped_grid)
     
                # apply conversion
                arr=export[:,2]+conv
                if datum!='CD' or t not in slope_tiles :
                    arr=np.where(arr==conv, 0, arr)
                # remove outliers
                arr=np.where(arr>1000, 0, arr)
                arr=np.where(arr<-1000, 0, arr)

                if len(np.unique(arr))==1 and np.unique(arr)[0]==0:
                    print('nothing left, skipping..')
                    continue
                
                ## switch back dataset array to raster
                clipped_grid_conv = os.path.join(dump_folder, f'{t}_clipped_{count}_grid_conv.tif')
                with rio.open(clipped_grid) as src:
                    ras_data = src.read()
                    ras_meta = src.profile
                    ras_meta.update({"driver": "GTiff", "nodata":0}
                )
                    
                arr=arr.reshape(ras_data.shape)
                with rio.open(clipped_grid_conv, 'w', **ras_meta) as dst:
                    dst.write(arr)
                     
                ## clip dataset raster with dataset extent that was previoulsy determined
                clipped = os.path.join(dump_folder, f'{t}_clipped_{count}.tif')
     
                ##interpolate sparse bathy datasets w DELAUNEY via gdal_grid  
                if 'NONNA' in name or 'NOAA' in name :
                    print('interpolating, sparse bathy....')
                    array_nonna=raster_to_XYZ_numpy(clipped_grid_conv)
                    array_nonna=array_nonna[~np.isnan(array_nonna).any(axis=1)]
                    array_nonna=array_nonna[array_nonna[:,2] != 0]
                    df_nonna=pd.DataFrame(array_nonna, columns=['XVAL', 'YVAL', 'ZVAL'], index=None)
                    csv=fr'{dump_folder}\nonna.csv'
                    df_nonna.to_csv(csv, sep=';', index=None)
                    dst_tif=csv.replace('.csv', '.tif')

                    csv_to_geotiff(csv, 'XVAL', 'YVAL', 'ZVAL', dst_tif, 10, 10, crs)
                    nonna_ext_poly=os.path.join(dump_folder, f'nonna_ext_poly.shp')                       
                    reclass_and_polygonize(dst_tif, nonna_ext_poly)
                    nonna_ext_gdf=gpd.read_file(nonna_ext_poly) 
                    
                    gdf_extent=gpd.read_file(extent_file)
                    gdf_extent=gdf_extent.explode()
                    to_clip=fr'{dump_folder}\nonna_clip.shp'
                    buffer_w_ogr(gdf_extent, crs, -8, to_clip)  
                    gdf_extent=gpd.read_file(to_clip)
                     
                    gdf_extent=gdf_extent.clip(gdf_tile)
                    gdf_extent=gdf_extent.explode()
                     
                    gdf_extent=remove_small_parts_and_holes(gdf_extent, 100000)
                    gdf_extent=gdf_extent.dissolve()
                    gdf_extent=gdf_extent.clip(nonna_ext_gdf)
                    if len(gdf_extent)==0:
                        print('skipping, nothing left of this dataset')
                        continue 
                    gdf_extent.to_file(to_clip)
                    clipping_raster(to_clip, dst_tif, clipped)
                    extent_file=to_clip
     
                else:   
                    clipping_raster(extent_file, clipped_grid_conv, clipped)
                 
     
                ##making sure that after the trim and buffer there is still dataset to process   
                gdf_dataset=gpd.read_file(extent_file)
                gdf_dataset=gdf_dataset.dissolve()
                total_area=gdf_dataset['geometry'].area.sum()
                if data_type=='bathy' and total_area<100000 and 'NONNA' not in name:
                    logger.info(fr'not much left of this dataset after prioritization and buffering... skipping')
                    logger.info('\n********************************************************\n')
                    print('not much left of this dataset... skipping')
                    continue
     
                gdf_dataset['dataset']=name
                gdf_dataset['type']=specs['type'].values[0]
                gdf_dataset['priority']=specs['priority'].values[0]
                gdfs.append(gdf_dataset)
     
                ## need to buffer topo by 20m to clip NONNA away from shoreline                     
                if data_type=='topo':
                    gdfs_topo.append(gdf_dataset)
                  
                if count ==len(topo):
                    extents_gdf_topo=gpd.pd.concat(gdfs_topo)
                    extents_gdf_topo=extents_gdf_topo[['dataset', 'type', 'priority', 'geometry']]
                    extent_topo=extents_gdf_topo.loc[extents_gdf_topo['type']=='topo']
                    extent_topo=extent_topo.dissolve()
                    extent_topo_file=fr'{res_folder}\extent_topo_{t}.shp'
                    extent_topo.to_file(extent_topo_file)
                    extent_topo_buff20_file=fr'{res_folder}\extent_topo_{t}_buff20.shp'
                    buffer_w_ogr(extent_topo, crs, 20, extent_topo_buff20_file)
                      
                if datum!='CD':
                    logger.info(f'original vertical datum is {datum}, conversion value to IGLD85 is {conv}')
                else:
                    logger.info(f'original vertical datum is {datum}, conversion values to IGLD85 are from {np.min(conv)} to {np.max(conv)} ')
     
                print('dataset to df....')
                export2=raster_to_XYZ_numpy(clipped)
                df=pd.DataFrame(export2, columns=['XVAL', 'YVAL', 'ZVAL'])
                df=df.loc[df['ZVAL']!=0]
                if datum!='CD' or t not in slope_tiles:
                    df=df.loc[df['ZVAL']!=conv]
                else:
                    pass
                df['ZVAL']=df['ZVAL'].round(2)
     
                ### stats of overlapping portion with previous dataset of same type (bathy/topo) datasets can explain steps between datasets and provide adjustment values tp minimize such steps ####
                if count > 1 and previous_data_type== data_type and data_type!= 'topo' :
                    if data_type=='bathy':
                        count_dataset_type=count_bathy
                    else:
                        count_dataset_type=count_topo
                    print('computing overlapping stats....')
                    count_dd=0
                    for dd in range(count_dataset_type-1):
                        count_dd+=1
                        path_tif=fr'{dump_folder}\{t}_clipped_{count-count_dd}_grid_conv.tif'
                        path_dd=fr'{dump_folder}\extent_{t}_dataset_{count-count_dd}_full.shp'
                        if os.path.exists(path_tif):
                            path_prev=path_dd
                            break
                        else:
                            pass
                    set_nb=count-count_dd
                    path_actual = os.path.join(dump_folder, f'extent_{t}_dataset_{count}_full.shp')
                    tif_actual = os.path.join(dump_folder, f'{t}_clipped_{count}_grid_conv.tif')
                    tif_prev = path_tif
                    if os.path.exists(path_tif):
                        set_nb=count-count_dd
                        med_error=overlappingstats(path_actual, path_prev, dump_folder, res_folder, tif_actual, tif_prev, count, set_nb)
                        logger.info(f'\n**WARNING** correction of {med_error*-1}m applied to this dataset based on median difference with overlapping points of previous dataset of same type\n')
                    else:
                        pass
                else:
                    pass
     
                ## removing med error with previous dataset if found one (to dataframe)
                df['ZVAL']=df['ZVAL']-med_error
                dfs.append(df)
                                
                ### to have the 'raw' datasets .csv  before nonna_10 data are interpolated with gdal_grid 
                if 'NONNA' in name:
                    df_real=df_nonna
                    df_real=df_real.loc[df_real['ZVAL']!=0]
                    df_real['ZVAL']=df_real['ZVAL'].round(2)
                    df_real['ZVAL']=df_real['ZVAL']-med_error
                    dfs_real.append(df_real)
                else:
                    dfs_real.append(df)
                logger.info('\n********************************************************\n')
     
                ## removing med error with previous dataset if found one (to the raster)
                dst_tif=fr'{dump_folder}\{t}_dataset_{count}.tif'
                with rio.open(clipped) as src:
                    rast_data = src.read()
                    rast_meta = src.profile
                    rast_meta.update({"driver": "GTiff", "nodata":0, "dtype":"float32"}
                )
                arr2=rast_data-med_error
                ## nodata 0 are now == to -med_error, so need to remove them
                arr2=np.where(arr2!=med_error*-1, arr2, 0)
                arr2=arr2.reshape(rast_data.shape)
                with rio.open(dst_tif, 'w', **rast_meta) as dst:
                    dst.write(arr2)  
                if os.path.exists(fr'{dump_folder}\{t}_dataset_{count-1}.tif'):
                    specs_previous=df_specs.loc[df_specs['DATASET']==datasets_sorted[count-2]]
                    previous_data_type = specs_previous['type'].values[0]
                    print(f'PREVIOUS DAT TYPE IS : {previous_data_type}')

                ## final_resoltuion is generally [1]:
                resolution=final_resolutions[0]
                ## resampling tif to same resolution before merging
                resamp_tif=dst_tif.replace('.tif', f'_resamp_{resolution}.tif')
                cmd_resamp=fr'{gdal_path}\gdalwarp.exe -tr {resolution} {resolution} -r cubicspline -overwrite {dst_tif} {resamp_tif}'
                os.system(cmd_resamp)
                 
                ## merging raster of same type with feathering mode
                feathered_tif=fr'{dump_folder}\{t}_feathered_{data_type}_{resolution}.tif'
                # reset count_feather to 0 when we change of dataset type 
                if previous_data_type != data_type:
                    count_feather=0
                # if not the first dataset and previous set is of same type (bathy ou topo)       
                if count > 1 and previous_data_type == data_type:
                    if data_type=='bathy':
                        count_dataset_type_f=count_bathy
                    else:
                        count_dataset_type_f=count_topo
                         
                    # if a feathering for that type of data has already been done we use it
                    if count_feather>0:
                        tif1=feathered_tif
                     
                    # if not we use the previous dataste of same type
                    else:
                        count_dd_f=0
                        for dd in range(count_dataset_type_f-1):
                            count_dd_f+=1
                            tif_resamp=fr'{dump_folder}\{t}_dataset_{count-count_dd_f}_resamp_{resolution}.tif'
                            if os.path.exists(tif_resamp):
                                tif1=tif_resamp
                                break
                            else:
                                pass
                                   
                    tif2=resamp_tif
                     
                    ## merging raster in feathering mode                
                    wbt.mosaic_with_feathering(
                    tif1, 
                    tif2, 
                    feathered_tif, 
                    method="cc", 
                    weight=4.0)
                    count_feather+=1
                            
                else:
                    print('no need for feathering...')
                    shutil.copy2(resamp_tif, feathered_tif)
                     
                previous_data_type=data_type
             
            ##export extent of each datset used in the DEM in one shp
            logger.info(f'Extent of each dataset exported to {res_folder}\datasets_extent_{t}.shp')
             
            if len(gdfs)>0:
                extents_gdf=gpd.pd.concat(gdfs)
                extents_gdf=extents_gdf[['dataset', 'type', 'priority', 'geometry']]
                extent_topo=extents_gdf.loc[extents_gdf['type']=='topo']
                extent_topo=extent_topo.dissolve()
                extent_topo=extent_topo.clip(gdf_tile)
                extent_topo=extent_topo.dissolve()
                extent_topo_file=fr'{res_folder}\extent_topo_{t}.shp'
                extent_topo.to_file(extent_topo_file)
                extents_gdf=clean_gdf(extents_gdf)
                new_extent=[]
                for row in extents_gdf.itertuples():
                    if row.type=='bathy':
                        df_bathy=pd.DataFrame([row], columns=['index', 'dataset', 'type', 'priority', 'geometry'])
                        new_bathy=gpd.GeoDataFrame(df_bathy, crs=extents_gdf.crs, geometry=df_bathy.geometry)
                        new_bathy=new_bathy[['dataset', 'type', 'priority', 'geometry']] 
                         
                        if len(extent_topo)>0:
                            new_bathy['geometry']=new_bathy['geometry'].difference(extent_topo['geometry'])
                        else:
                            new_bathy=new_bathy

                        new_bathy=clean_gdf(new_bathy)
                        new_bathy.to_file(fr'{dump_folder}\new_bathy_test.shp')
                        new_extent.append(new_bathy)
                         
                    else:
                        df_topo=pd.DataFrame([row], columns=['index', 'dataset', 'type', 'priority', 'geometry'])
                        new_topo=gpd.GeoDataFrame(df_topo, crs=extents_gdf.crs, geometry=df_topo.geometry)
                        new_topo=new_topo[['dataset', 'type', 'priority', 'geometry']]
                        new_topo=clean_gdf(new_topo)
                        new_extent.append(new_topo)
                 
                new_gdf=gpd.pd.concat(new_extent)
                new_gdf=clean_gdf(new_gdf)
                new_gdf.to_file(fr'{dump_folder}\new_gdf.shp')
                gdf_tile=clean_gdf(gdf_tile)
                gdf_tile.to_file(fr'{dump_folder}\gdf_tile.shp')
                extents_gdf=new_gdf
                extents_gdf=remove_small_parts_and_holes(extents_gdf, 1000)
                extents_gdf.to_file(fr'{res_folder}\datasets_extent_{t}.shp')
     
                ## uses interpolated NONNA dataset to put in the .csv ready for interpolation
                result = pd.concat(dfs)
                resfile_int=fr'{res_folder}\all_datasets_{t}_interpolated.csv'
                 
                if cfg.write_interpolated_data==True:
                    print('writing interpolated data to file...')
                    result.to_csv(resfile_int, sep=';', index=None)
                    logger.info(fr'All combined datasets ready for interpolation on regular grid are exported to {resfile_int}')
                 
                if cfg.write_raw_data==True:
                ## uses uninterpolated NONNA dataset to put in the 'raw' data .csv
                    results_real=pd.concat(dfs_real)
                    resfile=fr'{res_folder}\all_datasets_{t}.csv'
                    print('writing uninterpolated data to file...')
                    results_real.to_csv(resfile, sep=';', index=None)
                    logger.info(fr'All combined "raw" datasets are exported to {resfile}')
                 
            else:
                logger.info(fr'#################   !!!! NO DATA AVAILABLE FOR THIS TILE !!! #################')
                continue
     
            ##use topo and bathy merged tif and clip them with each type dataset extent
            final_res=final_resolutions[0]                
            topo_tif=fr'{dump_folder}\{t}_feathered_topo_{final_res}.tif'
            topo_tif_clipped=fr'{dump_folder}\{t}_feathered_topo_{final_res}_clipped.tif'
            if os.path.exists(topo_tif):
                clipping_raster(extent_topo_file,  topo_tif, topo_tif_clipped)
       
            bathy_tif=fr'{dump_folder}\{t}_feathered_bathy_{final_res}.tif'
            bathy_tif_clipped=fr'{dump_folder}\{t}_feathered_bathy_{final_res}_clipped.tif'
             
            extent_non_topo=gdf_tile
            if len(extent_topo)>0:
                extent_non_topo['geometry']=extent_non_topo['geometry'].difference(extent_topo['geometry'])
                 
            extent_non_topo_file=fr'{res_folder}\extent_non_topo.shp'   
            buffer_w_ogr(extent_non_topo, crs, 1,  extent_non_topo_file)
            extent_non_topo=gpd.read_file(extent_non_topo_file)
            extent_non_topo=clean_gdf(extent_non_topo)
            extents_gdf=remove_small_parts_and_holes(extents_gdf, 1000)
            extent_non_topo=extent_non_topo.dissolve()
            if len(extent_non_topo)>0:
                if str(extent_non_topo['geometry'].iloc[0]).split(' ')[0]=='GEOMETRYCOLLECTION':
                    pass
                else:
                    extent_non_topo_file=fr'{res_folder}\extent_non_topo.shp'
                    extent_non_topo.to_file(extent_non_topo_file)
            else:
                pass              
            if os.path.exists(bathy_tif):
                clipping_raster(extent_non_topo_file,  bathy_tif, bathy_tif_clipped)
                                
            ## merging topo and bathy DEMs
            print(f'merging different type DEMs and clipping w. tile extent...')
            merged_DEM=fr'{dump_folder}\{t}_{final_res}m_DEM_idw_both_types.tif'
            if os.path.exists(topo_tif_clipped) and os.path.exists(bathy_tif_clipped):
                if cfg.final_feathering:
                    wbt.mosaic_with_feathering(
                        topo_tif_clipped, 
                        bathy_tif_clipped, 
                        merged_DEM, 
                        method="cc", 
                        weight=1.0)
                else:
                    cmd_final_merge=f'python {gdal_scripts}\gdal_merge.py -o {merged_DEM} {bathy_tif_clipped} {topo_tif_clipped}'
                    os.system(cmd_final_merge)
     
            elif os.path.exists(topo_tif_clipped):
                print('only topo DEM...')
                shutil.copy2(topo_tif_clipped, merged_DEM)
            else:
                print('only bathy DEM...')
                shutil.copy2(bathy_tif_clipped, merged_DEM)
             
            ## fill small gaps in DEM
            merged_dem_nodat=fr'{dump_folder}\{t}_{final_res}m_DEM_idw_both_types_nodat.tif'
            merged_DEM_filled=fr'{dump_folder}\{t}_{final_res}m_DEM_idw_both_types_filled.tif'
            cmd_nodat=fr'{gdal_path}\gdalwarp.exe -t_srs EPSG:{crs} -ot Float32 -dstnodata -32768 {merged_DEM} {merged_dem_nodat}'
            cmd_fill=fr'python {gdal_scripts}\gdal_fillnodata.py -md 1000 {merged_dem_nodat} {merged_DEM_filled}'
            os.system(cmd_nodat)
            os.system(cmd_fill)

            ## clip with tile extent
            clipped_raster=fr'{res_folder}\{t}_{final_res}m_DEM_idw.tif'
            clipping_raster(tile_xtent, merged_DEM_filled, clipped_raster)
            dem_no_data_and_clip(crs, clipped_raster, clipped_raster, dump_folder, res_folder, final_res, gdal_scripts, gdal_path, tile_xtent)               
                 
            ##apply feature_preserving_smoothing filter to final dem
            filtered=f'{dump_folder}\{t}_{final_res}m_DEM_idw_filtered.tif'            
            wbt.feature_preserving_smoothing(
                clipped_raster, 
                filtered, 
                filter=11, 
                norm_diff=5.0, 
                num_iter=1, 
                max_diff=0.5, 
                zfactor=None)
             
            roto=f'{dump_folder}\{t}_{final_res}m_DEM_idw_roto.tif'
            wbt.remove_off_terrain_objects(
                filtered, 
                roto, 
                filter=10, 
                slope=40.0)
             
            logger.info(fr'*** Feature_preserving_smoothing filter applied to final DEM  ***')
             
            final_real=f'{res_folder}\{t}_{final_res}m_DEM_idw_filtered.tif'
            dem_no_data_and_clip(crs, roto, final_real, dump_folder, res_folder, final_res, gdal_scripts, gdal_path, tile_xtent)

            gdal_info = gdal.Info(final_real)
             
            logger.info(f'****************************\n gdal_info of {final_res}m DEM is : \n{gdal_info}\n ********************************')
            logger.info(fr'*** DEM at {final_res}m resolution can be found at {res_folder}\{t}_{final_res}m_DEM_idw_filetred.tif ***')
             
            ## create a hillshade overview for fast vizualisation
            hillshadeoverview(final_real, -32768)                
            logger.info(fr'*** hillshade overview created for faster visualization   ***')
             
            ## resample to create a 10m res dem 
            dem_10m=fr'{res_folder}\{t}_{cfg.resamp_res}m_DEM_idw_filtered.tif'   
            cmd_resamp_10=fr'{gdal_path}\gdalwarp.exe -tr {cfg.resamp_res} {cfg.resamp_res} -r cubicspline -overwrite {final_real} {dem_10m}'
            os.system(cmd_resamp_10)
            logger.info(fr'*** DEM at {cfg.resamp_res}m resolution can be found at {res_folder}\{t}_10m_DEM_idw_filetred.tif ***')   
 
            if clean_dump:
                for filename in os.listdir(dump_folder):
                    file_path = os.path.join(dump_folder, filename)
                    try:
                        if os.path.isfile(file_path) or os.path.islink(file_path):
                            os.unlink(file_path)
                        elif os.path.isdir(file_path):
                            shutil.rmtree(file_path)
                    except Exception as e:
                        print(f'Failed to delete {file_path}. Reason: {e}')
 
            print('################################ DEM CREATION COMPLETED WITHOUT ERROR #########################################')
                         
            if cfg.apply_correction:
                if t in cfg.tiles_w_corr:
                    print('################################ APPLYING WETLAND CORRECTION ALGORITHM #########################################')
                    datasets_corrected=Wetland_correction_seperate.execute(t, cfg, res_folder)
                    logger.info(fr'WETLAND CORRECTION MODEL SUCCESFULLY APPLIED TO {datasets_corrected}')

            print('################################ modify dataset extent #########################################')
            
            liste=[]
            liste.append(t) 
            dataset_extent_modif.execute(liste, cfg, cfg.res_folder_name, cfg.workdir, res_folder)
            logger.info(fr'Matadata info added to dataset extent file')
            print('################################ dataset extent modification completed without error #########################################')
             
            liste=[]
            liste.append(t) 
            cfg.final_resolutions[0]
            
            print('masking')
            
            MASK_DEM_EXEC.execute(liste, cfg, res_folder, dump_folder, cfg.final_resolutions[0])
            logger.info(fr'DEM masked based on nodata file')
            print('################################ DEM masked with no_data mask #########################################') 

            print('################################COMPLETED WITHOUT ERROR #########################################')

            list_worked.append(t)
            ## compile sucessfully processed tiles in a .csv
            df_worked=pd.DataFrame(list_worked, columns=['tiles'])
            df_worked.to_csv(fr'{workdir}\results\list_worked.csv', sep=';', index=None)
    
        except Exception as e:
            print("#########################   ERROR   ###############################")
            print(e)
            logger.info("#########################   ERROR   ###############################")
            error_msg=logger.exception(e)
            logger.info(error_msg)
            
            ##clean dump folder even if there is error in process
            if clean_dump:
                for filename in os.listdir(dump_folder):
                    file_path = os.path.join(dump_folder, filename)
                    try:
                        if os.path.isfile(file_path) or os.path.islink(file_path):
                            os.unlink(file_path)
                        elif os.path.isdir(file_path):
                            shutil.rmtree(file_path)
                    except Exception as e:
                        print(f'Failed to delete {file_path}. Reason: {e}')
                    
            ## compile tiles with error in a .csv
            file_handler.close()
            logger.handlers.pop()
            info=[t, e]
            list_bugged.append(info)
            df_bugg=pd.DataFrame(list_bugged, columns=['tiles', 'error_msg'])
            df_bugg.to_csv(fr'{workdir}\results\list_bugged.csv', sep=';', index=None)
    
    quit()



