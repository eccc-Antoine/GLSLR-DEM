import os
import shapely
from shapely.validation import make_valid
from shapely.geometry import Polygon, MultiPolygon
import numpy as np
import geopandas as gpd

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


def execute(tiles, cfg, res_folder, dump_folder, final_res):

    for t in tiles:
        DEMS=[fr"{res_folder}\{t}_{final_res}m_DEM_idw_filtered.tif", fr"{res_folder}\{t}_{cfg.resamp_res}m_DEM_idw_filtered.tif", fr"{res_folder}\{t}_{final_res}m_dem_wetland_corrected.tif", fr"{res_folder}\{t}_{cfg.resamp_res}m_dem_wetland_corrected.tif" ]
        
        clipper=fr"{res_folder}\datasets_extent_{t}_details.shp"
        
        gdf=gpd.read_file(clipper)
        
        gdf_clean=clean_gdf(gdf)
        
        gdf_clean=gdf_clean.dissolve()
        

        gdf_clean['geometry']=gdf_clean['geometry'].buffer(1)
        
       
        
        clipper_clean=fr'{dump_folder}\{t}_no_data_clean.shp'
        
        gdf_clean.to_file(clipper_clean)
        
        
        for D in DEMS:
            
            dst=D.replace('.tif', '_masked.tif')
            if os.path.exists(D):
                cmd=fr'{cfg.gdal_path2}\gdalwarp.exe -cutline {clipper_clean} -crop_to_cutline {D} {dst}'
                
                if not os.path.exists(dst): 
                    os.system(cmd)
                
                else:
                    os.remove(dst)
                    os.system(cmd)     
            
            else:
                print(fr'{D}, dont exists...')


if __name__ == '__main__':
    
    lot_of_tiles=[165]
    
    for tiles in lot_of_tiles:
        tiles=[tiles]
    
    
        res_folder=fr'F:\DEM_GLAMM\DEM_CREATION_CLEAN\results\{tiles[0]}_V4_2'
        dump_folder=r'F:\DEM_GLAMM\DEM_CREATION_CLEAN\dump'
        final_res=10
        import CFG_DEM_CREATION as cfg
        execute(tiles, cfg, res_folder, dump_folder, final_res)
             
    