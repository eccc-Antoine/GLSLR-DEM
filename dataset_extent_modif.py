import os
import pandas as pd
import geopandas as gpd
import shapely
from shapely.validation import make_valid
import CFG_DEM_CREATION as cfg

''' 
Script to:
    - Add metadata information in dataset extent file 
    - Clip dataset extent based on no-data masks (maks were created with semi-automated routines, would need better attention)
    - Computes an indicatives reliability score for each tile

Coded in the GLAM Expedited review context, may need several modification if
used for another study area 

Author: Antoine Maranda (antoine.maranda@ec.gc.ca)
Environment and Climate Change Canada, National Hydrologic Services, Hydrodynamic and Ecohydraulic Section
'''


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

def execute(tiles, cfg, version, workdir, res_folder):
    tiles_to_process=[]
    for t in tiles:
        gdf_extent=gpd.read_file(fr"{res_folder}\datasets_extent_{t}.shp")
        if len(gdf_extent)==0:
            tiles_to_process.append(t)
    gdf_tiles=gpd.read_file(fr"{workdir}\GLAM_ISEE_Tiles\Tuile_final_w_conversions.shp")
    dct_crs={18:'epsg:32618', 17:'epsg:32617'}
    
    for t in tiles_to_process:  
        gdf_t=gdf_tiles.loc[gdf_tiles['tile']==t] 
        utm=gdf_t['UTM'].iloc[0] 
        crs_tile=dct_crs[utm]    
        gdf_t=gdf_t.to_crs(crs_tile)
        gdf_t['dataset']='NOAA_NONNA'
        gdf_t['type']='bathy'
        gdf_t['priority']=70
        gdf_t['area']=gdf_t.area
        gdf_t=gdf_t[['dataset', 'type', 'priority', 'area', 'geometry']]   
        gdf_t.to_file(fr'{res_folder}\datasets_extent_{t}.shp')
    
    list_of_sets=[]  
    for t in tiles:
        spatial_available=False
        
        de=gpd.read_file(fr"{res_folder}\datasets_extent_{t}.shp")
        
        spatial_LKO=fr"{workdir}\USACE_metadata\LKO\spatial_meta_shp\{t}_1m_DEM_Spatial_Metadata.shp"
        
        spatial_USL=fr"{workdir}\USACE_metadata\USL\spatial_meta_shp\{t}_1m_DEM_Spatial_Metadata.shp"
        
        if os.path.exists(spatial_LKO):
            
            spatial_available=True
            meta_shp=gpd.read_file(spatial_LKO)
            meta_df=pd.read_csv(fr"{workdir}\USACE_metadata\LKO\VAT_CSV\{t}_spatial_meta.csv", sep=';')
            print(fr'spatial metadata available processing:{t}')
        
        if os.path.exists(spatial_USL):
            spatial_available=True
            meta_shp=gpd.read_file(spatial_USL)
            meta_df=pd.read_csv(fr"{workdir}\USACE_metadata\USL\VAT_CSV\{t}_spatial_meta.csv", sep=';')
            print(fr'spatial metadata available processing:{t}')
        
        if spatial_available:
    
            de=de.dissolve(by='dataset', as_index=False)
            de['area']=de.area
            de_filt=de.loc[de['area']>=1000]
            
            if len(de_filt) == 0:
                de.to_file(fr"{cfg.dump_folder}\datasets_extent_{t}_2.shp")
                continue
            
            de=de_filt.sort_values(by=['priority'])
            de['all_prior']=de['priority']
            de['t_priority']=(de.index+1).astype(int)
            de=de[['dataset', 'type', 't_priority', 'all_prior', 'area', 'geometry', ]]
            sets=de['dataset'].unique()
            last=de['dataset'].iloc[-1]
            usace_sets=[]
            
            for s in sets:
                if 'USACE' in s:
                    usace_sets.append(s)
                else:
                    continue
            set_poly=de.loc[de['dataset'].isin(usace_sets)]
            set_poly=clean_gdf(set_poly)
            meta_shp=clean_gdf(meta_shp)
            meta_clip=meta_shp.clip(set_poly)
            meta_clip=meta_clip.dissolve(by='DN', as_index=False)
            meta_clip['dataset']='foo'
            data_meta=[]
            
            count=-1
            for y in range(len(meta_clip)): 
                count+=1 
                m=meta_clip.iloc[[y]]
                number=m['DN'].unique()[0]
                m['dataset']=meta_df['dataset'].loc[meta_df['values']==number].unique()[0]
                m['type']='topo'
                m['all_prior']=set_poly['all_prior'].min()
                m['t_priority']=set_poly['t_priority'].min()+count
                m['area']=m['geometry'].area
                m=m[['dataset', 'type', 't_priority', 'all_prior', 'area', 'geometry', ]]
                m=gpd.GeoDataFrame(m, crs=meta_shp.crs, geometry=m['geometry'])                
                m=clean_gdf(m)                
                m=m.dissolve(by=['dataset'], as_index=False)
                de=de.append(m)
                set_name=m['dataset'][0]
                if set_name not in list_of_sets:
                    list_of_sets.append(set_name)
            de['t_priority'].loc[de['dataset']==last]=de['t_priority'].loc[de['dataset']==last]+(len(meta_clip)-1)

            for usa in usace_sets:
                de = de.loc[de['dataset'] != usa]
            gdf_de=gpd.GeoDataFrame(de, crs=de.crs, geometry=de['geometry'])
            
            noaa_grid=set_poly
            noaa_grid=noaa_grid.dissolve(by=['dataset'], as_index=False)
            noaa_grid=noaa_grid[['dataset', 'type', 't_priority', 'all_prior', 'area', 'geometry']]
            gdf_de=gdf_de.reset_index()
            gdf_de=gdf_de[['dataset', 'type', 't_priority', 'all_prior', 'area', 'geometry']]
            noaa_grid=noaa_grid.dissolve()
            gdf_de_diss=gdf_de.dissolve()
            noaa_grid['geometry']=noaa_grid['geometry'].difference(gdf_de_diss['geometry'])
            noaa_grid=noaa_grid.dissolve()            
            noaa_grid['dataset']='NOAA_GRID'            
            noaa_grid['type']='bathy'            
            noaa_grid['all_prior']=70            
            noaa_grid['t_priority']=gdf_de['t_priority'].max()+1                    
            gdf_de=gdf_de.append(noaa_grid)            
            gdf_de=clean_gdf(gdf_de)            
            gdf_de=gdf_de.dissolve(by='dataset', as_index=False)    
            gdf_de['area']=gdf_de.area.astype(int)            
            gdf_de.to_file(fr"{cfg.dump_folder}\datasets_extent_{t}_2.shp")
                    
        else:
            print(fr'no spatial metadata for tile {t}')
    
    for t in tiles:
        de_file=fr"{cfg.dump_folder}\datasets_extent_{t}_2.shp"        
        if not os.path.exists(de_file):
            de_file=fr"{res_folder}\datasets_extent_{t}.shp"        
        de=gpd.read_file(de_file)        
        noaa=fr"{cfg.nodata_folder}\extent_NOAA.shp"            
        print(fr'spatial metadata available processing:{t}')    
        noaa_shp=gpd.read_file(noaa)
        de=de.dissolve(by='dataset', as_index=False)
        if de_file==fr"{res_folder}\datasets_extent_{t}.shp":        
            de['area']=de.area            
            de_filt=de.loc[de['area']>=1000] 
            
            if len(de_filt)==0:
                de.to_file(fr"{cfg.dump_folder}\datasets_extent_{t}_3.shp")
                continue

            de=de_filt.sort_values(by=['priority'])            
            de['all_prior']=de['priority']        
            de['t_priority']=(de.index+1).astype(int)            
        de=de[['dataset', 'type', 't_priority', 'all_prior', 'area', 'geometry', ]]            
        sets=de['dataset'].unique()            
        last=de['dataset'].iloc[-1]
        noaa_nonna_sets=[]
        
        for s in sets:
            if 'NOAA_NONNA' in s:
                noaa_nonna_sets.append(s)
            else:
                continue
            
        if len(noaa_nonna_sets)>0:
            set_poly=de.loc[de['dataset'].isin(noaa_nonna_sets)]            
            crs=set_poly.crs            
            set_poly=clean_gdf(set_poly)
            noaa_shp=clean_gdf(noaa_shp)            
            noaa_shp=noaa_shp.to_crs(crs)            
            noaa_clip=noaa_shp.clip(set_poly)            
            noaa_clip=noaa_clip.dissolve()
            noaa_clip['dataset']='NOAA_GRID'            
            noaa_clip['type']='bathy'            
            noaa_clip['all_prior']=70        
            noaa_clip['t_priority']=de['t_priority'].max()+1
            set_poly=set_poly.dissolve()            
            nonna=set_poly
            if len(noaa_clip)>0:           
                nonna['geometry']=set_poly['geometry'].difference(noaa_clip['geometry'])            
            else:
                nonna['geometry']=set_poly['geometry']
            
            nonna['dataset']='NONNA_10'            
            nonna['type']='bathy'            
            nonna['all_prior']=69        
            nonna['t_priority']=de['t_priority'].max()
  
            for usa in noaa_nonna_sets:              
                de = de.loc[de['dataset'] != usa]                
            gdf_de=gpd.GeoDataFrame(de, crs=de.crs, geometry=de['geometry'])
                
            if len(nonna)>0:                
                nonna=nonna[['dataset', 'type', 't_priority', 'all_prior','geometry']]                    
                nonna=gpd.GeoDataFrame(nonna, crs=de.crs, geometry=nonna['geometry'])                
                nonna=clean_gdf(nonna)                
                nonna=nonna.dissolve(by=['dataset'], as_index=False)                
                nonna=nonna[['dataset', 'type', 't_priority', 'all_prior','geometry']]                
                gdf_de=gdf_de.append(nonna)                
                
            if len(noaa_clip) >0:                
                noaa_clip=noaa_clip[['dataset', 'type', 't_priority', 'all_prior','geometry']] 
                noaa_clip=gpd.GeoDataFrame(noaa_clip, crs=de.crs, geometry=noaa_clip['geometry'])                
                noaa_clip=clean_gdf(noaa_clip)                
                noaa_clip=noaa_clip.dissolve(by=['dataset'], as_index=False)                
                gdf_de=gdf_de.append(noaa_clip)
        else:
            de=de[['dataset', 'type', 't_priority', 'all_prior', 'geometry', ]]
            gdf_de=de
        
        gdf_de=clean_gdf(gdf_de)        
        gdf_de=gdf_de.dissolve(by='dataset', as_index=False)    
        gdf_de['area']=gdf_de.area.astype(int)        
        gdf_de.to_file(fr"{cfg.dump_folder}\datasets_extent_{t}_3.shp")
    
    for t in tiles:
        aoi=gpd.read_file(cfg.AOI)
        gdf=gpd.read_file(fr"{cfg.dump_folder}\datasets_extent_{t}_3.shp")

        if len(gdf)==1:
            if 'NOAA_GRID' in list(gdf['dataset']):
                
                print(f'{t} only has noaa')
                
                crs=gdf.crs
                
                gdf_tile=gdf_tiles.loc[gdf_tiles['tile']==t]
                
                gdf_tile=gdf_tile.clip(aoi)
                
                gdf_tile=gdf_tile.to_crs(crs)
           
                gdf['geometry']=gdf_tile['geometry'].iloc[0]
                gdf.to_file(fr"{cfg.dump_folder}\datasets_extent_{t}_4.shp")
            else:
                gdf.to_file(fr"{cfg.dump_folder}\datasets_extent_{t}_4.shp")
        else:
            gdf.to_file(fr"{cfg.dump_folder}\datasets_extent_{t}_4.shp")

        
    for t in tiles:
        set=fr"{cfg.dump_folder}\datasets_extent_{t}_4.shp"
        
        if not os.path.exists(set):
            print(f'{t} extent dont exists!')
            continue
        
        else:       
            set_gdf=gpd.read_file(set)
            if len(set_gdf)==0:
                print(f'{t} set is empty!')
                continue
            
            else:
                columns=['dataset', 'type', 't_priority', 'all_prior', 'area', 'geometry']
                
                if list(set_gdf)!= columns:
                    print(f'{t} has wrong columns!')
                    continue
                
                else:
                    set_gdf=clean_gdf(set_gdf)
                    set_gdf=set_gdf.explode()
                    set_gdf['area']=set_gdf.area
                    set_gdf=set_gdf.loc[set_gdf['area']>1000]
                    set_gdf=set_gdf.dissolve(by='dataset', as_index=False)
                    set_gdf.to_file(fr"{cfg.dump_folder}\datasets_extent_{t}_6.shp")
    
    for t in tiles:
        no_data=fr"{cfg.nodata_folder}\{t}_elevation_data_gaps.shp"
        de=fr"{cfg.dump_folder}\datasets_extent_{t}_6.shp"
        if os.path.exists(no_data):
            
            de_gdf=gpd.read_file(de)
            
            de_gdf=clean_gdf(de_gdf)
            
            de_gdf_diss=de_gdf.dissolve()
            
            crs=de_gdf.crs
            
            no_data_gdf=gpd.read_file(no_data)
            
            no_data_gdf=clean_gdf(no_data_gdf)
            
            no_data_gdf=no_data_gdf.to_crs(crs)
            
            no_data_gdf=no_data_gdf.dissolve()
            
            de_gdf_diss['geometry']=de_gdf_diss['geometry'].difference(no_data_gdf['geometry'])
            
            de_gdf_diss=clean_gdf(de_gdf_diss)
            
            de_gdf=de_gdf.clip(de_gdf_diss)
            
            de_gdf=de_gdf.dissolve(by='dataset', as_index=False)
            
            de_gdf=clean_gdf(de_gdf)
            
        else:
            de_gdf=gpd.read_file(de)
            de_gdf=clean_gdf(de_gdf)
            
        de_gdf.to_file(fr"{cfg.dump_folder}\datasets_extent_{t}_7.shp")
    
    for t in tiles:
        sets=['T2014_USGS_GreatLakes_Phase2_L3', 'OLD_USACE_extent', 'T2016_USGS_test']
        
        for s in sets:  
            gdf=gpd.read_file(fr"{cfg.dump_folder}\datasets_extent_{t}_7.shp")
               
            list_set=list(gdf['dataset'])
    
            set_dct={'T2014_USGS_GreatLakes_Phase2_L3':'T2014_USGS_GreatLakesPhase2_L3', 'OLD_USACE_extent':'NOAA_GRID', 'T2016_USGS_test':'NY_FEMAR-R2-B2_2016'}
            
            if s in list_set:

                gdf['dataset'].loc[gdf['dataset']==s]=set_dct[s]
                
                gdf.to_file(fr"{cfg.dump_folder}\datasets_extent_{t}_7.shp")
                
    df=pd.read_csv(cfg.specs_file_details, sep=';')
    
    colonnes=['Name', 'type', 'Data_type', 'Authority', 'Preprocessing', 'acquisition_date',  'Data_Hyperlink', 'Metadata_Hyperlink', 'Description', 'reliability']
      
    for t in tiles:

        gdf=gpd.read_file(fr"{cfg.dump_folder}\datasets_extent_{t}_7.shp")
        
        gdf=gdf.dissolve(by='dataset', as_index=False)
        
        gdf=gdf.drop(columns=['type'])
        
        crs=gdf.crs
        
        datasets=gdf['dataset']
 
        for c in colonnes:
            gdf[c] = gdf.apply(lambda _: "", axis=1)
            for d in datasets:
                gdf[c].loc[gdf['dataset']==d]=df[c].loc[df['DATASET']==d].iloc[0]
        
        gdf=gdf[['Name', 'type', 't_priority', 'all_prior', 'Data_type', 'Authority', 'Preprocessing', 'acquisition_date', 'Data_Hyperlink', 'Metadata_Hyperlink', 'Description', 'reliability', 'geometry']]
        
        gdf['area']=gdf.area
        
        gdf['Preprocessing']=gdf['Preprocessing'].astype(str)
        
        total_area=gdf.area.sum()
        
        gdf['score']=gdf['reliability']*gdf['area']/total_area

        gdf['tile_score']=gdf['score'].sum()

        gdf = gdf.apply(pd.to_numeric, errors='ignore')
        
        gdf=gpd.GeoDataFrame(gdf, crs=crs, geometry=gdf.geometry)
        
        gdf.to_file(fr"{res_folder}\datasets_extent_{t}_details.shp")
