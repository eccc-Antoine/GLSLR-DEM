import os
import rasterio
import numpy as np

tiles=list(range(79, 492))

tiles_to_remove=[160, 161, 138, 125, 85, 79, 80, 81, 333]

tiles=[x for x in tiles if x not in tiles_to_remove]

folder=r'F:\DEM_GLAMM\DEM_CREATION_FINAL\results'

for t in tiles:
    
    
    masked=fr"F:\DEM_GLAMM\DEM_CREATION_FINAL\results\{t}_V4_3\{t}_10m_DEM_idw_filtered_masked.tif"
    
    wetlands=fr"F:\DEM_GLAMM\DEM_CREATION_FINAL\results\{t}_V4_3\{t}_10m_dem_wetland_corrected_masked.tif"
    
    if os.path.exists(masked):
        with rasterio.open(masked) as src:
            image=src.read(1)
            
            values=np.unique(image)
            
            if len(values)==1:
                print(values)
                print(f'{t} as problem with masked file!!')
        
        pass
    else:
        print(t)
        print('NO masked file')


    if os.path.exists(wetlands):
        with rasterio.open(wetlands) as src:
            image=src.read(1)
            
            values=np.unique(image)
            
            if len(values)==1:
                print(values)
                print(f'{t} as problem with wetland file!!')
        #print('wetlands file')
        #pass
    
    else:
        pass
        #print(t)
        
        
    if not os.path.exists(wetlands) and not os.path.exists(masked) :
        print(fr'{t} PROBLEM')
        
quit()

    

    

