import os
import shutil

tiles=list(range(209, 492))


tiles_to_remove=[160, 161, 138, 125, 85,79, 80, 81, 82, 333]


tiles=[x for x in tiles if x not in tiles_to_remove]

for t in tiles:
    
    print(t)

    path1=fr"F:\DEM_GLAMM\DEM_CREATION_CLEAN\results\{t}_V4_2\{t}_10m_dem_wetland_corrected.tif"
    dst1=fr'H:\Projets\GLAM\DEM\V4_2_10m_nomask\{t}_10m_dem_wetland_corrected.tif'
    
    path2=fr"F:\DEM_GLAMM\DEM_CREATION_CLEAN\results\{t}_V4_2\{t}_10m_DEM_idw_filtered.tif"
    dst2=fr'H:\Projets\GLAM\DEM\V4_2_10m_nomask\{t}_10m_DEM_idw_filtered.tif'
    
    if os.path.exists(path1):
        print('wetland')
        shutil.copy(path1, dst1)
    elif os.path.exists(path2):
        shutil.copy(path2, dst2)
    else:
        print('problem!!')
        
quit()