import rasterio
import numpy as np

def rast_2_arr(path):
    with rasterio.open(path) as src:
        image= src.read(1)
        
        #print(image.shape)
        return image
    
def compare_2_rasters(new, prev, diff_path):
    
    with rasterio.open(new) as src:
        image= src.read()
        ras_meta = src.profile
    with rasterio.open(prev) as src2:
        image2= src2.read()
    
    diff=image-image2
    
    with rasterio.open(diff_path, 'w', **ras_meta) as dst:
        dst.write(diff)
    
    return diff
    

tiles=list(range(83, 492))

#tiles=list(range(487, 492))

tiles_to_remove=[160, 161, 138, 125, 85, 79, 80, 81, 333]

tiles=[x for x in tiles if x not in tiles_to_remove]

not_same_shape=[]
lot_diff=[]

for t in tiles:
    
    print(f'tile {t}')

    new_path=fr"F:\DEM_GLAMM\DEM_CREATION_FINAL\results\{t}_V4_3\{t}_1m_DEM_idw_filtered_masked.tif"
    
    prev_path=fr"F:\DEM_GLAMM\DEM_CREATION_CLEAN\results\{t}_V4_2\{t}_1m_DEM_idw_filtered_masked.tif"
    
    arr_new=rast_2_arr(new_path)
    
    arr_prev=rast_2_arr(prev_path)
    
    if arr_new.shape != arr_prev.shape:
        print(f'for tile {t} 2 rasters dont have same shape {arr_new.shape} vs {arr_prev.shape}')
        not_same_shape.append(t)
        continue
    
    #np.testing.assert_allclose(arr_new, arr_prev, rtol=1e-2, atol=1e-2, equal_nan=True)
    
    test1=np.allclose(arr_new, arr_prev, rtol=1e-1, atol=0, equal_nan=True)
    
    if not test1:
        
        test2=np.isclose(arr_new, arr_prev, rtol=1e-1, atol=0, equal_nan=True)
    
        qty=np.count_nonzero(test2 == False)
        
        print(t, qty)

        idx=list(zip(*np.where(test2 == False)))
        
        #print(idx)
        count=0
        for i in idx:
            #print(i[0])
            #print(arr_new[i[0], i[1]])
            #print(arr_prev[i[0], i[1]])
            new=arr_new[i[0], i[1]]
            if new > 78:
                continue
            
            prev=arr_prev[i[0], i[1]]
            diff=abs(new-prev)
            if diff<0.01:
                continue
            
            count+=1
        
        print(count)
        if count>1000:
            lot_diff.append(t)
            diff_path=fr"F:\DEM_GLAMM\DEM_CREATION_CLEAN\version_differences\42_43_1m_with_mask\{t}_diff_V4_3_V4_2_1m_masked.tif"
            
            diff=compare_2_rasters(new_path, prev_path, diff_path)
            
            #print(diff)
            
            qty2=np.count_nonzero(diff == 0)
            
            #print(qty2)
            
                #print('too picky!')
                #print(diff)
        
        #test3=np.testing.assert_allclose(arr_new, arr_prev, rtol=1e-4, atol=1e-4, equal_nan=True, verbose=True)
        
        #np.testing.assert_array_equal(arr_new, arr_prev, verbose=True)
print(not_same_shape)
print(lot_diff)           
quit()