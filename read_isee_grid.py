import pandas as pd
import os

path=r'H:\Projets\GLAM\DATA_ISEE\Grille_ISEE\GLAM_ISEE_Tiles\DEM_4.2_regular_grid_10m_20240314\feather'

liste=os.listdir(path)

for l in liste:
    src=os.path.join(path, l)

    df=pd.read_feather(src)
    
    print(len(df['Mask'].unique()))
    
    if len(df['Mask'].unique()) == 1:
        print(l, df['Mask'].unique())
    
quit()