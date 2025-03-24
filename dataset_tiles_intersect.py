import geopandas as gpd
import pandas as pd
import os, time, datetime
import pathlib
import glob

''' 
Script to perform intersection analysis between elevation datasets and ISEE tile 
For subsequent runs, the analysis is only performed for new or modified datasets, hence avoiding unnecesary computaion 

Author: Antoine Maranda (antoine.maranda@ec.gc.ca)
Environment and Climate Change Canada, National Hydrologic Services, Hydrodynamic and Ecohydraulic Section

'''

def dataset_tiles_intersect(data_file, tiles_file):
    gdf_data = gpd.read_file(data_file)
    crs_data = gdf_data.crs
    gdf_tiles = gpd.read_file(tiles_file)
    gdf_tiles = gdf_tiles.to_crs(crs_data)
    gdfs = []
    tiles = []
    for t in gdf_tiles['tile']:
        print(f'checking if dataset is intersecting with tile {t} and if so which part of it is intersecting')
        xmin, ymin, xmax, ymax = gdf_tiles.loc[gdf_tiles['tile'] == t].total_bounds
        sub = gdf_data.cx[xmin + 1:xmax - 1, ymin + 1:ymax - 1]
        if len(sub) == 0:
            pass
        else:
            sub['tile_grid'] = int(t)
            gdfs.append(sub)
            tiles.append(int(t))

    gdf_parts = pd.concat(gdfs)
    gdf_parts = gpd.GeoDataFrame(gdf_parts, crs=crs_data, geometry=gdf_parts['geometry'])
    return gdf_parts, tiles


def find_modfified_date(file):
    modified = os.path.getmtime(file)
    year, month, day, hour, minute, second = time.localtime(modified)[:-3]
    date = "%02d/%02d/%d %02d:%02d:%02d" % (day, month, year, hour, minute, second)
    return date


def main_intersect(tiles_file, workdir, previous_version, tile_intersect_folder, warning=True):
    affected_tiles = []
    if not os.path.exists(tile_intersect_folder):
        os.makedirs(tile_intersect_folder)
    liste_topo = list(pathlib.Path(fr'{workdir}\topo').glob('**/*.shp'))
    liste_bathy = list(pathlib.Path(fr'{workdir}\bathy').glob('**/*.shp'))
    liste = liste_topo + liste_bathy
    work_folder = workdir.split('\\')[-1]

    cols = ['dataset_extent', 'tiles', 'modified_date']
    if os.path.exists(previous_version):
        df_previous = pd.read_csv(previous_version, sep=';')
        previous_list = list(df_previous['dataset_extent'])
        df_previous['dataset_extent_relative'] = [file.split(work_folder)[1] for file in df_previous['dataset_extent']]
        previous_list = list(df_previous['dataset_extent_relative'])
    else:
        previous_list = []
    data = []
    for l in liste:
        l_path = str(l).split(work_folder)[1]

        dataset_name = str(l).split('\\')[-1]
        if str(l_path) in previous_list:
            print(f'dataset {str(l)} already there checking modified date...')
            date = find_modfified_date(l)

            if date == df_previous['modified_date'].loc[df_previous['dataset_extent_relative'] == str(l_path)].values:
                print(f'dataset {str(l)} already there and not modified since, nothing to do!')
                tiles = df_previous['tiles'].loc[df_previous['dataset_extent_relative'] == str(l_path)].values[0]
                modif_date = date
                d = [str(l), tiles, modif_date]
                data.append(d)
            else:
                print(f'dataset {str(l)} as been modified since last time, needs to perform tile intersection...')
                gdf_parts, tiles = dataset_tiles_intersect(l, tiles_file)
                if dataset_name[0:3] == 'SHC':
                    gdf_parts['tile'] = gdf_parts['layer']
                else:
                    pass
                gdf_parts.to_file(fr'{tile_intersect_folder}\{dataset_name.replace(".shp", "_tile_intersect.shp")}')
                modif_date = date
                d = [str(l), tiles, modif_date]
                affected_tiles.append(tiles)
                data.append(d)
        else:
            print(f'dataset {str(l)} is new, needs to perform tile intersection')
            gdf_parts, tiles = dataset_tiles_intersect(l, tiles_file)
            if dataset_name[0:3] == 'SHC':
                gdf_parts['tile'] = gdf_parts['layer']
            else:
                pass
            gdf_parts.to_file(fr'{tile_intersect_folder}\{dataset_name.replace(".shp", "_tile_intersect.shp")}')
            date = find_modfified_date(l)
            modif_date = date
            d = [str(l), tiles, modif_date]
            affected_tiles.append(tiles)
            data.append(d)
    df = pd.DataFrame(data, columns=cols)
    tile_dataset = os.path.join(workdir, 'tiles_dataset.csv')
    df.to_csv(tile_dataset, sep=';', index=None)
    if warning == True:
        if len(affected_tiles) > 0:
            print(
                fr'WARNING: tiles: {affected_tiles} are affected by an updated or new dataset, consider to rerun those tiles. If you decide not to do so, you can simply ignore this message and rerun')
            quit()

    return tile_dataset, affected_tiles


