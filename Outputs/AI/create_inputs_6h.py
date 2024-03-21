import xarray as xr
import numpy as np
import pandas as pd

def interpolate_dataframe(source_df, n):
    df_zero = pd.DataFrame(0, index=source_df.index, columns=source_df.columns)
    
    dfs_interp = []

    for i in range(1, n + 1):
        original_weight = i / n
        zero_weight = 1 - original_weight
        df_interp = source_df * original_weight + df_zero * zero_weight
        dfs_interp.append(df_interp)

    return dfs_interp


def dataframe_to_numpy(source_df):
    data = []

    for i in range(7):
        row = []
        for j in range(4):
            index = i*4 + j
            row.append(list(source_df.loc[index]))
        data.append(row)

    return np.asarray(data)


def process_file(year, month, day, hour):
    filename = f'/home/aireandino/Models/WRF-Chem/DATA/GFS/{year}{month}{day}R/gfs.t{str(hour).zfill(2)}z.pgrb2.0p25.f000'
    ds = xr.open_dataset(filename, engine='cfgrib', filter_by_keys={'typeOfLevel': 'surface'})
    df = ds.to_dataframe()

    min_lat, max_lat = -10.203889, -8.251944
    min_lon, max_lon = -78.041111, -77.010278

    min_lon = (min_lon + 180) % 360
    max_lon = (max_lon + 180) % 360

    filtered_df = df.loc[(min_lat <= df.index.get_level_values('latitude')) & 
                     (df.index.get_level_values('latitude') <= max_lat) &
                     (min_lon <= df.index.get_level_values('longitude')) & 
                     (df.index.get_level_values('longitude') <= max_lon)]

    sorted_df = filtered_df.sort_index(level=['latitude', 'longitude'], ascending=[True, True])
    sorted_df.reset_index(inplace=True)


    sorted_df.drop(['time', 'step', 'surface', 'hindex', 'valid_time', 'fco2rec', 'sdwe', 'sde', 'veg', 'wilt', 'fldcp', 'sit', 'lsm', 'siconc', 'unknown', 'csnow', 'cicep', 'cfrzr', 'crain', 'slt', 'latitude', 'longitude'], axis=1, inplace=True)

    n = 6

    dfs = interpolate_dataframe(sorted_df, n)

    for i in range(n):
        np.save(f'data/{year}{month}{day}/{year}{month}{day}{str(hour+i).zfill(2)}_gfs.npy', dataframe_to_numpy(dfs[i]))
    

def process_day(year, month, day):
    for i in range(4):
        process_file(year, month, day, i*6)

year = '2023'
month = '11'
day = 8

nDays = 5

for i in range(nDays):
    process_day(year, month, str(day+i).zfill(2))
