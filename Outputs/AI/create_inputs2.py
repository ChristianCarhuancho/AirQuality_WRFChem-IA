import xarray as xr
import numpy as np
import pandas as pd

def process_file(year, month, day, hour):
    filename = f'/home/aireandino/Models/WRF-Chem/DATA/GFS_24h/{year}{month}{day}/gfs.t00z.pgrb2.0p25.f0{str(hour).zfill(2)}'
    ds = xr.open_dataset(filename, engine='cfgrib', filter_by_keys={'stepType': 'instant', 'typeOfLevel': 'surface'})
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

    print(sorted_df.columns)

    print(sorted_df.shape)

    data = []

    for i in range(7):
        row = []
        for j in range(4):
            index = i*4 + j
            row.append(list(sorted_df.loc[index]))
        data.append(row)

    data = np.asarray(data)

    np.save(f'data/{year}{month}{day}/{year}{month}{day}{str(hour).zfill(2)}_gfs.npy', data)

def process_day(year, month, day):
    for i in range(24):
        process_file(year, month, day, i)

year = '2024'
month = '02'
day = 18

nDays = 5

for i in range(nDays):
    process_day(year, month, str(day+i).zfill(2))
