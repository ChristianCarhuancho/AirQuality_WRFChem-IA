import xarray as xr
import pandas as pd
import numpy as np

def processDay(year, month, day):
    for i in range(24):
        hour = str(i).zfill(2)

        data = xr.open_dataset(f'data_chem/{year}{month}{day}/wrfout_d02_{year}-{month}-{day}_{hour}:00:00')
        pm25_data = data['PM2_5_DRY'][0][0]
        
        np.save(f'data/{year}{month}{day}/{year}{month}{day}{hour}_chem.npy', pm25_data)

    return

year = '2022'
month = '09'

for i in range (14):
    day = str(i+1).zfill(2)
    processDay(year, month, day)