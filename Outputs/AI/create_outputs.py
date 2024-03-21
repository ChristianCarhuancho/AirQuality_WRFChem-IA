import subprocess
import xarray as xr
import pandas as pd
import numpy as np

def process_file(year, month, day, hour):
    try:
        data = xr.open_dataset(f'../Models/WRF-Chem/OUTPUT/atmos/{year}{month}{day}R/wrfout_d02_{year}-{month}-{day}_{hour}:00:00')

        dataT = data['T2'][0].data
        dataU = data['U10'][0].data
        dataV = data['V10'][0].data

        return np.dstack((dataT, dataU, dataV))
    except OSError as e:
        print(f"Error opening file: {e}")
        return None

def process_day(year, month, day):
    subprocess.run(f'mkdir -p data/{year}{month}{day}', shell=True)

    for i in range(24):
        matrix = process_file(year, month, day, str(i).zfill(2))
        if matrix is not None:
            np.save(f'data/{year}{month}{day}/{year}{month}{day}{str(i).zfill(2)}_chem.npy', matrix)

year = '2023'
month = '11'
day = 8

nDays = 5

for i in range(nDays):
    process_day(year, month, str(day+i).zfill(2))
