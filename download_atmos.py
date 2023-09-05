import subprocess
import xarray as xr
import pandas as pd

def processDataset(fileName, year, month, day):
    ds = xr.open_dataset(fileName)
    minLat, minLon, maxLat, maxLon = -10.203889, -78.041111, -8.251944, -77.010278

    ds = ds.sel(lat=slice(minLat,maxLat), lon=slice(minLon,maxLon))

    df = ds.to_dataframe()
    df.reset_index(inplace=True)

    for hour in range(24):
        aux_df = df[df['time'].dt.hour == hour]
        aux_df.to_pickle(f'data/{year}{month}{day}/{year}{month}{day}{str(hour).zfill(2)}_atmos.pkl')
    
    return



year = '2022'
month = '09'

# Days
subprocess.run('mkdir -p raw_data/atmos', shell=True)
subprocess.run('mkdir -p data', shell=True)

for i in range (14):
    day = str(i+1).zfill(2)
    subprocess.run('mkdir -p data/202209' + day, shell=True)
    fileURL = f'https://goldsmr4.gesdisc.eosdis.nasa.gov/data/MERRA2/M2T1NXLND.5.12.4/{year}/{month}/MERRA2_400.tavg1_2d_lnd_Nx.{year}{month}{day}.nc4'
    
    subprocess.run(f'wget --load-cookies ~/.urs_cookies --save-cookies ~/.urs_cookies --keep-session-cookies -O raw_data/atmos/{year}{month}{day}.nc4 {fileURL}', shell=True)
    processDataset(f'raw_data/atmos/{year}{month}{day}.nc4', year, month, day)
    subprocess.run(f'rm raw_data/atmos/{year}{month}{day}.nc4', shell=True)
    