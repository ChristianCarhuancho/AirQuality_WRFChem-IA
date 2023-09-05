import xarray as xr
import pandas as pd

def processDataset(fileName, year, month, day):
    ds = xr.open_dataset(fileName)
    minLat, minLon, maxLat, maxLon = -10.203889, -78.041111, -8.251944, -77.010278

    ds = ds.sel(lat=slice(minLat,maxLat), lon=slice(minLon,maxLon))

    df = ds.to_dataframe()
    df.reset_index(inplace=True)

    print(df.keys)
    print(df.columns)

    #for hour in range(24):
    #    aux_df = df[df['time'].dt.hour == hour]
    #    aux_df.to_pickle(f'data/{year}{month}{day}/{year}{month}{day}{str(hour).zfill(2)}.pkl')
    
    return

processDataset('geos_prueba.nc4', '2022', '09', '01')

#df = pd.read_pickle('data/20220901/2022090108.pkl')
#print(df.keys, df.shape)