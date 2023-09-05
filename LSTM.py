# from PIL import Image
# import numpy
# im = Image.open('DEM.tif')
# imarray = numpy.array(im)
# print(imarray)

import xarray as xr
ds = xr.open_dataset("prueba", engine="cfgrib", backend_kwargs={
                        'filter_by_keys': {'typeOfLevel': 'surface'},
                        'errors': 'ignore'
                    })

# Cambiar escala de longitud a [-180, 180], antes 0 - 360
df = ds.to_dataframe()
map_function = lambda lon: (lon - 360) if (lon > 180) else lon
df.reset_index(inplace=True)
df["longitude"] = df['longitude'].map(map_function)

minLat, minLon, maxLat, maxLon = -10.203889, -78.041111, -8.251944, -77.010278
lat_filter = (df["latitude"] >= minLat) & (df["latitude"] <= maxLat)
lon_filter = (df["longitude"] >= minLon) & (df["longitude"] <= maxLon)
df = df.loc[lat_filter & lon_filter]

df.sort_values(by=['longitude', 'latitude'])


# print(df.iloc[[90.0,2.25]])

