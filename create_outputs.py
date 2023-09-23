import numpy as np

year = '2022'
month = '09'

for i in range(14):
    day = str(i+1).zfill(2)

    for hour in range(24):
        img = np.random.rand(30)
        np.save(f'data/{year}{month}{day}/{year}{month}{day}{str(hour).zfill(2)}_chem.npy', img)
