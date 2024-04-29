import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image

def get_volume_from_step(step):
    data = np.load(f'data/{step[0:8]}/{step}_gfs.npy')
    return data

def get_test_volume_from_step(step):
    df_ae = pd.read_pickle(f'val_data/{step[0:8]}/{step}_aerosol.pkl')
    df_at = pd.read_pickle(f'val_data/{step[0:8]}/{step}_atmos.pkl')
    df = pd.concat([df_ae, df_at], axis=1, join='outer')
    df.drop(columns=['time', 'lat', 'lon', 'TPSNOW'], inplace=True)
    df = df.to_numpy()
    return df


def getStepFileFormat(year, month, day, hour):
    dayAux = day
    hourAux = hour

    if hour > 23 :
        day += 1
        hour -= 24
    elif hour < 0 :
        day -= 1
        hour += 24

    return f'{year}{str(month).zfill(2)}{str(day).zfill(2)}{str(hour).zfill(2)}'


def read_inputs():
    inputs = []

    year=2023
    month=11
    day = 8
    hour = 2

    while not(day == 12 and hour == 23):
        sample = []
        #print(getStepFileFormat(year, month, day, hour-2))
        #print(getStepFileFormat(year, month, day, hour-1))
        #print(getStepFileFormat(year, month, day, hour))
        #print()

        sample.append(get_volume_from_step(getStepFileFormat(year, month, day, hour-2)))
        sample.append(get_volume_from_step(getStepFileFormat(year, month, day, hour-1)))
        sample.append(get_volume_from_step(getStepFileFormat(year, month, day, hour)))

        inputs.append(sample)

        hour += 1

        if(hour > 23):
            day += 1
            hour -= 24

    return np.array(inputs)

def read_outputs():
    outputs = []

    year = 2023
    month = 11
    day = 8
    hour = 2

    while not(day == 12 and hour == 23):
        sample = []

        tStr = getStepFileFormat(year, month, day, hour)
        t1Str = getStepFileFormat(year, month, day, hour + 1)

        t = np.load(f'data/{tStr[0:8]}/{tStr}_chem.npy')
        t1 = np.load(f'data/{t1Str[0:8]}/{t1Str}_chem.npy')

        sample.append(t)
        sample.append(t1)

        outputs.append(sample)

        hour += 1

        if(hour > 23):
            day += 1
            hour -= 24

    return np.array(outputs)

def getIDWvalue(hour, row, column, stations, values):
    for i in range(len(stations)):
        if stations[i]['position'][0] == row and stations[i]['position'][1] == column:
            return values[i][hour]
    
    value = 0
    for i in range(len(stations)):
        value += values[i][hour]/(6*((row-stations[i]['position'][0])**2 + (column-stations[i]['position'][1])**2))

    return value

def get_date_matrix_outputs(year, month, day, stations, values):
    matrix_size = (36, 18)

    for hour in range(24):
        matrix = [[0]*matrix_size[1]]*matrix_size[0]

        # Per element
        for i in range(matrix_size[0]):
            for j in range(matrix_size[1]):
                matrix[i][j] = getIDWvalue(hour, i,j, stations, values)

        np.save(f'val_data/{year}{month}{day}/{year}{month}{day}{str(hour).zfill(2)}_real.npy', matrix)

def process_val_date(year, month, day):
    stations = [{'name': 'tumpa', 'position': (20,7)}, {'name': 'huaraz', 'position': (31,9)}, {'name': 'chacas', 'position': (18,14)}]
    values = []

    # 24 element array per station
    for station in stations:
        filename = f"val_data/{year}{month}{day}/{year}{month}{day}_{station['name']}.csv"
        data = pd.read_csv(filename)
        data['UTCDateTime'] = pd.to_datetime(data['UTCDateTime'])
        times = data['UTCDateTime']
        data = data.groupby([times.dt.hour]).pm2_5_atm.mean()
        values.append(data.values)

    get_date_matrix_outputs(year, month, day, stations, values)

def process_val_outputs_csv():
    year = '2023'
    month = '09'

    for i in range(2):
        day = str(i+1).zfill(2)
        process_val_date(year, month, day)

    return

def read_test_inputs():
    inputs = []

    year=2023
    month=9
    day = 1
    hour = 2

    while not(day == 2 and hour == 23):
        sample = []
        #print(getStepFileFormat(year, month, day, hour-2))
        #print(getStepFileFormat(year, month, day, hour-1))
        #print(getStepFileFormat(year, month, day, hour))
        #print()

        sample.append(get_test_volume_from_step(getStepFileFormat(year, month, day, hour-2)))
        sample.append(get_test_volume_from_step(getStepFileFormat(year, month, day, hour-1)))
        sample.append(get_test_volume_from_step(getStepFileFormat(year, month, day, hour)))

        inputs.append(sample)

        hour += 1

        if(hour > 23):
            day += 1
            hour -= 24

    return np.array(inputs)

def read_test_real_outputs():
    outputs = []

    year=2023
    month=9
    day = 1
    hour = 2

    while not(day == 2 and hour == 23):
        sample = []

        tStr = getStepFileFormat(year, month, day, hour)
        t1Str = getStepFileFormat(year, month, day, hour+1)

        t = np.load(f'val_data/{tStr[0:8]}/{tStr}_real.npy')
        t1 = np.load(f'val_data/{t1Str[0:8]}/{t1Str}_real.npy')

        sample.append(t.reshape(36,18,1))
        sample.append(t1.reshape(36,18,1))

        outputs.append(sample)

        hour += 1

        if(hour > 23):
            day += 1
            hour -= 24

    return np.array(outputs)

def read_test_chem_outputs():
    outputs = []

    year=2023
    month=9
    day = 1
    hour = 2

    while not(day == 2 and hour == 23):
        sample = []

        tStr = getStepFileFormat(year, month, day, hour)
        t1Str = getStepFileFormat(year, month, day, hour+1)

        t = np.load(f'val_data/{tStr[0:8]}/{tStr}_chem.npy')
        t1 = np.load(f'val_data/{t1Str[0:8]}/{t1Str}_chem.npy')

        sample.append(t.reshape(36,18,1))
        sample.append(t1.reshape(36,18,1))

        outputs.append(sample)

        hour += 1

        if(hour > 23):
            day += 1
            hour -= 24

    return np.array(outputs)

def show_input_images(data):
    imagen = Image.open('./images/map.png')
    imagen = imagen.resize((480,840))
    imagen_array = np.array(imagen)    

    samples = []
    samples.append(data[0][0])
    samples.append(data[0][1])
    samples.append(data[1][1])
    samples.append(data[2][1])
    samples.append(data[3][1])
    samples.append(data[4][1])

    for i, sample in enumerate(samples):
        matrix = sample[:, :, 1]
        plt.clf()
        heatmap_data = matrix.repeat(120, axis=0).repeat(120, axis=1)
        fig, ax = plt.subplots()
        ax.set_xticks([])
        ax.set_yticks([])
        ax.imshow(imagen_array)
        heatmap = ax.imshow(heatmap_data, cmap='Reds', alpha=0.5, vmin=0, vmax=20)
        cbar = fig.colorbar(heatmap)
        plt.savefig(f'./images/heatmap_{i}.png')
