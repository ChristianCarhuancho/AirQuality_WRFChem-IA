import pandas as pd
import numpy as np

def get_volume_from_step(step):
    df_ae = pd.read_pickle(f'data/{step[0:8]}/{step}_aerosol.pkl')
    df_at = pd.read_pickle(f'data/{step[0:8]}/{step}_atmos.pkl')
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

    year=2022
    month=9
    day = 1
    hour = 2

    while not(day == 14 and hour == 23):
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

    year=2022
    month=9
    day = 1
    hour = 2

    while not(day == 14 and hour == 23):
        sample = []

        tStr = getStepFileFormat(year, month, day, hour)
        t1Str = getStepFileFormat(year, month, day, hour+1)

        t = np.load(f'data/{tStr[0:8]}/{tStr}')
        t1 = np.load(f'data/{t1Str[0:8]}/{t1Str}')

        sample.append(t.flatten())
        sample.append(t1.flatten())

        outputs.append(sample)

        hour += 1

        if(hour > 23):
            day += 1
            hour -= 24

    return np.array(outputs)
