import pandas as pd

df = None

flag = 0

pd.set_option('display.max_columns', None)

# Atmos
for i in range(14):
    day = str(i+1).zfill(2)

    for j in range(24):
        hour = str(j).zfill(2)
        
        dfAux = pd.read_pickle(f'/media/datos/ccarhuancho/data/data/202209{day}/202209{day}{hour}_atmos.pkl')

        if(flag == 0):
            df = dfAux
            flag = 1
        else:
            df = pd.concat([df, dfAux])

print(df.columns)
print(df.dtypes)
print(df.isna().sum())
print(df.describe())


# Aerosol

df = None

flag = 0

# Atmos
for i in range(14):
    day = str(i+1).zfill(2)

    for j in range(24):
        hour = str(j).zfill(2)
        
        dfAux = pd.read_pickle(f'/media/datos/ccarhuancho/data/data/202209{day}/202209{day}{hour}_aerosol.pkl')

        if(flag == 0):
            df = dfAux
            flag = 1
        else:
            df = pd.concat([df, dfAux])

print(df.columns)
print(df.dtypes)
print(df.isna().sum())
print(df.describe())