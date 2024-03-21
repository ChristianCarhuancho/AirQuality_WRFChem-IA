from build_dataset import read_inputs, read_outputs

import numpy as np
import matplotlib.pyplot as plt

data = read_outputs()

# Supongamos que tu numpy array se llama 'data'
data_shape = data.shape

# Aplanar el array para que las variables estén en la última dimensión
flattened_data = np.reshape(data, (data_shape[0] * data_shape[1], data_shape[2], data_shape[3], data_shape[4]))

# Transponer el array para que las variables estén en la primera dimensión
transposed_data = np.transpose(flattened_data, (3, 0, 1, 2))

# Ahora 'transposed_data' tiene la forma (3, 30*3, 70, 35)

# Graficar un boxplot por cada variable y guardarlos como archivos PNG
for i in range(transposed_data.shape[0]):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.boxplot(transposed_data[i].flatten())
    ax.set_title(f'Variable {i+1}')
    
    # Guardar la imagen como PNG
    filename = f'boxplot_variable_{i+1}.png'
    plt.savefig(filename)
    plt.close()  # Cerrar la figura para liberar memoria

    print(f'Boxplot de la variable {i+1} guardado como {filename}')
