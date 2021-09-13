from imageio.core.util import Image
import tensorflow.keras as k
import matplotlib.pyplot as plt
import numpy as np
from random import shuffle
import sys
import os
import imageio

def iris_to_label(iris):
    if iris == 'Iris-setosa':
        return [1, 0, 0]
    elif iris == 'Iris-versicolor':
        return [0, 1, 0]
    elif iris == 'Iris-virginica':
        return [0, 0, 1]
    else:
        return []

def label_to_iris(label):
    if label == 1:
        return 'Iris-setosa'
    elif label == 2:
        return 'Iris-versicolor'
    elif label == 3:
        return 'Iris-virginica'
    else:
        return '...'

def read_data(filename):

    with open(filename, 'r') as f:
        data = f.readlines()
        data = list(map(lambda l: l.strip().split(','), data))
        data.pop()
        shuffle(data)
        return ( [[float(string) for string in inner[:-1] ] for inner in data],
            [ iris_to_label(line[-1]) for line in data ]
        )

def get_weights(layers, epoch, logs):

    data_from_current_epoch = []

    for i, layer in enumerate(layers):

        neurons = (layer.weights)[0].numpy()
        bias = (layer.weights)[1].numpy()

        layer_data = [neurons, bias]

        data_from_current_epoch.append(layer_data)

    return data_from_current_epoch

##########################################################################################

neurons = sys.argv[1:] # pobranie liczby neuronów w warstwach (np. 8 8 w wywolaniu
                       # tworzy dwie ukryte warstwy, kazda po 8 neuronów)


# czytanie danych o irysach
x, y = read_data('iris.data')

x_train = x[:140]
x_test = x[140:]

y_train = y[:140]
y_test = y[140:]

input_size = len(x[0])
output_size = len(y[0])

# tworzenie sieci
model = k.Sequential()

# pierwsza warstwa ukryta (input_dim informuje o rozmiarze danych na wejściu)
model.add(k.layers.Dense(units=neurons[0], input_dim=input_size, activation='relu'))

# kolejne warstwy ukryte
for neuron in neurons[1:]:
    model.add(k.layers.Dense(units=neuron, activation='relu'))

# ostatnia warstwa, units=3 bo na wyjściu wybieramy z trzech różnych typów irysów
model.add(k.layers.Dense(units=output_size, activation='sigmoid'))

# kompilacja sieci
model.compile(optimizer='sgd', loss='mse')

# tutaj ustawiamy funkcję, którajest wywoływana przy każdej epoce podczas treningu
# na razie tylko są wypisywane poszczególne wagi i biasy
weights = []
epoch_end = k.callbacks.LambdaCallback(on_epoch_end=lambda e, l: weights.append(get_weights(model.layers, e, l)))

# trening sieci
model.fit(x_train, y_train, epochs=100, callbacks=epoch_end)

#########################
filenames = []
for i in range(len(weights)):

    filename = f'{i}.png'
    filenames.append(filename)

    fig = plt.figure()
    plt.suptitle(f"Epoka {i+1}")
    plt.axis('off')

    for j, layer in enumerate(weights[i]):

        ax = fig.add_subplot(2, 2, j+1)
        im = ax.imshow(np.transpose(layer[0]), cmap='seismic', vmin=-0.5, vmax=0.5)
        ax.set_title(f"Warstwa {j+1}")
        ax.axis('off')

        plt.colorbar(im)        
    
    plt.savefig(filename)
    plt.close()

# build gif
with imageio.get_writer('mygif.gif', mode='I') as writer:
    for filename in filenames:
        image = imageio.imread(filename)
        writer.append_data(image)
        # for _ in range(20):
            # writer.append_data(image)
        
# Remove files
for filename in set(filenames):
    os.remove(filename)