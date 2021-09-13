import main
import matplotlib.pyplot as plt
# from .main import train_SSN
from math import sqrt
import numpy as np
import imageio
import os 

nr_of_layers = 4
nr_of_neurons = 12
nr_of_epochs = 100
main.train_SSN(nr_of_layers, nr_of_neurons ,nr_of_epochs)
epochs = main.CustomCallback.epochs
print(epochs)
images = []
filenames =[]
for i in range(len(epochs['Weights'])): 
    bias_in_layer = []
    for layers in epochs['Weights'][i][1]:
        j =0
        for bias_in_layer in layers:
            print("Bias", bias_in_layer)
            if len(bias_in_layer.shape) == 0: 
                continue 
            plt.plot(np.arange(1, nr_of_neurons+1, 1), bias_in_layer, label=f'layer_{j}')
            plt.title(f'epoch_{i}')
            plt.xticks(np.arange(1, nr_of_neurons+1, 1))
            plt.legend(loc='best')
            plt.xlabel('neuron')
            j += 1
    # plt.show()
    filename = f'images/epoch_{i}'
    filenames.append(filename+'.png')
    plt.savefig(f'images/epoch_{i}')
    plt.clf()


with imageio.get_writer('bias.gif', mode='I') as writer:
    for filename in filenames:
        image = imageio.imread(filename)
        writer.append_data(image)

        
# Remove files
for filename in set(filenames):
    os.remove(filename)
