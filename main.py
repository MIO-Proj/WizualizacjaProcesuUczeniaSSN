import tensorflow.keras as k
from random import shuffle
import sys
import json
import numpy as np

class CustomCallback(k.callbacks.Callback):
    epochs = {"Learning_rate":[], "Loss":[], "Weights":[]}

    def on_train_begin(self, logs=None):
        keys = list(logs.keys())
        print(f"Starting training; got log keys: {keys}")
        print(50 * '*')

    def on_train_end(self, logs=None):
        keys = list(logs.keys())
        print(f"Stop training; got log keys: {keys}")
        print(50 * '*')

    def on_epoch_begin(self, epoch, logs=None):
        keys = list(logs.keys())
        print(f"\nStart epoch {epoch} of training; got log keys: {keys}")
        lr = float(k.backend.get_value(self.model.optimizer.learning_rate))
        print(f"Learning rate: {lr}")
        CustomCallback.epochs['Learning_rate'].append(lr)
        print(50 * '-')

    def on_epoch_end(self, epoch, logs=None):
        keys = list(logs.keys())
        print(f"\nEnd epoch {epoch} of training; got log keys: {keys}")
        layers = []
        deltas = []
        for i, l in enumerate(self.model.layers):
            print(f"warstwa {i}")
            print(l.weights)
            layers.append([l.weights[0].numpy(), l.weights[1].numpy()])
            # print("What")
            # print(layers[0][0])
            # if epoch == 0:
            #     deltas.append([l.weights[0].numpy(), l.weights[1].numpy()])
            # else:
            #     deltas.append([np.subtract(l.weights[0].numpy(),layers[epoch-1][0]),
            #     np.subtract(l.weights[1].numpy(),layers[epoch-1][1])])

        CustomCallback.epochs['Weights'].append(layers)

        
        # CustomCallback.epochs['Weights_delta'].append(deltas)
            

        print( "The average loss for epoch {} is {:7.2f}."
        .format(epoch, logs["loss"]))
        CustomCallback.epochs['Loss'].append(logs["loss"])

    print(50 * "-")



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


neurons = sys.argv[1:3] # pobranie liczby neuronów w warstwach (np. 8 8 w wywolaniu
                       # tworzy dwie ukryte warstwy, kazda po 8 neuronów)


# czytanie danych o irysach
x, y = read_data('iris.data')

x_train = x[:120]
x_test = x[120:]

y_train = y[:120]
y_test = y[120:]


# tworzenie sieci
model = k.Sequential()

# pierwsza warstwa ukryta (input_dim informuje o rozmiarze danych na wejściu)
model.add(k.layers.Dense(units=neurons[0], input_dim=4, activation='relu'))

# kolejne warstwy ukryte
for neuron in neurons[1:]:
    print(neuron)
    model.add(k.layers.Dense(units=neuron, activation='relu'))

# ostatnia warstwa, units=3 bo na wyjściu wybieramy z trzech różnych typów irysów
model.add(k.layers.Dense(units=3, activation='relu'))

# kompilacja sieci
model.compile(optimizer='sgd', loss='mse')

# tutaj ustawiamy funkcję, którajest wywoływana przy każdej epoce podczas treningu
# na razie tylko są wypisywane poszczególne wagi i biasy
callback = CustomCallback()
# epoch_end = k.callbacks.LambdaCallback(on_epoch_end=lambda e, l: print_weights(model.layers, e, l))

# trening sieci
e = int(sys.argv[3])
model.fit(x_train, y_train, epochs=e, callbacks=callback)

# sprawdzamy wyniki, w sumie nie istotne dla naszego projektu
y_pred = model.predict(x_test)

print("\n***Resutls***")
print(y_pred)
print(y_test)

print(CustomCallback.epochs)
# with open("data.json", "w") as f:
#     json.dump(CustomCallback.epochs, f, indent=4)