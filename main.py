import tensorflow.keras as k
from random import shuffle
import sys

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

def print_weights(layers, epoch, logs):
    print(50 * "-")
    print(epoch)
    print(50 * "-")
    for i, l in enumerate(layers):
        print(f"warstwa {i}")
        # w zmiennej weights przechowywane są dwie tablice:
        # pierwsza to wagi pomiędzy połączeniami pomiędzy wartswami
        # druga to biasy neuronów
        print(l.weights)
    print(50 * "-")
    print(logs)
    print(50 * "-")

##########################################################################################

neurons = sys.argv[1:] # pobranie liczby neuronów w warstwach (np. 8 8 w wywolaniu
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
epoch_end = k.callbacks.LambdaCallback(on_epoch_end=lambda e, l: print_weights(model.layers, e, l))

# trening sieci
model.fit(x_train, y_train, epochs=5, callbacks=epoch_end)

# sprawdzamy wyniki, w sumie nie istotne dla naszego projektu
y_pred = model.predict(x_test)
print(y_pred)
print(y_test)
