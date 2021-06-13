import tensorflow.keras as k
from random import shuffle

def iris_to_label(iris):
    if iris == 'Iris-setosa':
        return 1
    elif iris == 'Iris-versicolor':
        return 2
    elif iris == 'Iris-virginica':
        return 3
    else:
        return 4

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
            [ [iris_to_label(line[-1])] for line in data ]
        )

def print_weights(weights, epoch, logs):
    print(50 * "-")
    print(epoch)
    print(50 * "-")
    print(weights)
    print(50 * "-")
    print(logs)

# read data from file
x, y = read_data('iris.data')

x_train = x[:120]
x_test = x[120:]

y_train = y[:120]
y_test = y[120:]

print("before layers")

# tworzenie sieci
model = k.Sequential([
    k.layers.Dense(units = 16, input_shape= (4,), activation = 'relu'),
    k.layers.Dense(units = 16, activation = 'relu'),
    k.layers.Dense(units = 16, activation = 'relu'),
    k.layers.Dense(1)
])

model.compile(optimizer='sgd', loss='mse')

epoch_end = k.callbacks.LambdaCallback(on_epoch_end=lambda e, l: print_weights(model.weights, e, l))

# trening sieci
model.fit(x_train, y_train, epochs=20, callbacks=epoch_end)

y_pred = model.predict(x_test)
