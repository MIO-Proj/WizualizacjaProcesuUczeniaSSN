import wandb
from wandb.keras import WandbCallback
import tensorflow.keras as k
from random import shuffle
from main import iris_to_label, label_to_iris, get_weights, read_data
import sys

# 1. Start a new run
wandb.login(key="a1c4163f233a795495d3f3955b8baf96668e47a8")
wandb.init(project='MIO-SSNvisualization', entity='laseka')


# 2. Save model inputs and hyperparameters
# config = wandb.config
# config.learning_rate = 0.01

# Defining a model

class CustomCallback(k.callbacks.Callback):
    epochs = {"Learning_rate": [], "Loss": [], "Weights": []}

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

        CustomCallback.epochs['Weights'].append(layers)

        print("The average loss for epoch {} is {:7.2f}."
              .format(epoch, logs["loss"]))
        CustomCallback.epochs['Loss'].append(logs["loss"])

    print(50 * "-")


if __name__ == "__main__":
    neurons = sys.argv[1:3]
    x, y = read_data('iris.data')

    x_train = x[:120]
    x_test = x[120:]

    y_train = y[:120]
    y_test = y[120:]

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
    callback = CustomCallback()

    # trening sieci
    e = int(sys.argv[3])
    # model.fit(x_train, y_train, epochs=e, callbacks=callback)

    # 3. Log layer dimensions and metrics over time
    model.fit(x_train, y_train, epochs=e, validation_data=(x_test, y_test),
              callbacks=[WandbCallback(log_weights=True)])
