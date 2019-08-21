import tensorflow as tf
from tensorflow import keras
import csv
import numpy as np


def load_data(addr):
    data = np.loadtxt(fname=addr)
    return data


def basic_nn():
    model = keras.Sequential([
        keras.layers.Dense(104, input_shape=(26, ), activation=tf.nn.sigmoid),
        keras.layers.Dense(208, activation=tf.nn.relu),
        keras.layers.Dense(104, activation=tf.nn.relu),
        keras.layers.Dense(1, activation=tf.nn.softmax)
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


# 26 dimensions, 8 proportional dimensions, 18 absolute value
def data_generate(num_samples):
    training_samples = []
    for i in range(num_samples):
        B = np.random.rand(18)*20000
        A = np.random.randn(8)
        fake_data = np.concatenate([A, B])
        training_samples.append(fake_data)

    return training_samples


estimator = keras.wrappers.scikit_learn.KerasClassifier(build_fn=basic_nn, nb_epoch=50, batch_size=20)
data_init = load_data('../data/v3.txt')
a = data_init[:, 0].reshape([1, 26])
b = data_init[:, 1].reshape([1, 26])
data_init = np.concatenate([a, b], axis=0)
label_init = np.array([0, 1])
x_train = np.array(data_generate(50000))
y_train = np.random.random_integers(low=0, high=1, size=50000)
estimator.fit(x=data_init, y=label_init)
estimator.fit(x=x_train, y=y_train)
pred = estimator.predict(data_init)


# estimator.fit(x=x_train, y=y_train)
# prediction = estimator.predict()