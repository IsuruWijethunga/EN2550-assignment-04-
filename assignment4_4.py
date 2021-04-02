import tensorflow as tf
import datetime
from tensorflow import keras
from tensorflow.keras import datasets, layers, models
from keras.optimizers import SGD
import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__)
mnist = keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
paddings = tf.constant([[0,0], [2,2], [2,2]])
x_train = tf.pad(x_train, paddings, constant_values = 0)
x_test = tf.pad(x_test, paddings, constant_values = 0)
x_train = tf.dtypes.cast(x_train,tf.float32)
x_test = tf.dtypes.cast(x_test, tf.float32)
x_train, x_test, = x_train[...,np.newaxis]/255, x_test[..., np.newaxis]/255

classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

plt.figure(figsize = (10,10))
for i in range(25):
  plt.subplot(5,5,i+1)
  plt.xticks([])
  plt.yticks([])
  plt.grid(False)
  plt.imshow(tf.reshape(x_train[i], [32,32]), cmap = plt.cm.gray)
  plt.xlabel(classes[y_train[i]])
plt.show()

model = keras.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape = (32,32,1)))
model.add(layers.MaxPool2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPool2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPool2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(64,activation = 'relu'))
model.add(layers.Dense(10))

sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9)
model.compile(optimizer='sgd',loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),metrics = ['accuracy'])
print(model.summary()) 

log_dir = ".\\logs\\fit\\" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir = log_dir, histogram_freq = 1)
history = model.fit(x_train, y_train, epochs = 10, batch_size = 50, validation_data=(x_test, y_test), callbacks = [tensorboard_callback])
test_loss, test_acc = model.evaluate(x_test, y_test, verbose = 2)
print("\nTest accuracy: ", test_acc)
