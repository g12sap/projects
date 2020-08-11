import pandas as pd 
import numpy as np 
import tensorflow as tf 
from tensorflow import keras 
print(tf.__version__)

#Callback
class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs = {}):
        if(logs.get('loss')< 0.4):
            print("\n Loss is low, so cancelling training!")
            self.model.stop_training = True

callbacks = myCallback()
mnist = tf.keras.datasets.fashion_mnist
(training_images, training_labels), (test_images, test_labels) = mnist.load_data()

model = keras.Sequential([
    keras.layers.Flatten(input_shape = (28,28)),
    keras.layers.Dense(128, activation = tf.nn.relu),
    keras.layers.Dense(10, activation = tf.nn.softmax)
])

#Viewing the pixels numerically
np.set_printoptions(linewidth=200)
import matplotlib.pyplot as plt
plt.imshow(training_images[42])
print(training_labels[42])
print(training_images[42])

#normalizing the data
training_images = training_images / 255.0
test_images = test_images / 255.0

#Compiling the model to calculate the loss function and the optimizer
model.compile(optimizer = tf.optimizers.Adam(),
            loss = 'sparse_categorical_crossentropy',
            metrics = ['accuracy'])
model.fit(training_images, training_labels, epochs = 10, callbacks = [callbacks])

#runnning on test data
model.evaluate(test_images, test_labels)

