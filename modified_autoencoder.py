from keras.layers import Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras import Input, Model, regularizers
from keras.datasets import mnist
import numpy as np
import time
import random
import matplotlib.pyplot as plt

# based on https://analyticsindiamag.com/guide-to-autoencoders-with-python-code/
# # https://blog.keras.io/building-autoencoders-in-keras.html

start = time.time()

# encoder model

encoding_dim = 32
input_img = Input(shape=(784,))
encoded = Dense(encoding_dim, activation='relu')(input_img)
# encoded = Dense(encoding_dim, activation='relu',
#                kernel_regularizer=regularizers.l1(1e-5))(encoded)
encoded = Dense(encoding_dim, activation='relu',
                kernel_regularizer=regularizers.l2(1e-5))(encoded)
decoded = Dense(784, activation='sigmoid')(encoded)
autoencoder = Model(input_img, decoded)

# This model shows encoded images
encoder = Model(input_img, encoded)
# Creating a decoder model
encoded_input = Input(shape=(encoding_dim,))
decoder_layer = autoencoder.layers[-1]
decoder = Model(encoded_input, decoder_layer(encoded_input))

autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

autoencoder.fit(x_train, x_train,
                epochs=100,
                batch_size=256,
                validation_data=(x_test, x_test))

encoded_img = encoder.predict(x_test)
decoded_img = decoder.predict(encoded_img)

MSE = np.square(np.subtract(x_test, decoded_img)).mean()

print(MSE)

end = time.time()

print(end - start)
