# -*- coding: utf-8 -*-
"""
Created on Sun Sep 24 20:44:27 2017

@author: brucelau
"""

import keras
import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop
from keras.models import load_model
from keras import initializers
from keras import regularizers
import matplotlib.pyplot as plt
#%%

y_train_noise_ = np.load('mnist_data_npy/y_train_nosie_.npy') # common labels for confusion matrix
y_train_noise = np.load('mnist_data_npy/y_train_nosie.npy') # one-hot encoding

batch_size = 128
num_classes = 10
epochs = 100

# the data, shuffled and split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()
model.add(Dense(512, activation='relu', input_shape=(784,)))
model.add(Dropout(0.2))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(10, activation='softmax'))
model.add(Dense(10,kernel_initializer=initializers.Identity(gain=1.0),
                   trainable=False,
                   bias=False))

model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(),
              metrics=['accuracy'])

history = model.fit(x_train, y_train_noise,
                    batch_size=batch_size,
                    epochs=100,
                    verbose=1,
                    validation_data=(x_test, y_test))
#%%
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


weights = []
for idx, layer in enumerate(model.layers):
    print(idx)
    weights.append(layer.get_weights())
#%% reconstruction

from keras import initializers
from keras import regularizers
from keras.constraints_m import pm_norm
model.save('saved_model\\mnist_mlp.h5')
cb_model = load_model('saved_model\\mnist_mlp.h5')
cb_model.pop()
cb_model.add(Dense(10,kernel_initializer=initializers.Identity(gain=1.0),
                   #kernel_constraint=unit_norm(axis=1),
                   #kernel_constraint=non_neg(),
                   kernel_constraint=pm_norm(axis=1),
                   trainable=True,
                   bias=False,
                   kernel_regularizer=regularizers.l2(0.1)
                   ))
#%% compile and prediction
cb_model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(),
              metrics=['accuracy'])

history = cb_model.fit(x_train, y_train_noise,
                    batch_size=batch_size,
                    epochs=100,
                    verbose=1,
                    validation_data=(x_test, y_test))
score = cb_model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
#%%
base_weights = []
for idx, layer in enumerate(cb_model.layers):
    print(idx)
    base_weights.append(layer.get_weights())
#%% obatain the output of the base intermediate layer - base model predict
from keras.models import Model
base_model = Model(inputs=cb_model.input,
                   outputs=cb_model.layers[-2].output)
#%%
base_model_predict = base_model.predict(x_test)
max_idx_base = np.argmax(base_model_predict,axis=-1)
max_idx_test = np.argmax(y_test,axis=-1)
accuracy = np.sum((max_idx_base==max_idx_test))/10000
#%%
im = base_weights[-1][0]
#%%
data = im

fig, ax = plt.subplots()
# Using matshow here just because it sets the ticks up nicely. imshow is faster.
ax.matshow(data, cmap='jet')

for (i, j), z in np.ndenumerate(data):
    ax.text(j, i, '{:0.01f}'.format(z), ha='center', va='center')

plt.show()
















