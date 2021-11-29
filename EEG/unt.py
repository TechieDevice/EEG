from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
from tensorflow.keras.utils import plot_model
from tensorflow.keras import regularizers
import matplotlib.pyplot as plt
import os
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras.initializers import *
from tensorflow.keras import backend as K
from tensorflow.keras.utils import get_custom_objects
from sklearn.model_selection import train_test_split, StratifiedKFold, LeaveOneOut
import scipy.io
import math
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, BatchNormalization, MaxPooling2D, Dropout, Flatten, Dense, LSTM, Concatenate
from tensorflow.keras.layers import Input, TimeDistributed, Reshape, MaxPooling1D, Permute, Conv1D, BatchNormalization
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
import random



data_root = '/content/drive/My Drive/Share/'

d = np.load(data_root + 'morlet_avg.npz')
X = d['a']
Y = d['b']
n_classes = len(np.unique(Y))
h = 17
w = 1250
c= 1
le = LabelEncoder()
le.fit(Y)
Y = le.transform(Y)
X = np.expand_dims(X, axis=3)
print(X.shape)
print(Y.shape)
print(len(np.unique(Y)))

# First let's separate the dataset from 1 matrix to a list of matricies
image_list = np.split(X, 402)
label_list = np.split(Y, 402)

left_input = []
right_input = []
targets = []

for i in range(402):
  left_input.append(image_list[i])
  right_input.append(image_list[i])
  targets.append(1)

for i in range(0, 402, 2):
  if (label_list[i] == label_list[i+1]):
    left_input.append(image_list[i])
    right_input.append(image_list[i+1])
    targets.append(1)

if (label_list[-1] != label_list[0]):
  left_input.append(image_list[-1])
  right_input.append(image_list[0])
  targets.append(0)

for i in range(1, 401):
  if (label_list[i] != label_list[i-1]):
    left_input.append(image_list[i-1])
    right_input.append(image_list[i])
    targets.append(0)
  if (label_list[i] == label_list[i+1]):
    left_input.append(image_list[i+1])
    right_input.append(image_list[i])
    targets.append(0)
            
left_input = np.squeeze(np.array(left_input))
right_input = np.squeeze(np.array(right_input))
targets = np.squeeze(np.array(targets))

print(left_input.shape)
print(right_input.shape)
print(np.unique(targets, return_counts=True))
plot_model(encoder, show_dtype=True, show_shapes=True)

encoder = tf.keras.Sequential([
    InputLayer(input_shape=(w, h, c)),
    Conv2D(filters=64, kernel_size=(12, 4),
           padding='valid'),
    BatchNormalization(),
    ReLU(),
    MaxPool2D((2, 1)),

    Conv2D(filters=128, kernel_size=(8, 2), padding='valid'),
    BatchNormalization(),
    ReLU(),
    MaxPool2D((4, 2)),

    Conv2D(filters=128, kernel_size=(8, 2), padding='valid'),
    BatchNormalization(),
    ReLU(),
    MaxPool2D((4, 2)),

    Conv2D(filters=128, kernel_size=(8, 2), padding='valid'),
    BatchNormalization(),
    ReLU(),

    Flatten(),
            
    Dense(18),
    Activation('sigmoid')]
    )
left_input  = Input((w,h,c))
right_input = Input((w,h,c))
processed_a = encoder(left_input)
processed_b = encoder(right_input)
L1_layer = Lambda(lambda tensor:K.abs(tensor[0] - tensor[1]))
L1_distance = L1_layer([processed_a , processed_b])
prediction = Dense(1,activation='sigmoid')(L1_distance)
siamese_net = Model(inputs=[left_input,right_input],outputs=prediction)
optimizer = Adam(0.001,decay = 1e-4)
siamese_net.compile(loss='binary_crossentropy',optimizer=optimizer,metrics=['accuracy'])
siamese_net.summary()

siamese_net.fit([left_input,right_input], targets,
          batch_size=16,
          epochs=30,
          verbose=1, validation_split=0.2)
