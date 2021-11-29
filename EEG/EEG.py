import pywt
import sys
import gc
import pandas as pd
from sklearn.feature_selection import *
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.utils import plot_model
from tensorflow.keras import regularizers
import matplotlib.pyplot as plt
import os
from pathlib import Path
from tensorflow.keras.layers import *
from tensorflow.keras.initializers import *
from tensorflow.keras import backend as K
from tensorflow.keras.utils import get_custom_objects
from sklearn.model_selection import train_test_split, StratifiedKFold, LeaveOneGroupOut
import scipy.io
import math
import mne
import gc
from scipy.spatial import distance_matrix
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import *
from tensorflow.keras.metrics import *
import random
from sklearn.model_selection import train_test_split

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
mne.set_log_level('CRITICAL')
i = 0.0
X = []
Y = []

data_marker = 0
if data_marker == 1:
    data_dir = Path(r'D:\DATASET\files')
    for inner_dir in data_dir.iterdir():
        for item in inner_dir.iterdir():
            if (item.suffix == '.edf'):
                x = mne.io.read_raw_edf(item.__str__())
                tmax = 1.5
                events = mne.make_fixed_length_events(x, id=1, duration=tmax, overlap=0.)
                epochs = mne.Epochs(x, events, tmin=0, tmax=tmax, baseline=None)
                t = epochs.get_data()
                X.append(t)
                Y.append(np.repeat(int(np.trunc(i)), t.shape[0]))
                i = i + 0.5
                print(Y[-1][0] + 1)
    X = np.vstack(X)
    Y = np.concatenate(Y)
    print(X.shape)
    print(Y.shape)

    dataX = np.array(X)
    dataY = np.array(Y)
    np.save(r'D:\DATASET\dataX.npy', dataX)
    np.save(r'D:\DATASET\dataY.npy', dataY)

X = np.load(r'D:\DATASET\dataX.npy')
Y = np.load(r'D:\DATASET\dataY.npy')
print(X.shape, Y.shape)


def eeg_feature_extract_wavelet2(x):
    c = pywt.wavedec(x, 'db4', level=5)
    mean = np.array([np.mean(v, axis=1) for v in c])
    std = np.array([np.std(v, axis=1) for v in c])
    rms = np.sqrt(np.array([np.mean(v**2, axis=1) for v in c]))
    print(np.concatenate((mean, std, rms), axis=0).shape)
    return np.concatenate((mean, std, rms), axis=0).ravel()

def Euclidean(A,B):
    v = tf.expand_dims(tf.reduce_sum(tf.square(A), 1), 1)
    p1 = tf.reshape(tf.reduce_sum(v,axis=1),(-1,1))
    v = tf.reshape(tf.reduce_sum(tf.square(B), 1), shape=[-1, 1])
    p2 = tf.transpose(tf.reshape(tf.reduce_sum(v,axis=1),(-1,1)))
    res = tf.sqrt(tf.add(p1, p2) - 2 * tf.matmul(A, B, transpose_b=True))
    res = tf.linalg.set_diag(res, tf.repeat(1000.0, repeats=res.shape[0]))
    res = tf.cast(res >= 0, res.dtype) * res
    return(res)

def predicts_top(A, B, Y):
    res = Euclidean(A, B)
    res = tf.argsort(dist, axis=1,direction='ASCENDING')
    res = tf.gather(Yn, res)
    return res

def predicts(A, B, Y):
    res = Euclidean(A, B)
    res = tf.math.argmin(res, axis=0, output_type=tf.dtypes.int64)
    res = Y[res]
    return res

Xn = np.expand_dims(X, axis=3)
Yn = np.copy(Y)
print(Xn.shape)

loo = LeaveOneGroupOut()
for train_index, test_index in loo.split(Xn, Yn, Yn):
  X_train, X_test = Xn[train_index], Xn[test_index]
  y_train, y_test = Yn[train_index], Yn[test_index]
  break

regularizer=tf.keras.regularizers.L2(l2=0.0001)

embedding_model = tf.keras.models.Sequential([
      tf.keras.layers.BatchNormalization(input_shape=(64,241,1)),                                          
      tf.keras.layers.Convolution2D(16, 2, input_shape=(64,241,1), activation='relu', kernel_regularizer=regularizer),
      tf.keras.layers.MaxPooling2D(2, 2),
      tf.keras.layers.BatchNormalization(),
      tf.keras.layers.Convolution2D(32, 2, activation='relu', kernel_regularizer=regularizer),
      tf.keras.layers.MaxPooling2D(3, 3),
      tf.keras.layers.BatchNormalization(),
      tf.keras.layers.Convolution2D(64, 2, activation='relu', kernel_regularizer=regularizer),
      tf.keras.layers.MaxPooling2D(6, 4),
      tf.keras.layers.BatchNormalization(),
      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(256, activation='relu', kernel_regularizer=regularizer),
      tf.keras.layers.Dense(128, activation=None, kernel_regularizer=regularizer),
      tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1))
  ])
embedding_model.summary()

batch_size = 128
epochs = 100
steps_per_epoch = int(X_train.shape[0]/batch_size)

embedding_model.compile(loss=tfa.losses.TripletSemiHardLoss(), optimizer='nadam')

_ = embedding_model.fit(
    X_train, y_train,
    steps_per_epoch=steps_per_epoch,
    epochs=epochs, verbose=True 
)

def get_batch(batch_size,s="train"):
    """
    Create batch of n pairs, half same class, half different class
    """
    if s == 'train':
        X = X_train
        categories = y_train
    else:
        X = X_test
        categories = y_test
    n_classes, n_examples, w, h = X.shape
    
    # randomly sample several classes to use in the batch
    categories = rng.choice(n_classes,size=(batch_size,),replace=False)
    
    # initialize 2 empty arrays for the input image batch
    pairs=[np.zeros((batch_size, h, w,1)) for i in range(2)]
    
    # initialize vector for the targets
    targets=np.zeros((batch_size,))
    
    # make one half of it '1's, so 2nd half of batch has same class
    targets[batch_size//2:] = 1
    for i in range(batch_size):
        category = categories[i]
        idx_1 = rng.randint(0, n_examples)
        pairs[0][i,:,:,:] = X[category, idx_1].reshape(w, h, 1)
        idx_2 = rng.randint(0, n_examples)
        
        # pick images of same class for 1st half, different for 2nd
        if i >= batch_size // 2:
            category_2 = category  
        else: 
            # add a random number to the category modulo n classes to ensure 2nd image has a different category
            category_2 = (category + rng.randint(1,n_classes)) % n_classes
        
        pairs[1][i,:,:,:] = X[category_2,idx_2].reshape(w, h,1)
    
    return pairs, targets

loo = LeaveOneGroupOut()
for train_index, test_index in loo.split(Xn, Yn, Yn):
  X_train, X_test = Xn[train_index], Xn[test_index]
  y_train, y_test = Yn[train_index], Yn[test_index]
  regularizer = tf.keras.regularizers.l2(0.0001)

  embedding_model = tf.keras.models.Sequential([
      tf.keras.layers.BatchNormalization(input_shape=(64,241,1)),                                          
      tf.keras.layers.Convolution2D(16, 2, input_shape=(64,241,1), activation='relu', kernel_regularizer=regularizer),
      tf.keras.layers.MaxPooling2D(2, 2),
      tf.keras.layers.BatchNormalization(),
      tf.keras.layers.Convolution2D(32, 2, activation='relu', kernel_regularizer=regularizer),
      tf.keras.layers.MaxPooling2D(3, 3),
      tf.keras.layers.BatchNormalization(),
      tf.keras.layers.Convolution2D(64, 2, activation='relu', kernel_regularizer=regularizer),
      tf.keras.layers.MaxPooling2D(6, 4),
      tf.keras.layers.BatchNormalization(),
      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(256, activation='relu', kernel_regularizer=regularizer),
      tf.keras.layers.Dense(128, activation=None, kernel_regularizer=regularizer),
      tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1))
  ])

  #embedding_model.summary()

  batch_size = 128
  epochs = 5
  steps_per_epoch = int(X_train.shape[0]/batch_size)

  embedding_model.compile(loss=tfa.losses.TripletSemiHardLoss(), optimizer='nadam')

  _ = embedding_model.fit(
      X_train, y_train,
      steps_per_epoch=steps_per_epoch,
      epochs=epochs, verbose=0 
  )

  X_train_emb = embedding_model.predict(X_train)
  X_test_emb = embedding_model.predict(X_test)
  X_emb = np.vstack((X_train_emb, X_test_emb))
  Y_emb = np.concatenate((y_train, y_test))
  
  res = Euclidean(X_train_emb, X_train_emb)
  _, _, cs = tf.unique_with_counts(y_train)
  acc_tr = tf.reduce_mean([tf.math.reduce_mean(tf.cast(tf.equal(tf.split(res[:, i], cs)[i], i), dtype=tf.float32)) for i in tf.range(len(cs), dtype=tf.float)])

  res = Euclidean(X_test_emb, X_train_emb)
  _, _, cs = tf.unique_with_counts(y_test)
  acc_tst = tf.reduce_mean([tf.math.reduce_mean(tf.cast(tf.equal(tf.split(res[:, i], cs)[i], i), dtype=tf.float32)) for i in tf.range(len(cs), dtype=tf.int64)])

