import pywt
import sys
import gc
import pandas as pd
from sklearn.feature_selection import *
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow import keras
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
import gc
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
import os.path as op

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
SIZE= -1
i = 0.0
mne.set_log_level('CRITICAL')
X = []
Y = []

data_dir = Path(r'D:\DATASET\files')
channels = ['O1..','O2..','P3..', 'C3..', 'F3..', 'F4..', 'C4..', 'P4..']
for inner_dir in data_dir.iterdir():
    for item in inner_dir.iterdir():
        if (item.suffix == '.edf'):
            x = mne.io.read_raw_edf(item.__str__())
            x = x.pick_channels(channels)
            tmax = 3.0
            events = mne.make_fixed_length_events(x, id=5, duration=tmax, overlap=0.)
            epochs = mne.Epochs(x, events, tmin=0, tmax=tmax, baseline=None).drop_bad()
            t = epochs.get_data()
            X.append(t)
            Y.append(np.repeat(int(np.trunc(i)), t.shape[0]))
            i = i + 0.5

X = np.vstack(X)
Y = np.concatenate(Y)
print(X.shape)
print(Y.shape)

PERSON_ID = 8

loo = LeaveOneGroupOut()
for train_index, test_index in loo.split(X, Y, groups=Y):
  y_train, y_test = Y[train_index], Y[test_index]
  i = Y[test_index][0]
  if (i == PERSON_ID):
    X_train, X_test = X[train_index], X[test_index]
    #y_train, y_test = Y[train_index], Y[test_index]
    print(y_test)
    break

# визуализация скалограммы одной записи
wavelet = 'cmor1.5-1.0'
#wavelet = 'morl'
sampling_period = 1 / 160
scales = np.logspace(np.log10(2), np.log10(320), 75)
f = pywt.scale2frequency(wavelet, scales)/sampling_period
#print(list(zip(f, scales)))
cwtmatr, freqs = pywt.cwt(X[85], np.logspace(np.log10(1), np.log10(320), 100), wavelet=wavelet, axis=1)
values = np.abs(cwtmatr)
values = np.swapaxes(values, 0, 1)
fig, axarr = plt.subplots(16, 4, figsize=(25,50)) 
for i in range(4):
  for j in range(4):
    axarr[i][j].imshow(values[i+j])

# генерация скалограмм
wavelet = 'morl'
sampling_period = 1 / 160
scales = np.logspace(np.log10(2), np.log10(320), 75)
f = pywt.scale2frequency(wavelet, scales)/sampling_period
results_tr = []
k = 4
for i in range(4000, 4355):
#for i in range(k * 1000, (k+1) * 1000, 1):
  cwtmatr, freqs = pywt.cwt(X[i], np.logspace(np.log10(1), np.log10(320), 100), wavelet=wavelet, axis=0)
  values = np.abs(cwtmatr)
  values = np.swapaxes(values, 1, 2)
  results_tr.append(values)
  if (i == 4354):
  #if (i == ((k+1) * 1000 - 1)):
    np.save(str(k), results_tr)
    results_tr = []
  gc.collect()
  print(i)

print(values.shape)
