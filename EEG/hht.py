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
from tensorflow.keras.layers.experimental.preprocessing import Normalization
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
import json
from io import StringIO
from scipy.signal import hilbert, chirp
import emd

gpus = tf.config.list_physical_devices('GPU')
print("Num GPUs Available: ", len(gpus))
try:
    tf.config.set_logical_device_configuration(
        gpus[0],
        [tf.config.LogicalDeviceConfiguration(memory_limit=5120)])
    logical_gpus = tf.config.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
except RuntimeError as e:
    print(e)
inp = -1
X = []
Y = []
X_new = []
while inp != 0:
    print('1 for input raw, 2 for create wavelet, 3 for input wavelet, 4 for learn, 5 for phisionet load, 6 for hilbert, 0 for exit')
    inp = input()

    if inp == '1':
        for j in range(10):
            with open('line'+str(j)+'.json') as json_file:
                line = json.load(json_file)   
                n = []
                for i in range(8):
                    for d in line['data']:
                        n.append(d)
            n = np.vstack(n)
            print(n.shape)
            n1 = []
            n1.append(n)
            X.append(n1)

        X = np.vstack(X)
        print(X.shape)

    elif inp == '2':
        wavelet = 'morl'
        sampling_period = 1 / 160
        scales = np.logspace(np.log10(2), np.log10(320), 75)
        f = pywt.scale2frequency(wavelet, scales)/sampling_period
        results_tr = []
        for i in range(10):
          cwtmatr, freqs = pywt.cwt(X[i], np.logspace(np.log10(1), np.log10(320), 100), wavelet=wavelet, axis=0)
          values = np.abs(cwtmatr)
          values = np.swapaxes(values, 1, 2)
          results_tr.append(values)
          if (i == 9):
            np.save('file', results_tr)
            results_tr = []
          gc.collect()
          print(i)

        print(values.shape)

    elif inp == '3':
        X_new = np.load('./file.npy')
        print(X_new.shape)
        gc.collect()

    elif inp == '4':
        X_train, X_test, y_train, y_test = train_test_split(X_new,
                                                    X_new,
                                                    test_size=0.2)
        
        layer = Normalization(axis=-1, input_shape=(481,420))
        layer.adapt(X_train)
        layer(X_test)
        layer.mean

        #model = Sequential()
        #model.add(InputLayer(input_shape=(481,420)))
        #model.add(Bidirectional(RNN(tfa.rnn.LayerNormLSTMCell(400, activation='relu'))))
        #model.add(RepeatVector(481))
        #model.add(Bidirectional(RNN(tfa.rnn.LayerNormLSTMCell(400, activation='relu'), return_sequences=True)))
        #model.add(TimeDistributed(Dense(420)))

        model = Sequential()
        model.add(LSTM(2000, activation='relu', input_shape=(481,420)))
        model.add(RepeatVector(481))
        model.add(LSTM(2000, activation='relu', return_sequences=True))
        model.add(TimeDistributed(Dense(420)))

        optimizer = tf.keras.optimizers.Adam(0.0001, clipnorm=1.0)
        model.compile(optimizer=optimizer, loss='mae')
        model.summary()

        sc_Xtr = layer(X_train)
        sc_Xts = layer(X_test)
        gc.collect()

        model.fit(sc_Xtr, sc_Xtr, epochs=300, batch_size=32, validation_data=(sc_Xts, sc_Xts))

        
    elif inp == '5':
        SIZE= -1
        i = 0.0
        data_dir = Path(r'D:\DATASET\files')
        channels = ['O1..','O2..','P3..', 'C3..', 'F3..', 'F4..', 'C4..', 'P4..']
        for inner_dir in data_dir.iterdir():
            for item in inner_dir.iterdir():
                if (item.suffix == '.edf'):
                    x = mne.io.read_raw_edf(item.__str__())
                    print(x.info['sfreq'])
                    x = x.pick_channels(channels)
                    tmax = 3
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
        print(gc.collect())

    elif inp == '6':
        for i in range(X.shape[0]):
          per_channel_hhts = []
          for j in range(7):
            imf = emd.sift.sift(X[i, j, :])
            IP, IF, IA = emd.spectra.frequency_transform(imf, 160, 'nht')
            freq_edges, freq_bins = emd.spectra.define_hist_bins(0, 60, 60)
            hht = emd.spectra.hilberthuang(IF, IA, freq_edges)
            hht = np.nan_to_num(hht)
            hht[hht > 0.0] = 1.0
            per_channel_hhts.append(hht)
          print(i)
          X_new.append(np.stack(per_channel_hhts).reshape((481, -1)))
          #break
          if (i % 100 == 0):
            gc.collect()
          if (i == 1001):
            break
        print(gc.collect())
        X_new = np.array(X_new)
        print(X_new.shape)
        np.save('file', X_new)

    elif inp == '0':
        exit(0)

    else:
        pass