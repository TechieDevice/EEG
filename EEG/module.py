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
import json
from io import StringIO
from scipy.signal import hilbert, chirp

inp = -1
X = []
while inp != 0:
    print('1 for input raw, 2 for create wavelet, 3 for input wavelet, 4 for learn, 5 for hilbert 0 for exit')
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
        X = np.load('./file.npy')
        X = np.vstack(X)
        print(X.shape)

    elif inp == '4':
        path_to_pb = './model'
        graph = tf.saved_model.load(path_to_pb)
        
    elif inp == '5':


    elif inp == '0':
        exit(0)

    else:
        pass