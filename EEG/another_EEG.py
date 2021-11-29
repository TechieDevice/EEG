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
from sklearn.model_selection import train_test_split, StratifiedKFold, LeaveOneOut
import scipy.io
import math
import mne
import gc
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import *
from tensorflow.keras.metrics import *
import random
from sklearn.model_selection import train_test_split
from scipy import signal
import matplotlib.pyplot as plt
from tensorflow.keras.applications import VGG16, ResNet152V2, NASNetLarge, MobileNetV3Small
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
# StandardScaler - нормализация данных
from sklearn.preprocessing import StandardScaler, label_binarize
# OneVsOneClassifier - на каждый класс отдельный классификатор, в итоге ансамбль классификаторов
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score
from sklearn.pipeline import make_pipeline

SIZE=10
i = 0.0
mne.set_log_level('CRITICAL')
X = []
Y = []

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

data_dir = Path(r'D:\DATASET\files')
for inner_dir in data_dir.iterdir():
    for item in inner_dir.iterdir():
        if (int(np.trunc(i)) == SIZE): break
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
Y = np.array(Y).ravel()
print(X.shape)
print(Y.shape)

def eeg_feature_extract_wavelet2(x):
    c = pywt.wavedec(x, 'db4', level=5)
    clear_output()
    mean = np.array([np.mean(v, axis=1) for v in c])
    std = np.array([np.std(v, axis=1) for v in c])
    rms = np.sqrt(np.array([np.mean(v**2, axis=1) for v in c]))
    print(np.concatenate((mean, std, rms), axis=0).shape)
    return np.concatenate((mean, std, rms), axis=0).ravel()
Xn = np.array([eeg_feature_extract_wavelet2(x) for x in X])
Xn = Normalizer().fit_transform(Xn)
Yn = np.copy(Y)
print(Xn.shape)

num_classes = 10
input_shape = (241, 64, 3)

learning_rate = 0.001
weight_decay = 0.0001
batch_size = 16
num_epochs = 100
image_size = 15424  # We'll resize input images to this size
patch_size = 6  # Size of the patches to be extract from the input images
num_patches = (image_size // patch_size) ** 2
projection_dim = 64
num_heads = 4
transformer_units = [
    projection_dim * 2,
    projection_dim,
]  # Size of the transformer layers
transformer_layers = 8
mlp_head_units = [2048, 1024]  # Size of the dense layers of the final classifier

t = np.linspace(-1, 1, 200, endpoint=False)
sig  = np.cos(2 * np.pi * 7 * t) + signal.gausspulse(t - 0.4, fc=2)
widths = np.arange(1, 31)
cwtmatr = signal.cwt(sig, signal.ricker, widths)
plt.imshow(cwtmatr, extent=[-1, 1, 1, 31], cmap='PRGn', aspect='auto',
           vmax=abs(cwtmatr).max(), vmin=-abs(cwtmatr).max())
plt.show()

cwtmatr = signal.cwt(X[0, 0, :], signal.morlet, np.arange(1, 50))
print(sys.getsizeof(cwtmatr)/1024/64)
plt.imshow(cwtmatr.astype(np.float32))
plt.show()

Xn2 = np.copy(Xn)
Xn = tf.keras.utils.normalize(Xn, axis=-2, order=2)
max = np.max(Xn)
min = np.min(Xn)
Xn = (Xn - min)/(max - min)
max = np.max(Xn)
Xn = np.swapaxes(np.repeat(np.expand_dims(X, 3), 3, axis=3), 1, 2)
Xn = tf.keras.utils.normalize(Xn, axis=-2, order=2)
w = Xn.shape[1]
h = Xn.shape[2]
c = Xn.shape[3]
spl = StratifiedKFold(10)
for train_index, test_index in spl.split(Xn, Yn):
  X_train, X_test = Xn[train_index], Xn[test_index]
  y_train, y_test = Yn[train_index], Yn[test_index]
  break
gc.collect()

metrics=[
         tfa.metrics.CohenKappa(len(np.unique(Yn)), name='Kappa', sparse_labels=True),
         SparseCategoricalCrossentropy(name='loss'),
         SparseCategoricalAccuracy(name='acc'),
]

plt.imshow(X[0])
plt.show()

gc.collect()
inp  = Input((w,h,c))
vgg = MobileNetV3Small(include_top=False)
vgg.trainable = True
dense1 = Dense(7000)
dr1 = Dropout(0.3)
dense2 = Dense(3500)
dr2 = Dropout(0.3)
processed = dense2(dr1(dense1(dr2(Flatten()(vgg(inp))))))
processed = Dropout(0.3)(processed)
processed = Dense(1700)(processed)
processed = Dropout(0.3)(processed)
processed = Dense(800)(processed)
processed = Dropout(0.3)(processed)
processed = Dense(512)(processed)
processed = Dropout(0.3)(processed)
processed = Dense(256)(processed)
processed = Dropout(0.3)(processed)
processed = Dense(128)(processed)
processed = Dropout(0.3)(processed)
prediction = Dense(len(np.unique(Yn)), activation='softmax')(processed)
model = Model(inputs=inp,outputs=prediction)
optimizer = Adam(0.001,decay = 1e-4)
model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), optimizer='adagrad',
                    metrics=metrics)
model.summary()

vgg.trainable = True
model.fit(X_train, y_train,
          batch_size=16,
          epochs=5, validation_data=(X_test, y_test),
          verbose=1)
