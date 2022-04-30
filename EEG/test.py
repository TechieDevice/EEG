import pywt
import sys
import gc
import numpy as np
import matplotlib.pyplot as plt
import os
import math
import mne
import gc
import random
import os.path as op
import json
from io import StringIO
import pathlib
import pandas as pd

def max_in_list (List):
    Max = 0
    for arr in List:
        if Max < arr.max(): Max = arr.max()
    return Max

def list_len (List):
    Len = 0
    for arr in List:
        Len = Len + arr.shape[0]
    return Len

raw1 = np.array([[0, 1, 2, 0, 11, 22, 33, 0, 111, 222, 0, 2222, 2222, 3333, 1111, 0],
                 [0, 2, 1, 0, 22, 11, 33, 0, 111, 111, 0, 1111, 3333, 3333, 1111, 0],
                 [0, 1, 2, 0, 22, 33, 11, 0, 222, 222, 0, 3333, 2222, 1111, 2222, 0]])

raw2 = np.array([[0, 5, 5, 0, 44, 55, 0, 444, 666, 555, 444, 555, 0, 5555, 4444, 6666, 0],
                 [0, 4, 5, 0, 55, 55, 0, 666, 555, 444, 666, 555, 0, 4444, 5555, 5555, 0],
                 [0, 5, 5, 0, 44, 44, 0, 444, 666, 666, 444, 444, 0, 4444, 4444, 6666, 0]])

raw3 = np.array([[0, 7, 9, 7, 0, 99, 88, 99, 0, 777, 888, 0],
                 [0, 9, 8, 9, 0, 88, 88, 77, 0, 888, 888, 0],
                 [0, 7, 9, 8, 0, 77, 77, 99, 0, 777, 777, 0]])

events_firsttouch1 = np.array([1, 4, 8, 11])
events_okbutton1 = np.array([3, 7, 10, 15])

events_firsttouch2 = np.array([1, 4, 7, 13])
events_okbutton2 = np.array([3, 6, 12, 16])

events_firsttouch3 = np.array([1, 5, 9])
events_okbutton3 = np.array([4, 8, 11])

allraw = [raw1, raw2, raw3]
allevents_firsttouch = np.array([events_firsttouch1, events_firsttouch2, events_firsttouch3])
allevents_okbutton = np.array([events_okbutton1, events_okbutton2, events_okbutton3])

names = ["raw1", "raw2", "raw3"]

totaldata = []
nn = []

for iraw in range(3):
    raw = allraw[iraw]
    events_firsttouch = allevents_firsttouch[iraw]
    events_okbutton = allevents_okbutton[iraw]

    data = np.array([])
    read = False
    i = 0
    n = np.array([])
    for j in range(raw.shape[1]):
        if (j in events_firsttouch):
            read = True
            n = np.append(n, 0)
        if (j in events_okbutton):
            read = False
            i = i + 1
        if (read):
            data = np.append(data, raw[:,j])
            n[i] = int(n[i] + 1)

    data = np.reshape(data, (int(data.shape[0]/(3)), 3))
    zerodata = np.array([])

    p = 0
    for j in range(i):
        for k in range(int(n.max())):
            if ((k + int(n[j])) < n.max()):
                zerodata = np.append(zerodata, np.zeros((3, 1)))
            else:
                zerodata = np.append(zerodata, data[p,:])
                p = p + 1

    zerodata = np.reshape(zerodata, (int(n.max()*i), 3))
    totaldata.append(zerodata)
    nn.append(n)

data_with_names = []
zerototaldata = np.array([])
data_names = np.array([])
p = 0
nn_max = int(max_in_list(nn))
for a in range(len(totaldata)):
    td = np.array(totaldata[a])
    p = 0
    n = nn[a]
    for j in range(n.shape[0]):
        data_names = np.append(data_names, names[a])
        for k in range(nn_max):
            if ((k + int(n.max())) < nn_max):
                zerototaldata = np.append(zerototaldata, np.zeros((3, 1, 1)))
            else:
                zerototaldata = np.append(zerototaldata, td[p,:])
                p = p + 1

zerototaldata = np.reshape(zerototaldata, (list_len(nn), nn_max, 3))
data_with_names.append(zerototaldata)
data_with_names.append(data_names)
print(data_with_names)