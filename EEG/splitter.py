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


dtype = [('Name', (np.str_, 15)), ('Max', np.float64), ('Min', np.float64), ('Avr', np.float64)]
csv = np.array([], dtype=dtype)
data_dir = pathlib.Path(__file__).parent.resolve()
for item in data_dir.iterdir():
    if (item.suffix == '.edf'):
        name = item.stem
        raw = mne.io.read_raw_edf(name+'.edf')
        raw.crop().load_data()
        e = mne.events_from_annotations(raw, use_rounding=False)
        events = e[0]
        dict = e[1]
        print(events)

        events_firsttouch = np.array([])
        events_okbutton = np.array([])
        first = dict['FirstTouch']
        ok = dict['OkButton']
        pad = 0
        try:
            pad = dict['Padding']
        except:
            pass
        for j in range(events.shape[0]):
            if (events[j,2] == first):
                if (j != events.shape[0]-1):
                    if (events[j+1,2] == ok):
                        events_firsttouch = np.append(events_firsttouch, events[j,0])
                        events_okbutton = np.append(events_okbutton, events[j+1,0])
                else:
                    events_firsttouch = np.append(events_firsttouch, events[j,0])
                    if pad == 0:
                        events_okbutton = np.append(events_okbutton, raw.times[raw.times.shape[0]-1]*125)
                    else:
                        events_okbutton = np.append(events_okbutton, events[j+1,0])

        print(events_firsttouch)
        print(events_okbutton)

        s = events_okbutton - events_firsttouch
        max = s.max()/125
        min = s.min()/125
        avr = (s.sum()/125)/s.shape[0]

        d = np.array([(name, max, min, avr)], dtype=dtype)
        csv = np.concatenate((csv, d), axis=0)

        data = np.array([])
        read = False
        for j in range(raw._data.shape[1]):
            if (j in events_firsttouch):
                read = True
            if (j in events_okbutton):
                read = False
            if (read):
                data = np.append(data, raw._data[:,j])
        
        data = np.reshape(data, (int(data.shape[0]/8), 8))
        np.save(name, data)

max = csv[:]['Max'].max()
min = csv[:]['Min'].min()
avr = (csv[:]['Avr'].sum())/csv.shape[0]
d = np.array([("Статистика:", max, min, avr)], dtype=dtype)
csv = np.concatenate((csv, d), axis=0)
np.savetxt('test data.csv', csv, delimiter='      ', fmt=['%s' , '%f', '%f', '%f'], header='Name                           Max                 Min                 Avr', comments='')
