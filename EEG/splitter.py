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


name = "G_OB Chagdurov 1"
raw = mne.io.read_raw_edf(name+'.edf')     
raw.crop().load_data()
events = mne.events_from_annotations(raw, use_rounding=False)[0]

events_firsttouch = np.array([])
events_okbutton = np.array([])
ok = 3
for j in range(events.shape[0]):
    if (events[j,2] == 2):
        if (j != events.shape[0]-1):
            if (events[j+1,2] == ok):
                events_firsttouch = np.append(events_firsttouch, events[j,0])
                events_okbutton = np.append(events_okbutton, events[j+1,0])
        else:
            events_firsttouch = np.append(events_firsttouch, events[j,0])
            events_okbutton = np.append(events_okbutton, raw.times[raw.times.shape[0]-1]*125)
            
if (events_firsttouch.shape[0] < 8):
    events_firsttouch = np.array([])
    events_okbutton = np.array([])
    ok = 4
    for j in range(events.shape[0]):
        if (events[j,2] == 2):
            if (j != events.shape[0]-1):
                if (events[j+1,2] == ok):
                    events_firsttouch = np.append(events_firsttouch, events[j,0])
                    events_okbutton = np.append(events_okbutton, events[j+1,0])
            else:
                events_firsttouch = np.append(events_firsttouch, events[j,0])
                events_okbutton = np.append(events_okbutton, raw.times[raw.times.shape[0]-1]*125)

print(events_firsttouch)
print(events_okbutton)

s = events_okbutton - events_firsttouch
print(s)

max = s.max()
print(max)

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



