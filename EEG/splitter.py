import numpy as np
import mne
import gc
import pathlib

totaldata = []
nn = np.array([])
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

        if s.shape[0] < 10:
            continue
        d = np.array([(name, max, min, avr)], dtype=dtype)
        csv = np.concatenate((csv, d), axis=0)

        data = np.array([])
        read = False
        i = 0
        n = np.array([])
        for j in range(raw._data.shape[1]):
            if (j in events_firsttouch):
                read = True
                n = np.append(n, 0)
            if (j in events_okbutton):
                read = False
                i = i + 1
            if (read):
                data = np.append(data, raw._data[:,j])
                n[i] = int(n[i] + 1)
        
        data = np.reshape(data, (int(data.shape[0]/(8)), 8))
        zerodata = np.array([])

        p = 0
        for j in range(i):
            for k in range(int(n.max())):
                if ((k + int(n[j])) < n.max()):
                    zerodata = np.append(zerodata, np.zeros((8, 1)))
                else:
                    zerodata = np.append(zerodata, data[p,:])
                    p = p + 1

        zerodata = np.reshape(zerodata, (int(n.max()*i), 8))
        totaldata.append(zerodata)
        nn = np.append(nn, n)
        nn = np.reshape(nn, (len(totaldata),n.shape[0]))


zerototaldata = np.array([])
p = 0
for a in range(len(totaldata)):
    td = np.array(totaldata[a])
    p = 0
    for j in range(i):
        for k in range(int(nn.max())):
            n = nn[a]
            if ((k + int(n.max())) < nn.max()):
                zerototaldata = np.append(zerototaldata, np.zeros((8, 1, 1)))
            else:
                zerototaldata = np.append(zerototaldata, td[p,:])
                p = p + 1

zerototaldata = np.reshape(zerototaldata, (len(totaldata), i, int(nn.max()), 8))
print(zerototaldata)


np.save("data_array", zerototaldata)
max = csv[:]['Max'].max()
min = csv[:]['Min'].min()
avr = (csv[:]['Avr'].sum())/csv.shape[0]
d = np.array([("Статистика:", max, min, avr)], dtype=dtype)
csv = np.concatenate((csv, d), axis=0)
np.savetxt('test data.csv', csv, delimiter='      ', fmt=['%s' , '%f', '%f', '%f'], header='Name                           Max                 Min                 Avr', comments='')
