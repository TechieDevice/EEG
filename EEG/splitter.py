import numpy as np
import mne
import gc
import pathlib

inp = -1
while inp != '0':
    print('1 for numpy array, 2 for list, 0 for exit')
    inp = input()

    if inp == '1':
        total_data = []
        nn = []
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

                file_data = np.array([])
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
                        file_data = np.append(file_data, raw._data[:,j])
                        n[i] = int(n[i] + 1)
        
                file_data = np.reshape(file_data, (int(file_data.shape[0]/(8)), 8))
                zero_file_data = np.array([])

                p = 0
                for j in range(i):
                    for k in range(int(n.max())):
                        if ((k + int(n[j])) < n.max()):
                            zero_file_data = np.append(zero_file_data, np.zeros((8, 1)))
                        else:
                            zero_file_data = np.append(zero_file_data, file_data[p,:])
                            p = p + 1

                zero_file_data = np.reshape(zero_file_data, (int(n.max()*i), 8))
                total_data.append(zero_file_data)
                nn.append(n)

        data_with_names = []
        zero_total_data = np.array([])
        data_names = np.array([])

        names = csv[:]['Name']
        nn_max = 0
        for arr in nn:
            if nn_max < arr.max(): nn_max = arr.max()

        p = 0
        for a in range(len(total_data)):
            td = np.array(total_data[a])
            p = 0
            n = nn[a]
            for j in range(n.shape[0]):
                data_names = np.append(data_names, names[a])
                for k in range(int(nn_max)):
                    if ((k + int(n.max())) < nn_max):
                        zero_total_data = np.append(zero_total_data, np.zeros((8, 1, 1)))
                    else:
                        zero_total_data = np.append(zero_total_data, td[p,:])
                        p = p + 1

        list_len = 0
        for arr in nn:
            list_len = list_len + arr.shape[0]
        zero_total_data = np.reshape(zero_total_data, (list_len, int(nn_max), 8))
        
        np.save("data_array", zero_total_data)
        np.save("names_list", names)

        max = csv[:]['Max'].max()
        min = csv[:]['Min'].min()
        avr = (csv[:]['Avr'].sum())/csv.shape[0]
        d = np.array([("Статистика:", max, min, avr)], dtype=dtype)
        csv = np.concatenate((csv, d), axis=0)
        np.savetxt('data.csv', csv, delimiter='      ', fmt=['%s' , '%f', '%f', '%f'], header='Name                           Max                 Min                 Avr', comments='')


    if inp == '2':
        total_data = []
        nn = []
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

                file_data = np.array([])
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
                        file_data = np.append(file_data, raw._data[:,j])
                        n[i] = int(n[i] + 1)
        
                file_data = np.reshape(file_data, (int(file_data.shape[0]/(8)), 8))
                zero_file_data = np.array([])

                p = 0
                for j in range(i):
                    for k in range(int(n.max())):
                        if ((k + int(n[j])) < n.max()):
                            zero_file_data = np.append(zero_file_data, np.zeros((8, 1)))
                        else:
                            zero_file_data = np.append(zero_file_data, file_data[p,:])
                            p = p + 1

                zero_file_data = np.reshape(zero_file_data, (int(n.max()*i), 8))
                total_data.append(zero_file_data)
                nn.append(n)

        total_data_to_file = []
        p = 0
        nn_max = 0
        for arr in nn:
            if nn_max < arr.max(): nn_max = arr.max()

        for a in range(len(total_data)):
            zero_total_data = np.array([])
            td = np.array(total_data[a])
            p = 0
            n = nn[a]
            for j in range(n.shape[0]):
                for k in range(int(nn_max)):
                    if ((k + int(n.max())) < nn_max):
                        zero_total_data = np.append(zero_total_data, np.zeros((8, 1, 1)))
                    else:
                        zero_total_data = np.append(zero_total_data, td[p,:])
                        p = p + 1
            zero_total_data = np.reshape(zero_total_data, (n.shape[0], int(nn_max), 8))
            total_data_to_file.append(zero_total_data)

        print(total_data_to_file)
        np.save("data_list", total_data_to_file)

        max = csv[:]['Max'].max()
        min = csv[:]['Min'].min()
        avr = (csv[:]['Avr'].sum())/csv.shape[0]
        d = np.array([("Статистика:", max, min, avr)], dtype=dtype)
        csv = np.concatenate((csv, d), axis=0)
        np.savetxt('data.csv', csv, delimiter='      ', fmt=['%s' , '%f', '%f', '%f'], header='Name                           Max                 Min                 Avr', comments='')
