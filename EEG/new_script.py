
import pandas as pd
df = pd.read_csv(filepath_or_buffer=r'D:\DATASET\EEG_header\Participants_LEMON.csv', sep=';', index_col=0)


SIZE = -1
import numpy as np
import matplotlib.pyplot as plt

import mne
import pywt
from mne.time_frequency import psd_welch
from pathlib import WindowsPath

# функция извлечения признаков c помощью метода PSD
def eeg_feature_extract_power_band(epochs):
    """EEG relative power band feature extraction.

    This function takes an ``mne.Epochs`` object and creates EEG features based
    on relative power in specific frequency bands that are compatible with
    scikit-learn.

    Parameters
    ----------
    epochs : Epochs
        The data.

    Returns
    -------
    X : numpy array of shape [n_samples, 5]
        Transformed data.
    """
    # specific frequency bands
    FREQ_BANDS = {"delta": [0.5, 4.5],
                  "theta": [4.5, 8.5],
                  "alpha": [8.5, 11.5],
                  "sigma": [11.5, 15.5],
                  "beta": [15.5, 30]}

    psds, freqs = psd_welch(epochs, picks='eeg', fmin=0.5, fmax=30., n_per_seg = 4*250)
    # Normalize the PSDs
    psds /= np.sum(psds, axis=-1, keepdims=True)

    X = []
    for fmin, fmax in FREQ_BANDS.values():
        psds_band = psds[:, :, (freqs >= fmin) & (freqs < fmax)].mean(axis=-1)
        X.append(psds_band.reshape(len(psds), -1))

    return np.concatenate(X, axis=1)

# функция извлечения признаков c помощью вейвлет-преобразования
def eeg_feature_extract_wavelet(epochs):
    x = pywt.wavedec(epochs.get_data(), 'db4', level=5)[1]
    mean = np.mean(x, axis=2)
    std = np.std(x, axis=2)
    rms = np.sqrt(np.mean(x**2, axis=2))
    return np.concatenate((mean, std, rms), axis=1)

# перечень каналов из которых извлекаем данные, их у нас в итоге 17
chn_eo_ec = ['AF3', 'C2', 'C3', 'CP3', 'CP5', 'F4', 'Oz', 'P1', 'P2', 'P4', 
       'P5', 'P6', 'P8', 'PO3', 'PO4', 'PO8', 'Pz']

# указываем путь к данным
data_dir = WindowsPath(r'D:\DATASET\EEG_Preprocessed')
epochs = []
X = []
Y = []
label_id = {}
i = 0

do = 1
if(do == 1):
# итерируем по людям (sub-032301, sub-032302 и т.д.)
    for inner_dir in data_dir.iterdir():
        # итерируем по состояниям (Eyes open and Eyes closed (EO и EC в названии файла))
        for item in inner_dir.iterdir():
            if (item.suffix=='.set'):
                x = mne.io.read_raw_eeglab(item.__str__())
                # выбираем нужные каналы
                x = x.pick_channels(chn_eo_ec)
                # устанавливаем длительность эпохи
                tmax = 2 - 1. / x.info['sfreq']
                # на каждой записи ээг отмечаются события (открыл глаза, закрыл и т.д.)
                # считываем какие есть метки событий в записи
                events, event_id = mne.events_from_annotations(x)
                ids = []
                # убираем все события кроме тех, по которым будем резать запись на эпохи 
                # эти события для 2 сек нарезки уже есть заранее
                if '32766' in event_id:
                    ids.append(event_id['32766'])
                    event_id.pop('32766')
                if '5' in event_id:
                    ids.append(event_id['5'])
                    event_id.pop('5')
                if 'boundary' in event_id:
                    ids.append(event_id['boundary'])
                    event_id.pop('boundary')
                events = mne.pick_events(events, exclude=ids)
                # делим на эпохи и отбрасываем плохие
                epochs_train = mne.Epochs(x, events, event_id, tmin=0., tmax=tmax, reject=None, event_repeated='merge', baseline=None).drop_bad()
                # X - массив входных данных без меток
                X.append(eeg_feature_extract_wavelet(epochs_train))
                # Y - массив классовых меток объектов
                Y.append(np.full((X[-1].shape[0],), i))
                label_id[str(i)] = item.stem[:-3]

        i = i + 1
        # if (i==SIZE): break
        print(i)


    # Сохраняем в отдельный файл для удобства
    dataX = np.array(X)
    dataY = np.array(Y)
    np.save(r'D:\DATASET\processed_dataX.npy', dataX)
    np.save(r'D:\DATASET\processed_dataY.npy', dataY)

    X = np.concatenate(X, axis=0)
    Y = np.concatenate(Y, axis=0)

    concatenated_dataX = np.array(X)
    concatenated_dataY = np.array(Y)
    np.save(r'D:\DATASET\concatenated_dataX.npy', concatenated_dataX)
    np.save(r'D:\DATASET\concatenated_dataY.npy', concatenated_dataY)

# %%
# print(epochs_train)


# %%
X = np.load(r'D:\DATASET\concatenated_dataX.npy')
Y = np.load(r'D:\DATASET\concatenated_dataY.npy')
print(X.shape, Y.shape)


# %%
np.array(X[1]).shape


# %%
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
# StandardScaler - нормализация данных
from sklearn.preprocessing import StandardScaler
# OneVsOneClassifier - на каждый класс отдельный классификатор, в итоге ансамбль классификаторов
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.pipeline import make_pipeline
Xt= X
Yt = Y
kf = StratifiedKFold(n_splits=5)
kf.get_n_splits(Xt)
accs = []
class_results = []
# Кроссвалидация по 5 разбиениям
for train_index, test_index in kf.split(Xt, Yt):
    X_train, X_test = Xt[train_index], Xt[test_index]
    y_train, y_test = Yt[train_index], Yt[test_index]
    for i in range(0, 5):
        model = make_pipeline(StandardScaler(),
                        OneVsRestClassifier(LinearSVC(tol=1e-6, dual=False, penalty='l2', max_iter=2000, C=0.15)))
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        report = confusion_matrix(y_test, y_pred)
        temp = []
        for i in range(len(report)):
            hits = report[i,i]
            misses = np.sum(report[i]) - hits
            temp.append([hits, misses])
        class_results.append(temp)
        print(acc)
        accs.append(acc)
class_results = np.mean(np.array(class_results), axis=0)
df_m = df.copy(deep=True)
df_m['hits'] = -1
df_m['misses'] = -1
for i in range(len(class_results)):
    label = label_id[str(i)]
    df_m.loc[[label], ['hits']] = class_results[i, 0]
    df_m.loc[[label], ['misses']] = class_results[i, 1]
print('mean acc: ' + str(np.mean(accs)))


# %%
print(accs)


# %%
len(df_m[:SIZE])
report


# %%
df_m_nomissing = df_m[df_m.index.isin(label_id.values())]


# %%
temp = df_m_nomissing[:SIZE].groupby('gender').sum()
temp.div(temp.sum(axis=1), axis=0)


# %%
temp = df_m_nomissing[:SIZE].groupby('age').sum()[['hits', 'misses']].copy(deep=True)
temp.div(temp.sum(axis=1), axis=0)


# %%



