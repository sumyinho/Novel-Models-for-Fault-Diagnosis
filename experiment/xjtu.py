import os
import pickle
import random

import pandas as pd
import numpy as np

from util.preprocess import data_enhancement

# Dataset Description:
# 1.pkl: IR
# 2.pkl: OR
# 3.pkl: CAGE(保持架)
# 4.pkl: OR
# 5.pkl: OR
# Sample Frequency: 25.6kHZ
# Rotational Speed: 2250 r/min
# the number of point per sample: 1024 (25600 * 60 / 2250 ≈ 683)

root = "../dataset/XJTU-SY_Bearing_Datasets/"
random_seed = 3407
np.random.seed(3407)


def split_XJTU_dataset():
    for i in range(1, 6):
        dir_name = root + "Bearing2_" + str(i)
        dir = os.listdir(dir_name)  # read all file in the directory.
        samples = None
        for file_name in dir:
            signal = pd.read_csv(dir_name + "/" + file_name)["Vertical_vibration_signals"]
            signal = signal.to_numpy()
            signal = data_enhancement(signal, win=1024, gap=1024)
            if samples is None:  # concatenate all samples in the file.
                samples = signal
            else:
                samples = np.concatenate((samples, signal))
        # store all sample to the file:
        with open(root + str(i) + ".pkl", "wb") as f:
            pickle.dump(samples, f)


# split_XJTU_dataset()
with open(root + "1.pkl", 'rb') as f:  # IR
    IR = pickle.load(f)
    idx = np.random.choice(len(IR), 1500, replace=False)
    IR = IR[idx]
    # add labels
    labels = [0] * len(IR)
    labels = np.array(labels).reshape(-1, 1)
    IR = np.concatenate((IR, labels), axis=1)
    print("IR:", IR.shape)

with open(root + "3.pkl", 'rb') as f:
    CAGE = pickle.load(f)
    idx = np.random.choice(len(CAGE), 1500, replace=False)
    CAGE = CAGE[idx]
    labels = [1] * len(CAGE)
    labels = np.array(labels).reshape(-1, 1)
    CAGE = np.concatenate((CAGE, labels), axis=1)
    print("CAGE:", CAGE.shape)


f_or_1 = open(root + "2.pkl", 'rb')
f_or_2 = open(root + "4.pkl", "rb")
f_or_3 = open(root + "5.pkl", "rb")

f_or = [f_or_1, f_or_2, f_or_3]
OR = None
for f in f_or:
    signal = pickle.load(f)
    idx = np.random.choice(len(signal), 500, replace=False)
    signal = signal[idx]
    if OR is None:
        OR = signal
    else:
        OR = np.concatenate((OR, signal))
    f.close()
labels = [2] * len(OR)
labels = np.array(labels).reshape(-1, 1)
OR = np.concatenate((OR, labels), axis=1)
print("OR:", OR.shape)
# with open(root + "XJTU.pkl", 'wb') as f:
#     data = np.concatenate((IR, CAGE, OR))
#     pickle.dump(data, f)
