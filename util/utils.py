import pickle
import subprocess
import random

import torch
import torch.nn as nn
import numpy as np
from scipy.io import loadmat
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from torch.nn.init import xavier_normal_

from collections import defaultdict
from util.preprocess import data_enhancement


def generate_mask(n, T):
    """
    @:param n: sequence length
    @:param T: Window size
    """
    mask = torch.zeros(size=(n, n))
    for i in range(n):
        start = i - T
        end = i + T + 1
        if start < 0:
            start = 0
        if end > n:
            end = n
        mask[i][start:end] = 1
        mask[i][i + 2 * T:n:T] = 1
    return mask


def weights_init(model):
    if isinstance(model, nn.Conv1d):
        xavier_normal_(model.weight.data)


@torch.no_grad()
def evaluate_model(model, data, y_true, *args):
    model.eval()
    y_pred = model(data, *args)
    y_pred = y_pred.max(1)[1]  # predict result
    if y_pred.is_cuda:
        y_true = y_true.cpu()
        y_pred = y_pred.cpu()
    # Calculate precision
    precision = precision_score(y_true, y_pred, average="macro", zero_division=0.0)
    # Calculate recall
    recall = recall_score(y_true, y_pred, average="macro")
    # Calculate F1 score
    f1 = f1_score(y_true, y_pred, average="macro")
    # Calculate confusion matrix
    conf_matrix = confusion_matrix(y_true, y_pred)
    return precision, recall, f1, conf_matrix


def calculate_mean_and_variance(file_name):
    """
    计算均值和方差
    calculate mean and variance of accuracy, precision, recall score and f1 score.
    :param file_name:
    :return:
    """
    with open(file_name, "rb") as f:
        data = pickle.load(f)
    result = []
    # The data including accuracy_best, precision, recall, f1, conf_matrix, loss_list, time_taken and model_best
    for items in data:
        acc, precision, recall, f1, _, _, time_taken, _ = items
        print("precision list:", acc)
        result.append(torch.tensor([acc, precision, recall, f1, time_taken]))
    result = torch.stack(result)
    return result.mean(dim=0), result.var(dim=0)


def load_PU_data(root='../dataset/Paderborn_manual_damage', win=1024, gap=1024):
    """
    加载帕德伯恩数据集，该数据集包含6种常见故障
    Load PU dataset, which contains 6 health conditions.
    version-1 2022-02-04
    :param root:
    :param win:
    :param gap:
    :return:(np.ndarray, np.ndarray)
    """
    # Load Data
    normal = loadmat(root + '/normal.mat')
    ir_edm = loadmat(root + '/ir_edm.mat')
    ir_ee = loadmat(root + '/ir_ee.mat')
    or_drill = loadmat(root + '/or_drill.mat')
    or_edm = loadmat(root + '/or_edm.mat')
    or_ee = loadmat(root + '/or_ee.mat')

    # Data Enhancement
    normal = data_enhancement(normal['vibration'][0], win=win, gap=gap)
    ir_edm = data_enhancement(ir_edm['vibration'][0], win=win, gap=gap)
    ir_ee = data_enhancement(ir_ee['vibration'][0], win=win, gap=gap)
    or_drill = data_enhancement(or_drill['vibration'][0], win=win, gap=gap)
    or_edm = data_enhancement(or_edm['vibration'][0], win=win, gap=gap)
    or_ee = data_enhancement(or_ee['vibration'][0], win=win, gap=gap)
    samples = np.concatenate((normal, ir_edm, ir_ee, or_drill, or_edm, or_ee), axis=0)

    shapes = [normal.shape[0], ir_edm.shape[0], ir_ee.shape[0],
              or_drill.shape[0], or_edm.shape[0], or_ee.shape[0]]
    print("shapes:", shapes)
    for i, size in zip(range(0, 6), shapes):
        if i == 0:
            labels = [i] * size
        else:
            labels.extend([i] * size)
    labels = np.array(labels).reshape(-1, 1)
    # print("samples.shape:", samples.shape)
    # print("labels.shape:", labels.shape)
    return np.concatenate([samples, labels], axis=-1)


def load_CWRU_multi_condition_data(number_of_samples, root='../dataset/CWRU', win=1024, gap=1024):
    # Load Data
    normal = loadmat(root + '/98.mat')['X098_DE_time']
    normal_no2 = loadmat(root + '/97.mat')['X097_DE_time']
    inner = loadmat(root + '/110.mat')['X110_DE_time']
    inner_no2 = loadmat(root + '/109.mat')['X109_DE_time']
    roll = loadmat(root + '/123.mat')['X123_DE_time']
    roll_no2 = loadmat(root + '/122.mat')['X122_DE_time']
    outter_6 = loadmat(root + '/136.mat')['X136_DE_time']
    outter_6_no2 = loadmat(root + '/135.mat')['X135_DE_time']
    outter_3 = loadmat(root + '/149.mat')['X149_DE_time']
    outter_3_no2 = loadmat(root + '/148.mat')['X148_DE_time']
    outter_12 = loadmat(root + '/162.mat')['X162_DE_time']
    outter_12_no2 = loadmat(root + '/161.mat')['X161_DE_time']

    normal = data_enhancement(normal, win=win, gap=gap).squeeze()
    normal_no2 = data_enhancement(normal_no2, win=win, gap=gap).squeeze()
    inner = data_enhancement(inner, win=win, gap=gap).squeeze()
    inner_no2 = data_enhancement(inner_no2, win=win, gap=gap).squeeze()
    roll = data_enhancement(roll, win=win, gap=gap).squeeze()
    roll_no2 = data_enhancement(roll_no2, win=win, gap=gap).squeeze()
    outter_6 = data_enhancement(outter_6, win=win, gap=gap).squeeze()
    outter_6_no2 = data_enhancement(outter_6_no2, win=win, gap=gap).squeeze()
    outter_3 = data_enhancement(outter_3, win=win, gap=gap).squeeze()
    outter_3_no2 = data_enhancement(outter_3_no2, win=win, gap=gap).squeeze()
    outter_12 = data_enhancement(outter_12, win=win, gap=gap).squeeze()
    outter_12_no2 = data_enhancement(outter_12_no2, win=win, gap=gap).squeeze()

    # Shuffle Samples
    np.random.shuffle(normal)
    np.random.shuffle(normal_no2)
    np.random.shuffle(roll)
    np.random.shuffle(roll_no2)
    np.random.shuffle(inner)
    np.random.shuffle(inner_no2)
    np.random.shuffle(outter_3)
    np.random.shuffle(outter_3_no2)
    np.random.shuffle(outter_6)
    np.random.shuffle(outter_6_no2)
    np.random.shuffle(outter_12)
    np.random.shuffle(outter_12_no2)

    number = number_of_samples // 2
    norm = np.concatenate([normal[:number], normal_no2[:number]])
    inner = np.concatenate([inner[:number], inner_no2[:number]])
    roll = np.concatenate([roll[:number], roll_no2[:number]])
    outter_6 = np.concatenate([outter_6[:number], outter_6_no2[:number]])
    outter_3 = np.concatenate([outter_3[:number], outter_3_no2[:number]])
    outter_12 = np.concatenate([outter_12[:number], outter_12_no2[:number]])
    samples = np.concatenate((normal, inner, roll, outter_6, outter_3, outter_12), axis=0)

    shapes = [normal.shape[0], inner.shape[0], roll.shape[0],
              outter_6.shape[0], outter_3.shape[0], outter_12.shape[0]]
    for i, size in zip(range(0, 6), shapes):
        if i == 0:
            labels = [i] * size
        else:
            labels.extend([i] * size)
    labels = np.array(labels).reshape(-1, 1)
    # print("samples.shape:", samples.shape)
    # print("labels.shape:", labels.shape)
    return np.concatenate([samples, labels], axis=-1)


def load_CWRU_data(root='../dataset/CWRU', win=1024, gap=1024):
    """
    加载凯斯西楚大学轴承数据集，该数据集包含6种常见故障
    Load CWRU dataset, which contains 6 health conditions.
    version-1 2022-02-04
    :param root:
    :param win:
    :param gap:
    :return:(np.ndarray, np.ndarray)
    """
    # Load Data
    normal = loadmat(root + '/98.mat')['X098_DE_time']
    inner = loadmat(root + '/110.mat')['X110_DE_time']
    roll = loadmat(root + '/123.mat')['X123_DE_time']
    outter_6 = loadmat(root + '/136.mat')['X136_DE_time']
    outter_3 = loadmat(root + '/149.mat')['X149_DE_time']
    outter_12 = loadmat(root + '/162.mat')['X162_DE_time']

    # DataEnhancement
    normal = data_enhancement(normal, win=win, gap=gap).squeeze()
    inner = data_enhancement(inner, win=win, gap=gap).squeeze()
    roll = data_enhancement(roll, win=win, gap=gap).squeeze()
    outter_6 = data_enhancement(outter_6, win=win, gap=gap).squeeze()
    outter_3 = data_enhancement(outter_3, win=win, gap=gap).squeeze()
    outter_12 = data_enhancement(outter_12, win=win, gap=gap).squeeze()

    samples = np.concatenate((normal, inner, roll, outter_6, outter_3, outter_12), axis=0)

    shapes = [normal.shape[0], inner.shape[0], roll.shape[0],
              outter_6.shape[0], outter_3.shape[0], outter_12.shape[0]]
    print("shapes:", shapes)
    for i, size in zip(range(0, 6), shapes):
        if i == 0:
            labels = [i] * size
        else:
            labels.extend([i] * size)
    labels = np.array(labels).reshape(-1, 1)
    # print("samples.shape:", samples.shape)
    # print("labels.shape:", labels.shape)
    return np.concatenate([samples, labels], axis=-1)


def automatic_run_experiments(file_name_list):
    """
    Run experiments automatically.
    :param file_name_list:
    :return:
    """
    for file_name in file_name_list:
        subprocess.run(["python", file_name], check=True)
    return True


def get_k_samples(data, k=5):
    label = data[:, -1].astype('long')

    samples = defaultdict(list)

    # 将数据按类别存储
    for i in range(len(data)):
        category = label[i]  # 假设类别作为整数标识符
        samples[category].append(data[i])

    # 创建一个新的字典，用于存储每个类别的随机选取的5个样本
    random_samples = {}

    # 对每个类别的数据进行随机选取
    for category, data_list in samples.items():
        random_samples[category] = random.sample(data_list, k)

    # 输出每个类别的随机选取的5个样本
    data = []
    for category, data_list in random_samples.items():
        data.append(np.array(data_list))

    data = np.concatenate(data, axis=0)
    return data
