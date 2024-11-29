import numpy as np
import torch.nn.functional as F

import pywt
import pickle
import torch

from torch.utils.data import DataLoader

from util.dataset import VibrationSignalDataset


def data_enhancement(signal, win=1024, gap=1024):
    """
    Sliding window
    2022-02-05 version-1
    :param signal: vibration signal
    :param win: the length of windows
    :param gap: the stride of steps
    :return: np.ndarray
    """

    samples = []
    end = int(1 + (signal.size - win) / gap)
    start = 0
    for i in range(end):
        temp = signal[start:start + win]
        samples.append(temp)
        start += gap
    return np.array(samples)


def fold_signal(data, window_size=64):
    data = data.squeeze()  # data.shape = (3600, 1, 1024)
    print("fold_signal:", data.shape)
    old_length = data.shape[-1]
    sequence_length = old_length // window_size
    data = data.reshape(-1, window_size, sequence_length)
    return data


def wavelet2image(signal, sampling_rate, freq_dim_scale=256, wavelet_name='morl'):
    """
    :param signal: 1D temporal sequence
    :param sampling_rate: sampling rate for the sequence
    :param freq_dim_scale: frequency resolution
    :param wavelet_name: wavelet name for CWT, here we have 'morl', 'gaus', 'cmor',...
    :return: time-freq image and its reciprocal frequencies
    """
    signal = signal.squeeze()
    freq_centre = pywt.central_frequency(wavelet_name)
    cparam = 2 * freq_centre * freq_dim_scale
    scales = cparam / np.arange(1, freq_dim_scale + 1, 1)
    [cwt_matrix, frequencies] = pywt.cwt(signal, scales, wavelet_name, 1.0 / sampling_rate)

    return abs(cwt_matrix), frequencies


def image_downsampling(image_set, pooling_size=2, form='max_pooling', axis=None):
    """
    :param image_set: input image with large size
    :param pooling_size: down-sampling rate
    :param form: 'max_pooling' or 'avg_pooling'
    :param axis: if axis is not None, it means that the image will be down-sampled
                 just within it row(axis=0) or column(axis=1).
    :return: image has been down-sampled
    """
    # num, time_dim, freq_dim = image_set.shape
    image_set = image_set.unsqueeze(-1)
    kernel_size = [pooling_size, 2 * pooling_size]
    if axis == 0:
        kernel_size = [pooling_size, 1]
    elif axis == 1:
        kernel_size = [1, pooling_size]
    pooling_fn = F.max_pool2d if form == 'max_pooling' else F.avg_pool2d
    down_sampling_im = pooling_fn(image_set, kernel_size=kernel_size, stride=kernel_size)
    return down_sampling_im.squeeze(-1)


def transform_to_graph(file_name, sampling_rate, target_file_name,
                       freq_dim_scale=256, pooling_size=4):
    """
    Transform vibration data to Time-Frequency graph.
    :param file_name:
    :param sampling_rate:
    :param target_file_name:
    :param freq_dim_scale:
    :param pooling_size:
    :return:
    """
    # Load data
    with open(file_name, "rb") as f:
        samples = pickle.load(f)
    data = samples[:, :-1]
    labels = samples[:, -1]
    dataset = VibrationSignalDataset(data, labels)
    dataloader = DataLoader(dataset, batch_size=1)
    num, signal_length = data.shape
    graphs_list = torch.zeros(size=(num, freq_dim_scale, signal_length // pooling_size))
    labels_list = torch.zeros(size=(num, 1))
    # transform data
    for i, (x, y) in enumerate(dataloader):
        cwt_matrix, frequencies = wavelet2image(x.numpy(), sampling_rate, freq_dim_scale=freq_dim_scale)
        cwt_matrix = torch.FloatTensor(cwt_matrix)
        cwt_matrix = image_downsampling(cwt_matrix, pooling_size=pooling_size, axis=0)
        graphs_list[i] = cwt_matrix
        labels_list[i] = y
        if i % 10 == 0:
            print(f"completed {i / num * 100}%")

    dic = {"data": graphs_list, "label": labels_list}
    with open(target_file_name, "wb") as f:
        pickle.dump(dic, f)
