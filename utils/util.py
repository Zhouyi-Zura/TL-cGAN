import torch
import torch.nn as nn
import os
import numpy as np

def sigmoid(x):
    return 1/(1+np.exp(-x))

def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / (_range + 1e-10)

def normalization(data):
    _range = torch.max(data) - torch.min(data)
    return (data - torch.min(data)) / (_range + 1e-10)

def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / (_range + 1e-10)

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def Get_path(filter, path):
    result = []
    for maindir, _, file_name_list in os.walk(path):
        for filename in file_name_list:
            apath = os.path.join(maindir, filename)
            ext = os.path.splitext(apath)[1]
            if ext in filter:
                result.append(apath)
    return result


class SSI_Loss(nn.Module):
    def __init__(self):
        super(SSI_Loss, self).__init__()

    def forward(self, img_ori, img_p):
        img_ori = normalization(img_ori)
        img_p = normalization(img_p)
        mean_ori = torch.mean(img_ori)
        std_ori = torch.std(img_ori)
        mean_p = torch.mean(img_p)
        std_p = torch.std(img_p)
        ssi = (std_ori * mean_p) / (mean_ori * std_p)

        return ssi
