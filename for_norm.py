from torch.utils.data import Dataset
import torch
import os
import pickle
from model import Normal_model
import math
import pickle
from tqdm import tqdm
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from dataload import physical_dataset
import random
from scipy.integrate import quad
import cv2

class BerHuLoss(nn.Module):
    def __init__(self, threshold=0.2):
        super(BerHuLoss, self).__init__()
        self.threshold = threshold

    def forward(self, outputs, targets):
        diff = torch.abs(outputs - targets)
        c = self.threshold * torch.max(diff).item()

        # 计算小于阈值部分的损失
        mask = diff <= c
        loss1 = (diff[mask] ** 2) / (2 * c)
        
        # 计算大于阈值部分的损失
        loss2 = diff[~mask] - 0.5 * c
        
        # 综合两部分的损失
        loss = torch.cat((loss1, loss2), dim=0)
        return loss.mean()


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data_path = '/home/shared/SPAD_TOF/Data_preprocessing/physycal_model/inves_dataset/dataset_board/mirror5_1_0_1/2024-04-20-13-49-40/tof_data.pkl'
with open(data_path, 'rb') as f:
    tof_data = pickle.load(f)
tof_bins = tof_data[50][1]
tof_depth = tof_data[50][0]
tof_bins = np.array(tof_bins)


print(tof_bins.shape)
tof_depth = np.flip(tof_depth, 0)
tof_bins = np.flip(tof_bins, 0)
tof_bins = tof_bins[4][1]
tof_depth = tof_depth[4][1]
tof_bins=torch.tensor(tof_bins).float()
tof_bins = tof_bins/torch.sum(tof_bins)
print(tof_depth)

fig, ax = plt.subplots()
ax.bar(range(18), tof_bins.cpu().detach().numpy())
plt.show()

model = Normal_model()
mse = torch.nn.MSELoss()
berhuloss = BerHuLoss(0.01)
best_loss = 1000
best_norm = None
for x in np.arange(-1,1,0.1):
    for y in np.arange(-1,1,0.1):
        for z in np.arange(0.1,1,0.1):
            out,norm = model.calculate(1,4,tof_depth, torch.tensor([x,y,z]).float())
            # loss = mse(out, tof_bins)
            loss = berhuloss(out, tof_bins)
            if loss < best_loss:
                best_loss = loss
                best_norm = norm
                print('out: ', out)
                print('loss: ', loss.item())
                print('norm', norm)
                print('x: ', x)
                print('y: ', y)
                print('z: ', z)
                print('-----------------')
print(best_norm)
            