from model import Normal_model
import numpy as np
import torch
from torch.utils.data import Dataset
import os
import pickle
import math
import pickle
from tqdm import tqdm
import matplotlib.pyplot as plt
from dataload import physical_dataset

from torch.utils.tensorboard import SummaryWriter
import random
from scipy.integrate import quad
import cv2
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
writer = SummaryWriter()
# data_path = '/home/shared/SPAD_TOF/Data_preprocessing/physycal_model/inves_dataset/dataset_physical/board_move_0/2024-04-13-15-42-14/tof_data.pkl'
data_path = '/home/shared/SPAD_TOF/Data_preprocessing/physycal_model/inves_dataset/dataset_board/mirror5_0_0_1/2024-04-20-13-50-57/tof_data.pkl'

with open(data_path, 'rb') as f:
    tof_data = pickle.load(f)
tof_bins = tof_data[50][1]
tof_depth = tof_data[50][0]
tof_bins = np.array(tof_bins)

print(tof_bins.shape)
tof_depth = np.flip(tof_depth, 0)
tof_bins = np.flip(tof_bins, 0)
# tof_bins = tof_bins[2][3]
# tof_depth = tof_depth[2][3]
tof_bins = tof_bins[4][1]
tof_depth = tof_depth[4][1]
tof_bins=torch.tensor(tof_bins).float().to(device)
# tof_bins = tof_bins/torch.sum(tof_bins)
print(tof_depth)

model = Normal_model().to(device)
mse = torch.nn.MSELoss()
mae = torch.nn.L1Loss()
best_loss = 100000
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
for i in range(50):
    out,norm = model(1,4,tof_depth)
    loss = mse(out, tof_bins)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    # print(model.surface_normal.grad)
    print('out: ', out)
    print('loss: ', loss.item())
    writer.add_scalar('Loss/train', loss.item(),i)
    if loss.item() < best_loss:
        best_loss = loss.item()
        np.save('best_norm.npy', norm.cpu().detach().numpy())     
        print('norm', norm)


# for x in np.arange(0,1,0.1):
#     for y in np.arange(0,1,0.1):
#         for z in np.arange(0,1,0.1):
#             out,norm = model(torch.tensor([x,y,z]).float().to(device))
#             print('out: ', out)
#             print('norm: ', norm)
#             print('x: ', x)
#             print('y: ', y)
#             print('z: ', z)
#             print('-----------------')

# #plot the output tof data
# out = out.cpu().detach().numpy()
# fig, ax = plt.subplots()
# ax.bar(range(18), out)
# plt.show()

# tof_bins = tof_bins.cpu().detach().numpy()
# fig, ax = plt.subplots()
# ax.bar(range(18), tof_bins)
# plt.show()

