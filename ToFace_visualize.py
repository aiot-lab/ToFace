import torch
import torch.nn as nn
import torch.optim as optim
from ToFace_dataset import physical_dataset, physical_dataset_class_FER
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
# use tensorboard
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import accuracy_score
import numpy as np
import time
import argparse
import re
from ToFace_tmp import ToFace,Unet_baseline, Mobilenet_baseline, ToFace_Unet,ToFace_Unet_cnn
from ToFace_1 import ToFace_Unet_3Dcnn
import cv2
import torch.nn.functional as F
import os
from tqdm import tqdm

def inference(data_path,model_path, mode_name = 'ToFace_Unet_cnn', out_size = 16):
    # model = ToFace()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model =Unet_baseline().to(device)
    # model.load_state_dict(torch.load('weights/_2024521171048unet_-4.pth'))
    if mode_name == 'ToFace_Unet_cnn':
        model = ToFace_Unet_cnn(out_size).to(device)
    elif model_name == 'ToFace_Unet_3Dcnn':
        model = ToFace_Unet_3Dcnn(out_size).to(device)
    else:

        model = ToFace_Unet().to(device)
    
    model.load_state_dict(torch.load(model_path))
    loss = nn.MSELoss()
    #mae loss
    loss = nn.L1Loss()

    model.eval()
    dataset = physical_dataset_class_FER(data_path, [2], False)
    dataset = DataLoader(dataset, batch_size=1, shuffle=False)
    out_orientation_list = []
    out_depth_list = []
    gt_orientation_list = []
    gt_depth_list = []
    total_l1 = 0
    total_l2 = 0
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    wide = 400
    height = 400
    num_row = 2
    num_colume = 4
    empty = np.zeros((wide, height, 3))
    # 转为uint8
    empty = (empty * 255).astype(np.uint8)
    n = 0
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_color = (255, 255, 255)  # 白色
    line_type = 2
    out = cv2.VideoWriter("ToFace_recording.mp4",fourcc,5,(wide*num_colume,height*num_row))
    
    with torch.no_grad():
        for i, (orientation,depth, tof_data, class_label,face_orientation, input_depth,input_reflectance,target,imgs, mask, uid) in tqdm(enumerate(dataset)):
            # mask = (depth != 0).float()
            scale = 64//out_size
            mask = F.max_pool2d(mask.unsqueeze(1), scale).squeeze(1)
            orientation = F.avg_pool2d(orientation.unsqueeze(1), scale).squeeze(1)
            depth = F.avg_pool2d(depth.unsqueeze(1), scale).squeeze(1)
            masked_orientation = orientation*mask
            masked_depth = depth*mask

            if model_name == 'ToFace_Unet_3Dcnn':
                tof_data = tof_data.unsqueeze(1)
            else:
                tof_data = tof_data.permute(0,3,1,2)
            tof_data = torch.log(tof_data+1)
            # tof_data = tof_data[:,0:18,:,:]
            # out_orientation, out_depth = model(tof_data)
            out_orientation, out_depth, upsampled, classifier = model(tof_data)
            # print(targets)
            out_orientation = out_orientation.reshape(-1, out_size, out_size)
            out_depth = out_depth.reshape(-1, out_size, out_size)
            out_depth = out_depth * mask
            out_orientation = out_orientation * mask
            l1 = loss(out_orientation, masked_orientation)
            total_l1 = total_l1 + l1.item()
            l2 = loss(out_depth, masked_depth)
            total_l2 = total_l2 + l2.item()
            n = n + 1
            out_orientation_list.append(out_orientation)
            out_depth_list.append(out_depth)
            gt_orientation_list.append(orientation)
            gt_depth_list.append(depth)
            # write to the mp4 file
            out_orientation = out_orientation.squeeze(0)
            out_orientation = out_orientation.cpu().numpy()
            # eps = 300
            # out_orientation = (out_orientation - (np.min(out_orientation)-0.1*eps))/eps
            out_orientation = cv2.resize(out_orientation, (wide, height), interpolation=cv2.INTER_NEAREST)

           
            out_orientation = cv2.normalize(out_orientation, None, 0, 255, cv2.NORM_MINMAX)
            out_orientation = cv2.applyColorMap(np.uint8(out_orientation), cv2.COLORMAP_JET)
            



            out_depth = out_depth.squeeze(0)
            out_depth = out_depth.cpu().numpy()
            # eps = 300
            # out_depth = (out_depth - (np.min(out_depth)-0.1*eps))/eps

            #remove noise in the depth map
            out_depth[out_depth < 300] = 0
            out_depth[out_depth > 1000] = 0
            out_depth = out_depth/1000
            out_depth = cv2.resize(out_depth, (wide, height), interpolation=cv2.INTER_NEAREST)
            out_depth = cv2.normalize(out_depth, None, 0, 255, cv2.NORM_MINMAX)
            out_depth = cv2.applyColorMap(np.uint8(out_depth), cv2.COLORMAP_JET)
            
            orientation = orientation.squeeze(0)
            orientation = orientation.cpu().numpy()
            orientation = cv2.resize(orientation, (wide, height), interpolation=cv2.INTER_NEAREST)
            orientation = cv2.normalize(orientation, None, 0, 255, cv2.NORM_MINMAX)
            orientation = cv2.applyColorMap(np.uint8(orientation), cv2.COLORMAP_JET)
            depth = depth.squeeze(0)
            depth = depth.cpu().numpy()
            depth = cv2.resize(depth, (wide, height), interpolation=cv2.INTER_NEAREST)
            depth = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX)
            depth = cv2.applyColorMap(np.uint8(depth), cv2.COLORMAP_JET)
            masked_orientation = masked_orientation.squeeze(0)
            masked_orientation = masked_orientation.cpu().numpy()
            # eps = 300
            # masked_orientation = (masked_orientation - (np.min(masked_orientation)-0.1*eps))/eps
            masked_orientation = cv2.resize(masked_orientation, (wide, height), interpolation=cv2.INTER_NEAREST)
            masked_orientation = cv2.normalize(masked_orientation, None, 0, 255, cv2.NORM_MINMAX)
            masked_orientation = cv2.applyColorMap(np.uint8(masked_orientation), cv2.COLORMAP_JET)

            masked_depth = masked_depth.squeeze(0)
            masked_depth = masked_depth.cpu().numpy()
            # eps = 300
            # masked_depth = (masked_depth - (np.min(masked_depth)-0.1*eps))/eps
            masked_depth[masked_depth < 300] = 0
            masked_depth[masked_depth > 1000] = 0
            masked_depth = masked_depth/1000
            masked_depth = cv2.resize(masked_depth, (wide, height), interpolation=cv2.INTER_NEAREST)
            masked_depth = cv2.normalize(masked_depth, None, 0, 255, cv2.NORM_MINMAX)
            masked_depth = cv2.applyColorMap(np.uint8(masked_depth), cv2.COLORMAP_JET)

            # print(empty.shape, empty.dtype)
            # print(out_orientation.shape, out_orientation.dtype)
            # print(out_depth.shape, out_depth.dtype)
            row1 = cv2.hconcat([empty,out_orientation, out_depth, empty])
            row2 = cv2.hconcat([orientation,masked_orientation,masked_depth, depth])
            frame = cv2.vconcat([row1, row2])
            text = f'UID: {uid[0].item()}, Class: {class_label[0].item()}'
            cv2.putText(frame, text, (10, 30), font, font_scale, font_color, line_type)
            out.write(frame)
    out.release()
    print(f"Test Loss: {total_l1/n}, {total_l2/n}")



    return out_orientation_list, out_depth_list, gt_orientation_list, gt_depth_list

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = str(0)
    # data_path = '/home/shared/SPAD_TOF/Data_preprocessing/physycal_model/inves_dataset/dataset_mask_generate/'
    data_path = '/data/share/SPAD_TOF/Data_preprocessing/physycal_model/inves_dataset/dataset_FER/'

    # model_path = '/home/shared/SPAD_TOF/Data_preprocessing/physycal_model/weights/_202471721311ToFace_Unet_cnnToFace_Unet_FER.pth'
    model_path = '/data/share/SPAD_TOF/Data_preprocessing/physycal_model/weights/_20248410169ToFace_Unet_3Dcnn_32ToFace_Unet_FER.pth'
    class_path = '/data/share/SPAD_TOF/Data_preprocessing/physycal_model/weights/_20248410169ToFace_Unet_3Dcnn_32ToFace_Unet_FER_class.pth'
    out_size = 32
    model_name = 'ToFace_Unet_3Dcnn'
    out_orientation_list, out_depth_list, gt_orientation_list, gt_depth_list = inference(data_path,model_path, model_name, out_size)
    # print('inference done')
    # print(len(out_orientation_list))
    # print(len(out_depth_list))
    # print(len(gt_orientation_list))
    # print(len(gt_depth_list))
    # print(out_orientation_list[0].shape)
    # print(out_depth_list[0].shape)
    # print(gt_orientation_list[0].shape)
    # print(gt_depth_list[0].shape)
    # print(out_orientation_list[0])
    # print(out_depth_list[0])
    # print(gt_orientation_list[0])
    # print(gt_depth_list[0])

    print('done')