import torch
import torch.nn as nn
import torch.optim as optim
from ToFace_dataset import physical_dataset_mask
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
# use tensorboard
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import accuracy_score
import numpy as np
import time
import argparse
import re
from To_Face_maskGenerate import MobileNetDetection
import cv2
from To_Face_maskGenerate_Train import calculate_iou
import matplotlib.patches as patches

def inference(data_path, data_type = 'rd'):
    # model = ToFace()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model =Unet_baseline().to(device)
    # model.load_state_dict(torch.load('weights/_2024521171048unet_-4.pth'))
    model = model = MobileNetDetection().to(device)
    if data_type == 'rd':
        model.load_state_dict(torch.load('weights/_202472215016mask.pth'))
    elif data_type == 'r':
        model.load_state_dict(torch.load('weights/_20247311942mask.pth'))
    else:
        model.load_state_dict(torch.load('weights/_202472215212mask.pth'))
    # loss = nn.MSELoss()
    #mae loss
    loss = nn.L1Loss()

    model.eval()
    dataset = physical_dataset_mask(data_path)
    dataset = DataLoader(dataset, batch_size=1, shuffle=True)
    out_detect_list = []
    gt_detect_list = []
    gt_orientation_list = []
    gt_depth_list = []
    total_loss = 0
    total_iou = 0
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    wide = 400
    height = 400
    num_row = 1
    num_colume = 1
    n = 0
    mae = nn.L1Loss()
    out = cv2.VideoWriter("ToFace_mask_generate.mp4",fourcc,5,(wide*num_colume,height*num_row))
    
    with torch.no_grad():
        for i, (orientation,depth, detect, input_depth, input_reflectance) in enumerate(dataset):
            mask = (depth != 0).float()
            # tof_data = tof_data.permute(0,3,1,2)
            if data_type == 'rd':
                input_depth = input_depth.unsqueeze(1)*0.001
                input_reflectance = input_reflectance.unsqueeze(1)
                input_data = input_reflectance*torch.pow(input_depth, 5)
            elif data_type == 'r':
                input_reflectance = input_reflectance.unsqueeze(1)
                input_data = input_reflectance
            else:
                input_depth = input_depth.unsqueeze(1)*0.001
                input_data = input_depth
            # out_orientation, out_depth = model(tof_data)
            out_detect = model(input_data)
            # print(targets)
            # out_orientation = out_orientation.reshape(-1, 16, 16)
            # out_depth = out_depth.reshape(-1, 16, 16)
            # out_depth = out_depth * mask
            # out_orientation = out_orientation * mask
            loss = mae(out_detect, detect)
            total_loss = total_loss + loss.item()
            iou = calculate_iou(detect, out_detect)
            iou = iou.mean()
            total_iou = total_iou + iou.item()
            # l1 = loss(out_orientation, orientation)
            # total_l1 = total_l1 + l1.item()
            # l2 = loss(out_depth, depth)
            # total_l2 = total_l2 + l2.item()
            n = n + 1
            out_detect_list.append(out_detect)
            gt_detect_list.append(detect)
            # out_orientation_list.append(out_orientation)
            # out_depth_list.append(out_depth)
            # gt_orientation_list.append(orientation)
            # gt_depth_list.append(depth)
            # write to the mp4 file
            
            input_data = input_data.squeeze(0)
            input_data = input_data.squeeze(0)
            input_data = input_data.cpu().numpy()
            input_data = cv2.resize(input_data, (wide, height), interpolation=cv2.INTER_NEAREST)
            # print(input_data.shape)
           
            input_data = cv2.normalize(input_data, None, 0, 255, cv2.NORM_MINMAX)
            # print(input_data.shape)
            input_data = cv2.applyColorMap(np.uint8(input_data), cv2.COLORMAP_JET)
            out_detect = out_detect.squeeze(0)
            out_detect = out_detect.cpu().numpy()
            x1, y1, width1, height1 = out_detect
            x2, y2, width2, height2 = detect.squeeze(0).cpu().numpy()
            print(x1, y1, width1, height1)
            print(x2, y2, width2, height2)
            print('------------------------------')
            fig, ax = plt.subplots(1)

            # 显示图像
            ax.imshow(input_data)

            # 创建矩形检测框1
            rect1 = patches.Rectangle((x1*wide, y1*height), width1*wide, height1*height, linewidth=2, edgecolor='r', facecolor='none')
            # 创建矩形检测框2
            rect2 = patches.Rectangle((x2*wide, y2*height), width2*wide, height2*height, linewidth=2, edgecolor='b', facecolor='none')

            # 将检测框添加到轴中
            ax.add_patch(rect1)
            ax.add_patch(rect2)

            # 去掉坐标轴
            ax.axis('off')
            fig.canvas.draw()
            image_with_boxes = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            image_with_boxes = image_with_boxes.reshape(fig.canvas.get_width_height()[::-1] + (3,))

            plt.close(fig)
            image_with_boxes = cv2.resize(image_with_boxes, (wide, height), interpolation=cv2.INTER_NEAREST)
            # convert rgb to bgr
            image_with_boxes = cv2.cvtColor(image_with_boxes, cv2.COLOR_RGB2BGR)


            # out_depth = out_depth.squeeze(0)
            # out_depth = out_depth.cpu().numpy()
            # out_depth = cv2.resize(out_depth, (wide, height), interpolation=cv2.INTER_NEAREST)
            # out_depth = cv2.normalize(out_depth, None, 0, 255, cv2.NORM_MINMAX)
            # out_depth = cv2.applyColorMap(np.uint8(out_depth), cv2.COLORMAP_JET)
            # orientation = orientation.squeeze(0)
            # orientation = orientation.cpu().numpy()
            # orientation = cv2.resize(orientation, (wide, height), interpolation=cv2.INTER_NEAREST)
            # orientation = cv2.normalize(orientation, None, 0, 255, cv2.NORM_MINMAX)
            # orientation = cv2.applyColorMap(np.uint8(orientation), cv2.COLORMAP_JET)
            # depth = depth.squeeze(0)
            # depth = depth.cpu().numpy()
            # depth = cv2.resize(depth, (wide, height), interpolation=cv2.INTER_NEAREST)
            # depth = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX)
            # depth = cv2.applyColorMap(np.uint8(depth), cv2.COLORMAP_JET)
            # print (out_orientation.shape)
            # print (out_depth.shape)
            # row1 = cv2.hconcat([out_orientation, out_depth])
            # row2 = cv2.hconcat([orientation, depth])
            # frame = cv2.vconcat([row1, row2])
            out.write(image_with_boxes)
    out.release()
    print(f"Test Loss: {total_loss/n}, {total_iou/n}")



    return out_detect_list, gt_detect_list

if __name__ == '__main__':
    data_path = '/home/shared/SPAD_TOF/Data_preprocessing/physycal_model/inves_dataset/dataset_mask_generate/'

    out_detect_list, gt_detect_list = inference(data_path)
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