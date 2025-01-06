import torch
import torch.nn as nn
import torch.optim as optim
from ToFace_dataset import physical_dataset_class_FER_infer
from torch.utils.data import DataLoader, random_split
from torchvision.models.detection import fasterrcnn_resnet50_fpn, ssdlite320_mobilenet_v3_large, get_new_ssd
# use tensorboard
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import accuracy_score
import numpy as np
import time
import argparse
import re
from ToFace_1 import ToFace, ToFace_Unet,ToFace_Unet_cnn, ClassificationNetwork, ToFace_Unet_3Dcnn,ClassificationNetwork_mobilenet,ToFace_Unet_3Dcnn_complex
import pickle
import torch.nn.functional as F
import os

from teacher_Poster_v2.POSTER_V2.models.PosterV2_7cls import *
from teacher_Poster_v2.POSTER_V2.data_preprocessing.sam import SAM
import warnings
from sklearn import metrics
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
warnings.filterwarnings("ignore")
import torch.utils.data as data
import os
import argparse
from sklearn.metrics import f1_score, confusion_matrix
from teacher_Poster_v2.POSTER_V2.data_preprocessing.sam import SAM
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import matplotlib.pyplot as plt
from torchvision import transforms
import random
import cv2
import torchvision.transforms.functional as TF
from tqdm import tqdm




def test(modelname,model_path, class_model_path,data_path, out_size):
    batch_size = 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model =Unet_baseline().to(device)
    # model.load_state_dict(torch.load('weights/_2024521171048unet_-4.pth'))
    mse = nn.MSELoss()
    cosine = nn.CosineEmbeddingLoss()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if modelname == 'ToFace_Unet':
        model = ToFace_Unet().to(device)
    elif modelname == 'ToFace_Unet_cnn':
        model = ToFace_Unet_cnn(out_size).to(device)
    elif modelname == 'ToFace_Unet_3Dcnn':
        model = ToFace_Unet_3Dcnn(out_size).to(device)
    elif modelname == 'ToFace_Unet_3Dcnn_complex':
        model = ToFace_Unet_3Dcnn_complex(out_size).to(device)
    else:
        model = ToFace().to(device)

        model = ToFace_Unet().to(device)
    
    model.load_state_dict(torch.load(model_path))
    class_model = ClassificationNetwork(10, out_size).to(device)
    class_model.load_state_dict(torch.load(class_model_path))
    loss = nn.MSELoss()
    #mae loss
    loss = nn.L1Loss()

    model.eval()
    users = [1,2,3,4,5,6,7,8,9,10,11,12,13]
    if modelname == 'ToFace_Unet_3Dcnn' or modelname == 'ToFace_Unet_3Dcnn_complex':
        dataset = physical_dataset_class_FER_infer(data_path, users, False,['P1','P2'], [],True)
    else:
        dataset = physical_dataset_class_FER_infer(data_path, users, False,['P1','P2'], [],True,True)
    dataset = DataLoader(dataset, batch_size=1, shuffle=False)
    out_orientation_list = []
    out_depth_list = []
    gt_orientation_list = []
    gt_depth_list = []
    total_l1 = 0
    total_l2 = 0
    # fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    wide = 400
    height = 400
    num_row = 2
    num_colume = 4
    empty = np.zeros((wide, height, 3))
    # 转为uint8
    empty = (empty * 255).astype(np.uint8)
    n = 0
    # font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_color = (255, 255, 255)  # 白色
    line_type = 2
    total_correct = 0
    total_samples = 0
    total_pred = []
    total_true = []
    user = []
    min_depths = []
    # out = cv2.VideoWriter("ToFace_recording.mp4",fourcc,5,(wide*num_colume,height*num_row))
    
    with torch.no_grad():
        for i, (orientation,depth, tof_data, class_label,face_orientation, input_depth,input_reflectance,target,imgs, mask, uid,min_depth) in tqdm(enumerate(dataset)):
            # mask = (depth != 0).float()
            scale = 64//out_size
            mask = F.max_pool2d(mask.unsqueeze(1), scale).squeeze(1)
            orientation = F.avg_pool2d(orientation.unsqueeze(1), scale).squeeze(1)
            depth = F.avg_pool2d(depth.unsqueeze(1), scale).squeeze(1)
            masked_orientation = orientation*mask
            masked_depth = depth*mask

            if modelname == 'ToFace_Unet_3Dcnn' or modelname == 'ToFace_Unet_3Dcnn_complex':
                tof_data = tof_data
            else:
                tof_data = tof_data.permute(0,3,1,2)
            tof_data = torch.log(tof_data+1)
            # tof_data = tof_data[:,0:18,:,:]
            # out_orientation, out_depth = model(tof_data)
            out_orientation, out_depth, upsampled, classifier = model(tof_data)
            # print(targets)
            # out_orientation = out_orientation.reshape(-1, out_size, out_size)
            # out_depth = out_depth.reshape(-1, out_size, out_size)
            out_len = out_size//8
            if modelname == 'ToFace_Unet_3Dcnn' or modelname == 'ToFace_Unet_3Dcnn_complex':
                out_orientation = out_orientation.squeeze(-1)
                out_depth = out_depth.squeeze(-1)
            else:

                out_orientation = out_orientation.reshape(batch_size, 8, 8, out_len, out_len).permute(0,1,3,2,4).reshape(batch_size, out_size, out_size)
                
                out_depth = out_depth.reshape(batch_size, 8, 8, out_len, out_len).permute(0,1,3,2,4).reshape(batch_size, out_size, out_size)
            out_depth = out_depth * mask
            out_orientation = out_orientation * mask
            l1 = loss(out_orientation, masked_orientation)
            total_l1 = total_l1 + l1.item()
            l2 = loss(out_depth, masked_depth)
            total_l2 = total_l2 + l2.item()
            n = n + 1
            class_data = torch.cat((out_orientation.detach().unsqueeze(1), out_depth.detach().unsqueeze(1)), dim = 1)
            # class_data = torch.cat((orientation.unsqueeze(1), depth.unsqueeze(1)), dim = 1)
            # class_data = torch.cat((out_orientation.unsqueeze(1), out_depth.unsqueeze(1)), dim = 1)
            # classifier,out_orient = class_model(class_data)
            class_input = []
            # print(class_data.shape)
            # print(target)
            for b in range(batch_size):
                detect = target['boxes'][b][0]
                # print(detect)
                detect = detect/7*(out_size-1)
                x_min = torch.floor(detect[0]).int()
                x_max = torch.ceil(detect[2]).int()
                y_min = torch.floor(detect[1]).int()
                y_max = torch.ceil(detect[3]).int()
                x_min = torch.clamp(x_min, 0, out_size-1)
                x_max = torch.clamp(x_max, 0, out_size-1)
                y_min = torch.clamp(y_min, 0, out_size-1)
                y_max = torch.clamp(y_max, 0, out_size-1)
                tmp_class = class_data[b, :, y_min:y_max, x_min:x_max]

                # interpolate the class data to out_size*out_size
                tmp_class = F.interpolate(tmp_class.unsqueeze(0), size=(out_size, out_size), mode='bilinear', align_corners=False).squeeze(0)
                # print(tmp_class.shape)
                tmp_depth = tmp_class[1].clamp(300,1000)
                tmp_class[1] = tmp_class[1] -torch.min(tmp_depth)
                tmp_class[1] = tmp_class[1]/torch.max(tmp_class[1])
                # tmp_class[1],tmp_class[0] = synchronized_transforms(tmp_class[1].unsqueeze(0),tmp_class[0].unsqueeze(0))
                # print(tmp_class[1].shape)
                

                class_input.append(tmp_class)
            class_input = torch.stack(class_input, dim = 0)

            classifier,face_orientation_out,uid_out,need_teach = class_model(class_data)
            preds = torch.argmax(classifier, dim=1)#
            # true_labels = torch.argmax(class_label, dim=1)
            true_labels = class_label
            # transfer to int
            
            total_pred.append(preds)
            total_true.append(true_labels)
            total_correct += (preds == true_labels).sum().item()
            total_samples += true_labels.size(0)
            user.append(uid)
            min_depths.append(min_depth)
    print(f"Accuracy: {total_correct/total_samples}")
    with open(modelname+'_preds_mask.pkl', 'wb') as f:
        pickle.dump([total_pred,total_true, user, min_depths], f)
    # print(total_pred)


            

    # out.release()
    print(f"Test Loss: {total_l1/n}, {total_l2/n}")
if __name__ == '__main__':
    # model_path = '/data/share/SPAD_TOF/Data_preprocessing/physycal_model/weights/_202496222640ToFace_Unet_3Dcnn_complex_32ToFace_Unet_FER.pth'
    # class_model_path = '/data/share/SPAD_TOF/Data_preprocessing/physycal_model/weights/_202496222640ToFace_Unet_3Dcnn_complex_32ToFace_Unet_FER_class.pth'
    class_model_path = '/data/share/SPAD_TOF/Data_preprocessing/physycal_model/weights/_202497235453ToFace_Unet_3Dcnn_complex_32ToFace_Unet_FER_class.pth'
    model_path = '/data/share/SPAD_TOF/Data_preprocessing/physycal_model/weights/_202497235453ToFace_Unet_3Dcnn_complex_32ToFace_Unet_FER.pth'
    data_path = '/data/share/SPAD_TOF/Data_preprocessing/physycal_model/inves_dataset/dataset_FER/'
    modelname = 'ToFace_Unet_3Dcnn_complex'
    out_size = 32
    cuda_number = 1
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_number)
    test(modelname,model_path, class_model_path, data_path, out_size)