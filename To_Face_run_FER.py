import torch
import torch.nn as nn
import torch.optim as optim
from ToFace_dataset import physical_dataset_class_FER
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

import torchvision.transforms.functional as TF

def synchronized_transforms(depth_map, reflectance_map, angle_range=10, noise_level=0.05):
    # 生成随机旋转角度
    angle = torch.rand(1).item() * 2 * angle_range - angle_range
    
    # 旋转两个图像
    depth_map = TF.rotate(depth_map, angle)
    reflectance_map = TF.rotate(reflectance_map, angle)
    
    # 生成随机噪声
    depth_noise = torch.randn_like(depth_map) * noise_level
    reflectance_noise = torch.randn_like(reflectance_map) * noise_level
    
    # 添加噪声
    depth_map += depth_noise
    reflectance_map += reflectance_noise

    return depth_map, reflectance_map

labels = ['A', 'B', 'C', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O']
class RecorderMeter1(object):
    """Computes and stores the minimum loss value and its epoch index"""

    def __init__(self, total_epoch):
        self.reset(total_epoch)

    def reset(self, total_epoch):
        self.total_epoch = total_epoch
        self.current_epoch = 0
        self.epoch_losses = np.zeros((self.total_epoch, 2), dtype=np.float32)  # [epoch, train/val]
        self.epoch_accuracy = np.zeros((self.total_epoch, 2), dtype=np.float32)  # [epoch, train/val]

    def update(self, output, target):
        self.y_pred = output
        self.y_true = target

    def plot_confusion_matrix(self, cm, title='Confusion Matrix', cmap=plt.cm.binary):
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        y_true = self.y_true
        y_pred = self.y_pred

        plt.title(title)
        plt.colorbar()
        xlocations = np.array(range(len(labels)))
        plt.xticks(xlocations, labels, rotation=90)
        plt.yticks(xlocations, labels)
        plt.ylabel('True label')
        plt.xlabel('Predicted label')

        cm = confusion_matrix(y_true, y_pred)
        np.set_printoptions(precision=2)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        plt.figure(figsize=(12, 8), dpi=120)

        ind_array = np.arange(len(labels))
        x, y = np.meshgrid(ind_array, ind_array)
        for x_val, y_val in zip(x.flatten(), y.flatten()):
            c = cm_normalized[y_val][x_val]
            if c > 0.01:
                plt.text(x_val, y_val, "%0.2f" % (c,), color='red', fontsize=7, va='center', ha='center')
        # offset the tick
        tick_marks = np.arange(len(7))
        plt.gca().set_xticks(tick_marks, minor=True)
        plt.gca().set_yticks(tick_marks, minor=True)
        plt.gca().xaxis.set_ticks_position('none')
        plt.gca().yaxis.set_ticks_position('none')
        plt.grid(True, which='minor', linestyle='-')
        plt.gcf().subplots_adjust(bottom=0.15)

        # self.plot_confusion_matrix(cm_normalized, title='Normalized confusion matrix')
        # show confusion matrix
        plt.savefig('./log/confusion_matrix.png', format='png')
        # fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print('Saved figure')
        plt.show()

    def matrix(self):
        target = self.y_true
        output = self.y_pred
        im_re_label = np.array(target)
        im_pre_label = np.array(output)
        y_ture = im_re_label.flatten()
        # im_re_label.transpose()
        y_pred = im_pre_label.flatten()
        im_pre_label.transpose()
    def plot_confusion_matrix(self, y_true, y_pred, save_path=None):
        cm = confusion_matrix(y_true, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot()
        if save_path is not None:
            plt.savefig(save_path, dpi=80, bbox_inches='tight')
            print('Saved confusion matrix figure')
        plt.show()
class TVLoss(nn.Module):
    def __init__(self):
        super(TVLoss, self).__init__()

    def forward(self, x):
        batch_size, c, h, w = x.size()
        tv_h = torch.pow(x[:, :, 1:, :] - x[:, :, :-1, :], 2).sum()
        tv_w = torch.pow(x[:, :, :, 1:] - x[:, :, :, :-1], 2).sum()
        return (tv_h + tv_w) / (batch_size * c * h * w)
    

class RecorderMeter(object):
    """Computes and stores the minimum loss value and its epoch index"""

    def __init__(self, total_epoch):
        self.reset(total_epoch)

    def reset(self, total_epoch):
        self.total_epoch = total_epoch
        self.current_epoch = 0
        self.epoch_losses = np.zeros((self.total_epoch, 2), dtype=np.float32)  # [epoch, train/val]
        self.epoch_accuracy = np.zeros((self.total_epoch, 2), dtype=np.float32)  # [epoch, train/val]

    def update(self, idx, train_loss, train_acc, val_loss, val_acc):
        self.epoch_losses[idx, 0] = train_loss * 30
        self.epoch_losses[idx, 1] = val_loss * 30
        self.epoch_accuracy[idx, 0] = train_acc
        self.epoch_accuracy[idx, 1] = val_acc
        self.current_epoch = idx + 1

    # def plot_curve(self, save_path):
    #     title = 'the accuracy/loss curve of train/val'
    #     dpi = 80
    #     width, height = 1800, 800
    #     legend_fontsize = 10
    #     figsize = width / float(dpi), height / float(dpi)

    #     fig = plt.figure(figsize=figsize)
    #     x_axis = np.array([i for i in range(self.total_epoch)])  # epochs
    #     y_axis = np.zeros(self.total_epoch)

    #     plt.xlim(0, self.total_epoch)
    #     plt.ylim(0, 100)
    #     interval_y = 5
    #     interval_x = 5
    #     plt.xticks(np.arange(0, self.total_epoch + interval_x, interval_x))
    #     plt.yticks(np.arange(0, 100 + interval_y, interval_y))
    #     plt.grid()
    #     plt.title(title, fontsize=20)
    #     plt.xlabel('the training epoch', fontsize=16)
    #     plt.ylabel('accuracy', fontsize=16)

    #     y_axis[:] = self.epoch_accuracy[:, 0]
    #     plt.plot(x_axis, y_axis, color='g', linestyle='-', label='train-accuracy', lw=2)
    #     plt.legend(loc=4, fontsize=legend_fontsize)

    #     y_axis[:] = self.epoch_accuracy[:, 1]
    #     plt.plot(x_axis, y_axis, color='y', linestyle='-', label='valid-accuracy', lw=2)
    #     plt.legend(loc=4, fontsize=legend_fontsize)

    #     y_axis[:] = self.epoch_losses[:, 0]
    #     plt.plot(x_axis, y_axis, color='g', linestyle=':', label='train-loss-x30', lw=2)
    #     plt.legend(loc=4, fontsize=legend_fontsize)

    #     y_axis[:] = self.epoch_losses[:, 1]
    #     plt.plot(x_axis, y_axis, color='y', linestyle=':', label='valid-loss-x30', lw=2)
    #     plt.legend(loc=4, fontsize=legend_fontsize)

    #     if save_path is not None:
    #         fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
    #         print('Saved figure')
    #     plt.close(fig)
    def plot_curve(self, save_path=None):
        title = 'the accuracy/loss curve of train/val'
        dpi = 80
        width, height = 1800, 800
        legend_fontsize = 10
        figsize = width / float(dpi), height / float(dpi)

        fig = plt.figure(figsize=figsize)
        x_axis = np.array([i for i in range(self.total_epoch)])  # epochs
        y_axis = np.zeros(self.total_epoch)

        plt.xlim(0, self.total_epoch)
        plt.ylim(0, 100)
        interval_y = 5
        interval_x = 5
        plt.xticks(np.arange(0, self.total_epoch + interval_x, interval_x))
        plt.yticks(np.arange(0, 100 + interval_y, interval_y))
        plt.grid()
        plt.title(title, fontsize=20)
        plt.xlabel('the training epoch', fontsize=16)
        plt.ylabel('accuracy', fontsize=16)

        y_axis[:] = self.epoch_accuracy[:, 0]
        plt.plot(x_axis, y_axis, color='g', linestyle='-', label='train-accuracy', lw=2)
        plt.legend(loc=4, fontsize=legend_fontsize)

        y_axis[:] = self.epoch_accuracy[:, 1]
        plt.plot(x_axis, y_axis, color='y', linestyle='-', label='valid-accuracy', lw=2)
        plt.legend(loc=4, fontsize=legend_fontsize)

        y_axis[:] = self.epoch_losses[:, 0]
        plt.plot(x_axis, y_axis, color='g', linestyle=':', label='train-loss-x30', lw=2)
        plt.legend(loc=4, fontsize=legend_fontsize)

        y_axis[:] = self.epoch_losses[:, 1]
        plt.plot(x_axis, y_axis, color='y', linestyle=':', label='valid-loss-x30', lw=2)
        plt.legend(loc=4, fontsize=legend_fontsize)

        if save_path is not None:
            fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
            print('Saved figure')
        plt.close(fig)

    def plot_confusion_matrix(self, y_true, y_pred, save_path=None):
        cm = confusion_matrix(y_true, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot()
        if save_path is not None:
            plt.savefig(save_path, dpi=80, bbox_inches='tight')
            print('Saved confusion matrix figure')
        plt.show()

def hinge_loss(outputs, labels):
    
    labels = 2 * labels - 1
    return torch.mean(torch.clamp(1 - outputs * labels, min=0))
def entropy_loss(output):
    # 确保output的总和不是0
    total = output.sum()
    if total == 0:
        total = 1e-6

    # 计算概率分布，并确保概率值不小于1e-9
    p = output / total
    p = torch.clamp(p, min=1e-9)

    # 计算熵
    entropy = -torch.sum(p * torch.log(p))
    return entropy

def test(model,class_model, test_dataset, out_size, modelname, use_range_mask):
    model.eval()
    class_model.eval()
    entropy_loss = nn.CrossEntropyLoss()
    mae = nn.L1Loss()
    mse = nn.MSELoss()
    val_loss = 0
    mae_loss = 0
    L1 = 0
    L2 = 0
    L3 = 0
    L4 = 0
    L5 = 0
    L6 = 0
    l = 0
    L22 = 0
    total_correct = 0
    total_samples = 0
    total_pred = []
    total_true = []
    with torch.no_grad():
        for i, (orientation,depth, tof_data, class_label,face_orientation, input_depth,input_reflectance,target,imgs,mask,uid) in enumerate(test_dataset):
            # mask = (depth != 0).float()
            scale = 64//out_size
            small_mask = F.max_pool2d(mask.unsqueeze(1), 8).squeeze(1)
            mask = F.max_pool2d(mask.unsqueeze(1), scale).squeeze(1)
            orientation = F.avg_pool2d(orientation.unsqueeze(1), scale).squeeze(1)
            depth = F.avg_pool2d(depth.unsqueeze(1), scale).squeeze(1)
            # depth[depth>700] = 0
            # orientation[depth>700] = 0
            range_mask = torch.ones_like(depth)
            range_mask[depth>1000] = 0
            if use_range_mask == 1:
                mask = range_mask
            depth = depth * mask#*0.001#cnvert to meters
            orientation = orientation * mask
            depth = depth * range_mask
            orientation = orientation * range_mask

            # tof_data = tof_data*small_mask.unsqueeze(-1)#

            if modelname == 'ToFace_Unet_3Dcnn':
                tof_data = tof_data.unsqueeze(1)
            elif modelname == 'ToFace_Unet_3Dcnn_complex':
                tof_data = tof_data
            else:
                tof_data = tof_data.permute(0,3,1,2)
            tof_data = torch.log(tof_data+1)
            out_orientation, out_depth, upsampled,class_branch = model(tof_data)
            # print(targets)
            # out_orientation = out_orientation.reshape(-1, out_size, out_size)
            # out_depth = out_depth.reshape(-1, out_size, out_size)
            out_len = out_size//8
            if modelname == 'ToFace_Unet_3Dcnn' or modelname == 'ToFace_Unet_3Dcnn_complex':
                out_depth = out_depth.squeeze(-1)
                out_orientation = out_orientation.squeeze(-1)
            else:
                out_depth = out_depth.reshape(-1, 8, 8, out_len, out_len).permute(0,1,3,2,4).reshape(-1, out_size, out_size)
                out_orientation = out_orientation.reshape(-1, 8, 8, out_len, out_len).permute(0,1,3,2,4).reshape(-1, out_size, out_size)
            # out_orientation = out_orientation.reshape(-1, 8, 8, out_len, out_len).permute(0,1,3,2,4).reshape(-1, out_size, out_size)
            
            # out_depth = out_depth.reshape(-1, 8, 8, out_len, out_len).permute(0,1,3,2,4).reshape(-1, out_size, out_size)
            # out_orientation = out_orientation * range_mask
            # out_depth = out_depth * range_mask
            
            out_depth = out_depth * mask
            out_orientation = out_orientation * mask
            upsampled_sum = upsampled.view(-1,28,20,8,8).sum(dim = 2)
            # normalize the tof_data along the last dimention, the sum of the 18 elements is 1
            # tof_data = tof_data/(torch.sum(tof_data, dim = 2, keepdim = True)+1e-6)
            l1 = torch.abs(out_orientation - orientation)
            l1 = l1.sum() / mask.sum()
            # l1 = mae(out_orientation, orientation)
            l2 = torch.abs(out_depth - depth)
            l2 = l2.sum() / mask.sum()
            # l2 = mae(out_depth, depth)
            tmp_out = out_depth.clamp(300,1000)
            min_out = out_depth.view(tmp_out.shape[0], -1).min(1, keepdim=True)[0].unsqueeze(-1)
            # max_out = out_depth.view(out_depth.shape[0], -1).max(1, keepdim=True)[0].unsqueeze(-1)
            out_depth_l = (out_depth - min_out)
            out_depth_l = out_depth_l.clamp(0, 400)
            max_out = out_depth_l.view(out_depth_l.shape[0], -1).max(1, keepdim=True)[0].unsqueeze(-1)
            out_depth_l = (out_depth_l / max_out)*1200

            tmp_depth = depth.clamp(300,1000)
            min_gt = depth.view(tmp_depth.shape[0], -1).min(1, keepdim=True)[0].unsqueeze(-1)
            # max_gt = depth.view(depth.shape[0], -1).max(1, keepdim=True)[0].unsqueeze(-1)
            depth_l = (depth - min_gt)
            depth_l = depth_l.clamp(0, 400)
            max_gt = depth_l.view(depth_l.shape[0], -1).max(1, keepdim=True)[0].unsqueeze(-1)
            depth_l = (depth_l / max_gt)*1200

            l22 = mae(out_depth_l, depth_l)

            if modelname == 'ToFace_Unet_3Dcnn':
                tof_data = tof_data.squeeze(1)
                tof_data = tof_data.permute(0,3,1,2)
            elif modelname == 'ToFace_Unet_3Dcnn_complex':
                tof_data = tof_data[:,-1,:,:, :]#batch,t, h, w,c
                tof_data = tof_data.permute(0,3,1,2)
            l3 = mae(upsampled_sum, tof_data)
            # out_depth = out_depth * mask
            # out_orientation = out_orientation * mask
            class_data = torch.cat((out_orientation.unsqueeze(1), out_depth.unsqueeze(1)), dim = 1)
            # class_data = torch.cat((out_orientation.unsqueeze(1), out_depth.unsqueeze(1)), dim = 1)
            class_input = []
            
            for b in range(depth.shape[0]):
                detect = target['boxes'][b][0]
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
                # print(tmp_class.shape)
                # interpolate the class data to out_size*out_size
                tmp_class = F.interpolate(tmp_class.unsqueeze(0), size=(out_size, out_size), mode='bilinear', align_corners=False).squeeze(0)
                # tmp_class[1] = tmp_class[1] -torch.min(tmp_class[1])
                tmp_depth = tmp_class[1].clamp(300,1000)
                tmp_class[1] = tmp_class[1] -torch.min(tmp_depth)
                tmp_class[1] = tmp_class[1]/torch.max(tmp_class[1])
                class_input.append(tmp_class)
            class_input = torch.stack(class_input, dim = 0)

            classifier,face_orientation_output,uid_output,need_teach = class_model(class_data)
            
            # classifier = classifier_output[:,0:7]
            # face_orientation_output = classifier_output[:,7:]
            class_label = class_label.squeeze(-1)
            l4 = entropy_loss(classifier, class_label)
            l42 = entropy_loss(class_branch, class_label)
            l5 = mae(face_orientation_output, face_orientation)
            l6 = -entropy_loss(uid_output, uid)
            preds = torch.argmax(classifier, dim=1)#
            # true_labels = torch.argmax(class_label, dim=1)
            true_labels = class_label
            total_pred.append(preds)
            total_true.append(true_labels)
            total_correct += (preds == true_labels).sum().item()
            total_samples += true_labels.size(0)
            
            class_error = entropy_loss(classifier, class_label)

            val_loss = l1 + l2 + l3+l4+l5
            # val_loss = l1 + l2 + l3
            # print(f"Test recons Loss: {tmp_loss.item()}")
            val_loss += val_loss.item()
            L1 += l1.item()
            L2 += l2.item()
            L3 += l3.item()
            L4 += l4.item()
            L5 += l5.item()
            L6 += l42.item()
            L22 += l22.item()
            
            l+=1
    print(f"Test Loss: {val_loss/l}")
    print(f"Test L1: {L1/l}")
    print(f"Test L2: {L2/l}")
    print(f"Test L3: {L3/l}")
    print(f"Test L4: {L4/l}")
    print(f"Test L5: {L5/l}")
    print(f"Test L22: {L22/l}")
    # print(f"Test L22: {L22/l}")
    print(f"Test class_error: {class_error/l}")
    print(f"Test Accuracy: {total_correct/total_samples}")

    return val_loss/l,total_correct/total_samples, total_pred, total_true, L1/l, L2/l, L3/l, L4/l, L5/l, L22/l

def relu_loss(x):
    return torch.relu(-x).sum()  # 惩罚所有小于0的元素
def entropy_loss(output):
    # 确保output的总和不是0
    total = output.sum()
    if total == 0:
        total = 1e-6

    # 计算概率分布，并确保概率值不小于1e-9
    p = output / total
    p = torch.clamp(p, min=1e-9)

    # 计算熵
    entropy = -torch.sum(p * torch.log(p))
    return entropy

def train(data_path, modelname, out_size, use_range_mask):
    batch_size = 64
    print('batch_size:', batch_size)
    # set the seeds
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    np.random.seed(42)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    

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
    # for param in model.parameters():
    #     print(1)
    #     print(param.name, param.shape)
    detection_model = get_new_ssd(2)
    detection_model.to(device)

   
    class_model = ClassificationNetwork(10, out_size).to(device)
    # class_model.to(device)
    # class_model = ClassificationNetwork_mobilenet(10, out_size).to(device)
    teacher_model = pyramid_trans_expr2(img_size=224, num_classes=7).cuda()
    recorder = RecorderMeter(args.epochs)
    recorder1 = RecorderMeter1(args.epochs)
    checkpoint = torch.load('/data/share/SPAD_TOF/Data_preprocessing/physycal_model/teacher_Poster_v2/POSTER_V2/checkpoint/affectnet-7-model_best.pth')
    state_dict = checkpoint['state_dict']
    for key in list(state_dict.keys()):
        if 'module.' in key:
            state_dict[key.replace('module.', '')] = state_dict.pop(key)
    teacher_model.load_state_dict(state_dict)
    best_acc = checkpoint['best_acc']
    best_acc = best_acc.to()
    teacher_model.eval()
    # print(model)

    # optimizer = optim.SGD(model.parameters(), lr=0.001)
    entropy_loss = nn.CrossEntropyLoss()
    
    # optimizer = optim.Adam(model.parameters(), lr=0.01)

    # optimizer = optim.Adam(list(model.parameters())+list(class_model.parameters()), lr=0.01)
    
    if modelname == 'ToFace_Unet_3Dcnn' or modelname == 'ToFace_Unet_3Dcnn_complex':
        fer_classifier_params = list(map(id, model.classifier.parameters()))
        base_params = filter(lambda p: id(p) not in fer_classifier_params, model.parameters())
        optimizer = optim.AdamW([
            {'params': base_params, 'lr': 0.01},
            {'params': model.classifier.parameters(), 'lr': 0.0001},
            {'params': class_model.parameters(), 'lr': 0.0001}
        ])
    else:
        optimizer = optim.AdamW([
            {'params': model.parameters(), 'lr': 0.01},
            {'params': class_model.parameters(), 'lr': 0.0001}
        ])
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=80, gamma=0.1)
    # writer = SummaryWriter()
    localtime = time.localtime(time.time())
    log_name = '_'+ str(localtime.tm_year) + str(localtime.tm_mon) + str(localtime.tm_mday) + str(localtime.tm_hour) + str(localtime.tm_min) + str(localtime.tm_sec)
    print(log_name)
    writer = SummaryWriter(log_dir='runs/'+modelname+'_'+str(out_size)+'_'+log_name)
    num_epoch = 400
    best_loss = float('inf')
    best_class_loss = float('inf')
    best_model = None
    # 
    # x = torch.randn(10, 64, 18)  # 
    # y = torch.randint(0, 2, (10,))  # 0 or 1
    users = [1,2,3,4,5,6,7,8,9,10,11,12,13]
    if modelname == 'ToFace_Unet_3Dcnn' or modelname == 'ToFace_Unet_3Dcnn_complex':
        dataset = physical_dataset_class_FER(data_path, users, False,['P1','P2'], [],True)
    else:
        dataset = physical_dataset_class_FER(data_path, users, False,['P1','P2'], [],True,True)

    train_size = int(0.8 * len(dataset))
    validation_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - validation_size
    train_dataset, validation_dataset, test_dataset = random_split(dataset, [train_size, validation_size, test_size])

    



    # train_size = int(0.8 * len(dataset))
    # validation_size = len(dataset) - train_size
    # train_dataset, validation_dataset = random_split(dataset, [train_size, validation_size])
    # # test_dataset = physical_dataset_class_FER(data_path, [5,6,7,8], False)
    # test_dataset = physical_dataset_class_FER(data_path, [1,2,3,4], False,['P1','P2'],['50'],False)
    # print(len(test_dataset))
    # print(len(train_dataset))
    # test_dataset = physical_dataset_class_FER(data_path, [5,6,7,8], True)
    # validation_dataset = physical_dataset_class_FER(data_path, [5,6,7,8], True)

    train_dataset = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    validation_dataset = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    test_dataset = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    num = 0
    early_stop = 60
    pre_loss = 100000
    lambda_ = 0.01
    TVl = TVLoss()
    mae = nn.L1Loss()
    crop_size = 28
    angle_range = 10
    noise_level = 0.05
    # preprocess_transforms = transforms.Compose([
    #     transforms.ToPILImage(),  # 如果你的数据是Tensor，需要先转换为PIL Image进行图像操作
    #     transforms.RandomCrop(crop_size),
    #     transforms.RandomRotation(angle_range),
    #     transforms.ToTensor(),  # 将PIL Image转换回Tensor
    #     transforms.Lambda(lambda x: x + torch.randn_like(x) * noise_level)  # 添加随机噪声
    # ])
    for epoch in range(num_epoch):  
        model.train()
        class_model.train()
        for i, (orientation,depth, tof_data, class_label,face_orientation, input_depth,input_reflectance,target,imgs, mask, uid) in enumerate(train_dataset):
            optimizer.zero_grad()

            # input_depth = input_depth.unsqueeze(1)*0.001
            # input_reflectance = input_reflectance.unsqueeze(1)
            # # 将input_reflectance中小于10的置为0
            # input_reflectance[input_reflectance < 10] = 0
            # input_data = input_reflectance*torch.pow(input_depth, 5)
            # input_data[input_data > 10] = 0
            # input_data = list(img.to(device) for img in input_data)
            # targets = []
            # for ii in range(len(target['boxes'])):
            #     targets.append({'boxes': target['boxes'][ii].to(device), 'labels': target['labels'][ii].to(device)})

            
            # mask = (depth != 0).float()
            start_time = time.time()
            scale = 64//out_size
            small_mask = F.max_pool2d(mask.unsqueeze(1), 8).squeeze(1)
            mask = F.max_pool2d(mask.unsqueeze(1), scale).squeeze(1)
            orientation = F.avg_pool2d(orientation.unsqueeze(1), scale).squeeze(1)
            depth = F.avg_pool2d(depth.unsqueeze(1), scale).squeeze(1)
            # depth[depth>700] = 0
            # orientation[depth>700] = 0
            range_mask = torch.ones_like(depth)
            range_mask[depth>1000] = 0
            if use_range_mask == 1:
                mask = range_mask
            depth = depth * mask#*0.001#cnvert to meters
            orientation = orientation * mask
            depth = depth * range_mask
            orientation = orientation * range_mask
            # tof_data = tof_data*small_mask.unsqueeze(-1)#
            # mask = F.avg_pool2d(mask.unsqueeze(1), scale).squeeze(1)
            # print(land3D)
            if modelname == 'ToFace_Unet_3Dcnn' or modelname == 'ToFace_Unet_3Dcnn_complex':
                tof_data = tof_data
            else:
                tof_data = tof_data.permute(0,3,1,2)
            #tof取log
            tof_data = torch.log(tof_data+1)
            # print(tof_data.shape)
            # end_time = time.time()
            # print('preprocess time:', end_time-start_time)
            # start_time = time.time()
            
            

            out_orientation, out_depth, upsampled,class_branch = model(tof_data)
            # print(depth.shape)
            out_len = out_size//8
            if modelname == 'ToFace_Unet_3Dcnn' or modelname == 'ToFace_Unet_3Dcnn_complex':
                out_orientation = out_orientation.squeeze(-1)
                out_depth = out_depth.squeeze(-1)
            else:

                out_orientation = out_orientation.reshape(batch_size, 8, 8, out_len, out_len).permute(0,1,3,2,4).reshape(batch_size, out_size, out_size)
                
                out_depth = out_depth.reshape(batch_size, 8, 8, out_len, out_len).permute(0,1,3,2,4).reshape(batch_size, out_size, out_size)
            # out_orientation = out_orientation * range_mask
            # out_depth = out_depth * range_mask
            # print(out_depth.shape)
            # print(out_orientation.shape)
            # out_depth = out_depth.squeeze(-1)
            # out_orientation = out_orientation.squeeze(-1)
            out_depth = out_depth * mask
            out_orientation = out_orientation * mask
            # print(out_depth.shape)
            upsampled_sum = upsampled.view(-1,28,20,8,8).sum(dim = 2)
            # normalize the tof_data along the last dimention, the sum of the 18 elements is 1
            # tof_data = tof_data/(torch.sum(tof_data, dim = 2, keepdim = True)+1e-6)
            
            # l1 = mse(out_orientation, orientation)
            # l2 = mse(out_depth, depth)
            l1 = torch.abs(out_orientation - orientation)
            l1 = l1.sum() / mask.sum()
            l2 = torch.abs(out_depth - depth)
            l2 = l2.sum() / mask.sum()
            tmp_out = out_depth.clamp(300,1000)
            min_out = out_depth.view(tmp_out.shape[0], -1).min(1, keepdim=True)[0].unsqueeze(-1)
            # max_out = out_depth.view(out_depth.shape[0], -1).max(1, keepdim=True)[0].unsqueeze(-1)
            out_depth_l = (out_depth - min_out)
            out_depth_l = out_depth_l.clamp(0, 400)
            max_out = out_depth_l.view(out_depth_l.shape[0], -1).max(1, keepdim=True)[0].unsqueeze(-1)
            out_depth_l = (out_depth_l / (max_out+1e-4))*1200

            tmp_depth = depth.clamp(300,1000)
            min_gt = depth.view(tmp_depth.shape[0], -1).min(1, keepdim=True)[0].unsqueeze(-1)
            # max_gt = depth.view(depth.shape[0], -1).max(1, keepdim=True)[0].unsqueeze(-1)
            depth_l = (depth - min_gt)
            depth_l = depth_l.clamp(0, 400)
            max_gt = depth_l.view(depth_l.shape[0], -1).max(1, keepdim=True)[0].unsqueeze(-1)
            depth_l = (depth_l / (max_gt+1e-4))*1200
            # print(out_depth_l.shape)
            # print(depth_l.shape)
            l22 = mae(out_depth_l, depth_l)
            l2_smooth = TVl(out_depth.unsqueeze(1))

            if modelname == 'ToFace_Unet_3Dcnn':
                tof_data = tof_data.squeeze(1)
                tof_data = tof_data.permute(0,3,1,2)
            elif modelname == 'ToFace_Unet_3Dcnn_complex':
                tof_data = tof_data[:,-1,:,:, :]#batch,t, h, w,c
                tof_data = tof_data.permute(0,3,1,2)
            l3 = mse(upsampled_sum, tof_data)
            # end_time = time.time()
            # print('model time:', end_time-start_time)
            # start_time = time.time()
            out_depth = out_depth * mask
            out_orientation = out_orientation * mask
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
            # print(classifier.shape)
            # print(class_label.shape)
            # class_gt = torch.cat((class_label,face_orientation), dim=1)
            # print(classifier.shape)
            # print(class_label.shape)
            class_label = class_label.squeeze(-1)
            l4 = entropy_loss(classifier, class_label)
            
            # l42 = entropy_loss(class_branch, class_label)
            # print(classifier)
            l5 = mae(face_orientation_out, face_orientation)
            
            l6 = -entropy_loss(uid_out, uid)
            if l4.item() < pre_loss:
                lambda_ = min(lambda_ * 1.1, 1)
                pre_loss = l5.item()
            else:
                lambda_ = max(lambda_ * 0.9, 0.01)
                # pre_loss = l5.item()
            l6 = lambda_ * l6
            img = imgs.permute(0,3,1,2)
            _, teacher_output = teacher_model(img)
            l42 = cosine(teacher_output, need_teach, torch.ones(batch_size).to(device))
            l422 = entropy_loss(class_branch, class_label)#
            # l4 = entropy_loss(classifier, class_label)

            loss = l1 + l2 + l3+l4+l5#+l42+l22#+0.1*l2_smooth#
            # loss = l4+l5
            # end_time = time.time()
            # print('classifier time:', end_time-start_time)
            # start_time = time.time()
            # loss = l1 + l2 + l3
            loss.backward()

           
            optimizer.step()
            end_time = time.time()
            print('backward time:', end_time-start_time)

            print('total_loss:', loss.item())
            print('l1:', l1.item())
            print('l2:', l2.item())
            print('l3:', l3.item())
            print('l4:', l4.item())
            print('l5:', l5.item())
            print('l42:', l42.item())
            print('l22:', l22.item())
            print('smooth:', l2_smooth.item())
            # print('l4:', l4.item())


            
            print(f"Epoch {epoch+1}, Loss: {loss.item()}")

            writer.add_scalar('Loss/train', loss.item(), epoch * len(train_dataset) + i)
            writer.add_scalar('Loss/l1', l1.item(), epoch * len(train_dataset) + i)
            writer.add_scalar('Loss/l2', l2.item(), epoch * len(train_dataset) + i)
            writer.add_scalar('Loss/l3', l3.item(), epoch * len(train_dataset) + i)
            writer.add_scalar('Loss/l4', l4.item(), epoch * len(train_dataset) + i)
            writer.add_scalar('Loss/l5', l5.item(), epoch * len(train_dataset) + i)
            writer.add_scalar('Loss/l42', l42.item(), epoch * len(train_dataset) + i)
            writer.add_scalar('Loss/l22', l22.item(), epoch * len(train_dataset) + i)
            writer.add_scalar('Loss/l2_smooth', l2_smooth.item(), epoch * len(train_dataset) + i)
            # writer.add_scalar('Loss/l4', l4.item(), epoch * len(train_dataset) + i)

        val_loss, acc,_,_,l1,l2,l3,l4, l5,l22 = test(model, class_model, validation_dataset, out_size, modelname, use_range_mask)
        scheduler.step()
        writer.add_scalar('Loss/validation', val_loss, epoch)
        writer.add_scalar('Accuracy/validation', acc, epoch)
        writer.add_scalar('l1/validation', l1, epoch)
        writer.add_scalar('l2/validation', l2, epoch)
        writer.add_scalar('l3/validation', l3, epoch)
        writer.add_scalar('l4/validation', l4, epoch)
        writer.add_scalar('l5/validation', l5, epoch)
        writer.add_scalar('l22/validation', l22, epoch)
        # writer.add_scalar('l42/validation', l6, epoch)
        # writer.add_scalar('Loss/l2', l2, epoch)
        # writer.add_scalar('Loss/l3', l3, epoch)
        if epoch % 5 == 0:
            test_loss,acc,total_pred, total_true,l1,l2,l3, l4, l5,l422 = test(model,class_model, test_dataset, out_size,modelname, use_range_mask)
            writer.add_scalar('Loss/test', test_loss, epoch)
            writer.add_scalar('Accuracy/test', acc, epoch)
            writer.add_scalar('L1/test', l1, epoch)
            writer.add_scalar('L2/test', l2, epoch)
            writer.add_scalar('L3/test', l3, epoch)
            writer.add_scalar('L4/test', l4, epoch)
            writer.add_scalar('L5/test', l5, epoch)
            writer.add_scalar('L22/test', l22, epoch)
            # writer.add_scalar('L42/test', l6, epoch)


    
        num += 1
        class_loss = l4
        if class_loss < best_class_loss:
            best_class_loss = class_loss
            best_model = model.state_dict()
            class_best_model = class_model.state_dict()
        if val_loss < best_loss:
            num = 0
            best_loss = val_loss
            best_model = model.state_dict()
            # class_best_model = class_model.state_dict()

            # save coefficient and signal(tensor) to npy
        if i%5 == 0:
            save_path = 'weights/'+ log_name +modelname+'_'+str(out_size)+'ToFace_Unet_FER.pth'
            torch.save(model, save_path)
            save_path = 'weights/'+ log_name +modelname+'_'+str(out_size)+'ToFace_Unet_FER_class.pth'
            torch.save(class_model, save_path)
        if num > early_stop:
            break


    test_loss,acc,total_pred, total_true,l1,l2,l3, l4, l5,l422 = test(model,class_model, test_dataset, out_size,modelname, use_range_mask)
    with open('pred_true_FER/'+log_name +modelname+'_'+str(out_size)+'.pkl', 'wb') as f:
        pickle.dump([total_pred, total_true], f)
    writer.add_scalar('Loss/test', test_loss, 0)
    writer.add_scalar('Accuracy/test', acc, 0)
    writer.add_scalar('L1/test', l1, 0)
    writer.add_scalar('L2/test', l2, 0)
    writer.add_scalar('L3/test', l3, 0)
    writer.add_scalar('L4/test', l4, 0)
    writer.add_scalar('L5/test', l5, 0)
    writer.add_scalar('L22/test', l22, 0)
    # writer.add_scalar('L42/test', l6, 0)
    # writer.add_scalar('Loss/l2', l2, 0)
    # writer.add_scalar('Loss/l3', l3, 0)
    save_path = 'weights/'+ log_name +modelname+'_'+str(out_size)+'ToFace_Unet_FER.pth'
    torch.save(best_model, save_path)
    save_path = 'weights/'+ log_name +modelname+'_'+str(out_size)+'ToFace_Unet_FER_class.pth'
    torch.save(class_best_model, save_path)
    writer.close()
    print(log_name)



if __name__ == '__main__':
    #python To_Face_run_FER.py --data_folder inves_dataset/dataset_FER/ --model ToFace_Unet
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_folder', type=str, default='dataset/')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--model', type=str, default='ToFace_Unet')
    parser.add_argument('--out_size', type=int, default=32)
    parser.add_argument('--cuda', type=int, default=0)
    parser.add_argument('--use_rangemask', type=int, default=0)
    print(parser.parse_args())
    args = parser.parse_args()
    datafolder = args.data_folder
    epochs = args.epochs
    model_name= args.model
    out_size = args.out_size
    cuda_number = args.cuda
    range_mask = args.use_rangemask
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_number)

    train(datafolder, model_name, out_size, range_mask)