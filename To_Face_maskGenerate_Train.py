import torch
import torch.nn as nn
import torch.optim as optim
from ToFace_dataset import physical_dataset_mask, physical_dataset_class_FER
from torch.utils.data import DataLoader, random_split

# use tensorboard
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn, ssdlite320_mobilenet_v3_large, get_new_ssd
from sklearn.metrics import accuracy_score
import numpy as np
import time
import argparse
import re
# from ToFace_1 import ToFace, ToFace_Unet
from To_Face_maskGenerate import MobileNetDetection
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.anchor_utils import DefaultBoxGenerator
from torchvision.models.detection.backbone_utils import BackboneWithFPN
from torchvision.models.detection.ssdlite import SSDLiteClassificationHead, SSDLiteRegressionHead
import os

def calculate_iou(boxes1, boxes2):
    """
    计算两个批次的边界框的 IoU
    :param boxes1: 形状为 (batch, 4) 的张量
    :param boxes2: 形状为 (batch, 4) 的张量
    :return: 形状为 (batch,) 的张量，表示每对边界框的 IoU
    """
    # 将边界框的表示转换为 (x1, y1, x2, y2)
    if isinstance(boxes1, list):
        boxes1 = torch.stack(boxes1)
    if isinstance(boxes2, list):
        boxes2 = torch.stack(boxes2)
    boxes1_x1 = boxes1[:, 0]
    boxes1_y1 = boxes1[:, 1]
    boxes1_x2 = boxes1[:, 0] + boxes1[:, 2]
    boxes1_y2 = boxes1[:, 1] + boxes1[:, 3]

    boxes2_x1 = boxes2[:, 0]
    boxes2_y1 = boxes2[:, 1]
    boxes2_x2 = boxes2[:, 0] + boxes2[:, 2]
    boxes2_y2 = boxes2[:, 1] + boxes2[:, 3]

    # 计算交集区域的坐标
    inter_x1 = torch.max(boxes1_x1, boxes2_x1)
    inter_y1 = torch.max(boxes1_y1, boxes2_y1)
    inter_x2 = torch.min(boxes1_x2, boxes2_x2)
    inter_y2 = torch.min(boxes1_y2, boxes2_y2)

    # 计算交集区域的面积
    inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)

    # 计算每个边界框的面积
    boxes1_area = (boxes1_x2 - boxes1_x1) * (boxes1_y2 - boxes1_y1)
    boxes2_area = (boxes2_x2 - boxes2_x1) * (boxes2_y2 - boxes2_y1)

    # 计算 IoU
    iou = inter_area / (boxes1_area + boxes2_area - inter_area)

    return iou


def test(model, test_dataset, data_type, model_name):
    model.eval()
    
    mae = nn.L1Loss()
    mse = nn.MSELoss()
    val_loss = 0
    mae_loss = 0
    total_iou = 0
    total_acc = 0
    l = 0
    with torch.no_grad():
        for i, (orientation,depth, tof_data, class_label,face_orientation, input_depth,input_reflectance,target,imgs,mask,uid) in enumerate(test_dataset):

            if data_type == 'rd':
                input_depth = input_depth.unsqueeze(1)*0.001
                input_reflectance = input_reflectance.unsqueeze(1)
                input_reflectance[input_reflectance < 5] = 0
                input_data = input_reflectance*torch.pow(input_depth, 4)
                input_data[input_data > 10] = 0
            elif data_type == 'r':
                input_reflectance = input_reflectance.unsqueeze(1)
                input_data = input_reflectance
            else:
                input_depth = input_depth.unsqueeze(1)*0.001
                input_data = input_depth
            # print(land3D)
            # tof_data = detect.permute(0,3,1,2)
            # print(input_data)
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            if model_name == 'fastrcnn' or model_name == 'ssd' or model_name == 'New_ssd':
                input_data = list(img.to(device) for img in input_data)
                predictions = model(input_data)

                # Assuming detect is the ground truth boxes and labels
                # Format detect to match the output format of Fast R-CNN
                targets = []
                if model_name == 'New_ssd':
                    for ii in range(len(target['boxes'])):
                        targets.append({'boxes': target['boxes'][ii].to(device), 'labels': target['labels'][ii].to(device)})
                else:
                    # targets = []
                    for ii in range(len(target['boxes'])):
                        targets.append({'boxes': target['boxes'][ii].to(device), 'labels': class_label[ii].to(device)})
                # true_boxes = [d['boxes'] for d in targets]
                # true_labels = [d['labels'] for d in targets]
                # print(len(predictions))
                # pred_boxes = [p['boxes'] for p in predictions]
                # pred_labels = [p['labels'] for p in predictions]
                # pred_scores = [p['scores'] for p in predictions]
                # print(pred_boxes[0].shape)

                # print(len(true_boxes))
                batch_size = len(predictions)
                pred_boxes = []
                true_boxes = []
                pred_labels = []
                true_labels = []
                for j in range(batch_size):
                    if len(predictions[j]['boxes']) > 0:  # 检查是否有检测框
                        pred_box = predictions[j]['boxes'][0]  # 选择置信度最高的预测框
                        pred_label = predictions[j]['labels'][0]
                        true_box = targets[j]['boxes'][0]
                        true_label = targets[j]['labels'][0]
                        pred_boxes.append(pred_box)
                        true_boxes.append(true_box)
                        pred_labels.append(pred_label)
                        true_labels.append(true_label)
                    else:
                        continue  # 如果没有检测框，跳过该样本
                if pred_boxes:
                    pred_boxes = torch.stack(pred_boxes)
                    true_boxes = torch.stack(true_boxes)
                    pred_labels = torch.stack(pred_labels)
                    true_labels = torch.stack(true_labels)
                    #calculate the acc of the labels
                    # acc = accuracy_score(true_labels, pred_labels)
                    correct = (true_labels == pred_labels).sum().item()
                    total = true_labels.size(0)
                    acc = correct / total
                    total_acc += acc
                    # Calculate IoU and other metrics
                    iou = calculate_iou(pred_boxes, true_boxes)
                    iou = iou.mean()
                    total_iou += iou.item()
                    l += 1

                    # for j in range(len(pred_boxes)):
                    #     loss = mae(pred_boxes[j], true_boxes[j])
                    #     # loss += nn.CrossEntropyLoss()(pred_labels[j], true_labels[j])
                    #     val_loss += loss.item()
                    loss = mae(pred_boxes, true_boxes)
                    val_loss += loss.item()
            # else:
            #     out_detect = model(input_data)

            #     val_loss = mae(out_detect, detect)
            #     # print(f"Test recons Loss: {tmp_loss.item()}")
            #     val_loss += val_loss.item()
            #     iou = calculate_iou(detect, out_detect)
            #     iou = iou.mean()
            #     iou = iou.item()
            #     total_iou += iou
            #     l+=1
    if l == 0:
        l = 1
    print(f"Test Loss: {val_loss/l}")
    print(f"Test IoU: {total_iou/l}")
    print(f"Test Acc: {total_acc/l}")
    
    return val_loss/l, total_iou/l, total_acc/l


def train(data_path, model_name, data_type):
    batch_size = 16
    # set the seeds
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    np.random.seed(42)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    

    mse = nn.MSELoss()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # if modelname == 'ToFace_Unet':
    #     model = ToFace_Unet().to(device)
    # else:
    if model_name == 'fastrcnn':
        model = fasterrcnn_resnet50_fpn(pretrained=False, num_classes=8)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        # if data_type == 'r':
        #     first_conv = model.backbone.body.conv1
        #     new_first_conv = nn.Conv2d(2, first_conv.out_channels, 
        #                             kernel_size=first_conv.kernel_size, 
        #                             stride=first_conv.stride, 
        #                             padding=first_conv.padding, 
        #                             bias=first_conv.bias is not None)
        #     model.backbone.body.conv1 = new_first_conv

        # Replace the pre-trained head with a new one
        # model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes=2)  # Assuming 2 classes (background + target)
        model.to(device)
    elif model_name == 'ssd':
        model = ssdlite320_mobilenet_v3_large(pretrained=False, num_classes = 8)
        print(model.backbone)
        # if data_type == 'r':
        #     conv_layer = model.backbone.features[0][0][0]

        #     # 创建一个新的卷积层，输入通道数为自定义的通道数，输出通道数与原始模型相同
        #     new_conv_layer = nn.Conv2d(2, conv_layer.out_channels, 
        #                             kernel_size=conv_layer.kernel_size, 
        #                             stride=conv_layer.stride, 
        #                             padding=conv_layer.padding, 
        #                             bias=conv_layer.bias is not None)

        #     # 重新初始化新卷积层的权重
        #     nn.init.kaiming_normal_(new_conv_layer.weight, mode='fan_out', nonlinearity='relu')
        #     if conv_layer.bias is not None:
        #         new_conv_layer.bias.data = conv_layer.bias.data

        #     # 替换原始模型的第一个卷积层
        #     model.backbone.features[0][0][0] = new_conv_layer
        

        # model.head.classification_head.num_classes = 2
        
        # anchor_generator = model.anchor_generator
        # backbone = model.backbone

        # # 重新定义分类头和回归头
        # new_classification_head = SSDLiteClassificationHead(in_channels=672, num_anchors=anchor_generator.num_anchors_per_location()[0], num_classes=2)
        # new_regression_head = SSDLiteRegressionHead(in_channels=model.head.regression_head.in_channels, num_anchors=anchor_generator.num_anchors_per_location()[0])

        # # 替换原始模型的头
        # model.head.classification_head = new_classification_head
        # model.head.regression_head = new_regression_head
        model.to(device)
    elif model_name == 'New_ssd':
        model = get_new_ssd(2)
        model.to(device)
    else:
        model = MobileNetDetection().to(device)
    for param in model.parameters():
        print(1)
        print(param.name, param.shape)
    
    # optimizer = optim.SGD(model.parameters(), lr=0.001)
    
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    if model_name == 'fastrcnn':
        optimizer = optim.Adam(model.parameters(), lr=0.0001)
   
    localtime = time.localtime(time.time())
    log_name = '_'+ str(localtime.tm_year) + str(localtime.tm_mon) + str(localtime.tm_mday) + str(localtime.tm_hour) + str(localtime.tm_min) + str(localtime.tm_sec)
    print(log_name)
    writer = SummaryWriter(log_dir='runs/'+model_name+'_'+data_type+log_name)
    num_epoch = 400
    best_loss = float('inf')
    best_model = None
    early_stop = 0
    # 
    # x = torch.randn(10, 64, 18)  # 
    # y = torch.randint(0, 2, (10,))  # 0 or 1
    # dataset = physical_dataset_mask(data_path)
    # dataset = physical_dataset_class_FER(data_path, 3,True)
    users = [1,2,3,4,5,6,7,8,9,10,11,12,13]
    dataset = physical_dataset_class_FER(data_path, users, False,['P1','P2'], [],True)
    train_size = int(0.8 * len(dataset))
    validation_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - validation_size
    train_dataset, validation_dataset, test_dataset = random_split(dataset, [train_size, validation_size, test_size])

    train_dataset = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    validation_dataset = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    test_dataset = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    # val_loss, iou = test(model, validation_dataset,data_type, model_name)
    for epoch in range(num_epoch):  
        model.train()
        for i, (orientation,depth, tof_data, class_label,face_orientation, input_depth,input_reflectance,target,imgs,mask,uid) in enumerate(train_dataset):
            optimizer.zero_grad()
            
            # 
            if data_type == 'rd':
                input_depth = input_depth.unsqueeze(1)*0.001
                input_reflectance = input_reflectance.unsqueeze(1)
                # 将input_reflectance中小于10的置为0
                input_reflectance[input_reflectance < 5] = 0
                input_data = input_reflectance*torch.pow(input_depth, 4)
                input_data[input_data > 10] = 0
            elif data_type == 'r':
                input_reflectance = input_reflectance.unsqueeze(1)
                input_depth = input_depth.unsqueeze(1)*0.001
                #concatenate the reflectance and depth in channel 1
                input_data = torch.cat((input_reflectance, input_depth), 1)
                input_data = torch.cat((input_reflectance, input_data), 1)
            else:
                input_depth = input_depth.unsqueeze(1)*0.001
                input_data = input_depth
            # print(land3D)
            # tof_data = detect.permute(0,3,1,2)
            # print(input_data)
            if model_name == 'fastrcnn' or model_name == 'ssd' or model_name == 'New_ssd':
                input_data = list(input_d.to(device) for input_d in input_data)
                # targets = [{k: v.to(device) for k, v in t.items()} for t in target]
                # print(target)
                targets = []
                if model_name == 'New_ssd':
                    for ii in range(len(target['boxes'])):
                        targets.append({'boxes': target['boxes'][ii].to(device), 'labels': target['labels'][ii].to(device)})
                else:
                    # targets = []
                    for ii in range(len(target['boxes'])):
                        targets.append({'boxes': target['boxes'][ii].to(device), 'labels': class_label[ii].to(device)})
                # targets = [{k: v.to(device) for k, v in t.items()} for t in target]
                
                loss_dict = model(input_data, targets)
                losses = sum(loss for loss in loss_dict.values())
                optimizer.zero_grad()
                losses.backward()
                optimizer.step()
                print(f"Epoch {epoch+1}, Loss: {losses.item()}")
                writer.add_scalar('Loss/train', losses.item(), epoch * len(train_dataset) + i)
            # else:
            #     out_detect = model(input_data)
                
            #     loss = mse(out_detect, detect)
            #     iou = calculate_iou(detect, out_detect)
            #     # print(detect)
            #     # loss = l1 + l2 + l3
            #     loss.backward()

           
            #     optimizer.step()
            #     print('total_loss:', loss.item())
            #     print('iou:', iou.mean().item())
            #     # print('l1:', l1.item())
            #     # print('l2:', l2.item())
            #     # print('l3:', l3.item())


                
            #     print(f"Epoch {epoch+1}, Loss: {loss.item()}")

            #     writer.add_scalar('Loss/train', loss.item(), epoch * len(train_dataset) + i)
            #     writer.add_scalar('IoU/train', iou.mean().item(), epoch * len(train_dataset) + i)
            #     # writer.add_scalar('Loss/l1', l1.item(), epoch * len(train_dataset) + i)
            #     # writer.add_scalar('Loss/l2', l2.item(), epoch * len(train_dataset) + i)
            #     # writer.add_scalar('Loss/l3', l3.item(), epoch * len(train_dataset) + i)

        val_loss, iou, acc = test(model, validation_dataset,data_type, model_name)
        print(f"Epoch {epoch+1}, Validation Loss: {val_loss}")
        print(f"Epoch {epoch+1}, Validation IoU: {iou}")
        print(f"Epoch {epoch+1}, Validation Acc: {acc}")
        writer.add_scalar('Loss/validation', val_loss, epoch)
        writer.add_scalar('IoU/validation', iou, epoch)
        writer.add_scalar('Acc/validation', acc, epoch)

        if val_loss < best_loss:
            best_loss = val_loss
            best_model = model.state_dict()
            early_stop = 0
            # save coefficient and signal(tensor) to npy
        else:
            early_stop += 1
            if early_stop > 20:
                break
        if epoch % 10 == 0:
            save_path = 'weights/'+ log_name +model_name+'_'+data_type+'_baseline'+'.pth'
            torch.save(best_model, save_path)

    test_loss, iou_test, acc = test(model, test_dataset,data_type, model_name)
    print(f"Test Loss: {test_loss}")
    print(f"Test IoU: {iou_test}")
    print(f"Test Acc: {acc}")
    writer.add_scalar('Loss/test', test_loss, 0)
    writer.add_scalar('IoU/test', iou_test, 0)
    writer.add_scalar('Acc/test', acc, 0)

    save_path = 'weights/'+ log_name +model_name+'_'+data_type+'_baseline'+'.pth'
    torch.save(best_model, save_path)
    writer.close()
    print(log_name)

if __name__ == '__main__':
    #python To_Face_maskGenerate_Train.py --data_folder inves_dataset/dataset_FER/ --model fastrcnn --data_type rd
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_folder', type=str, default='dataset/')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--model', type=str, default='mask')
    parser.add_argument('--data_type', type=str, default='rd', help='r: reflectance, d: depth, rd: reflectance and depth')
    parser.add_argument('--cuda', type=int, default=0)
    print(parser.parse_args())
    args = parser.parse_args()
    datafolder = args.data_folder
    epochs = args.epochs
    model_name= args.model
    data_type = args.data_type
    cuda_number = args.cuda
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_number)
    train(datafolder, model_name, data_type)