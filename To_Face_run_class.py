import torch
import torch.nn as nn
import torch.optim as optim
from ToFace_dataset import physical_dataset, physical_dataset_class, physical_dataset_class_FER
from torch.utils.data import DataLoader, random_split

# use tensorboard
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import accuracy_score
import numpy as np
import time
import argparse
import re
from ToFace_1 import ToFace, Unet_baseline, Mobilenet_baseline, DualUNet, ClassificationNetwork
import pickle
# from torchmetrics import Accuracy



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

def test(class_model, model, test_dataset):
    model.eval()
    class_model.eval()
    entropy_loss = nn.CrossEntropyLoss()
    val_loss = 0
    total_pred = []
    total_true = []
    l = 0
    with torch.no_grad():
        for i, (orientation,depth, tof_data, class_label) in enumerate(test_dataset):
            mask = (depth != 0).float()
            tof_data = tof_data.permute(0,3,1,2)
            out_orientation, out_depth = model(tof_data)
            # print(targets)
            out_orientation = out_orientation.reshape(-1, 16, 16)
            out_depth = out_depth.reshape(-1, 16, 16)
            out_depth = out_depth * mask
            out_orientation = out_orientation * mask
            out_depth = out_depth.view(-1,4, 8,8)
            out_orientation = out_orientation.view(-1,4, 8,8)
            class_output = class_model(out_orientation, out_depth)
            # normalize the tof_data along the last dimention, the sum of the 18 elements is 1
            # tof_data = tof_data/(torch.sum(tof_data, dim = 2, keepdim = True)+1e-6)
            preds = torch.argmax(class_output, dim=1)
            true_labels = torch.argmax(class_label, dim=1)
            total_pred.append(preds)
            total_true.append(true_labels)
            total_correct += (preds == true_labels).sum().item()
            total_samples += true_labels.size(0)
            val_loss = entropy_loss(class_output, class_label)
            # print(f"Test recons Loss: {tmp_loss.item()}")
            val_loss += val_loss.item()
            
            
            l+=1
    print(f"Test Loss: {val_loss/l}")
    print(f"Test Accuracy: {total_correct/total_samples}")

    return val_loss/l, total_correct/total_samples, total_pred, total_true

def relu_loss(x):
    return torch.relu(-x).sum()  # 惩罚所有小于0的元素

def train(data_path, modelname = 'unet'):
    batch_size = 16
    # set the seeds
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    np.random.seed(42)
    entropy_loss = nn.CrossEntropyLoss()
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    

    mse = nn.MSELoss()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if modelname == 'unet':
        model = Unet_baseline().to(device)
    elif modelname == 'dualunet':
        model = DualUNet().to(device)
    else:
        model = Mobilenet_baseline().to(device)
    model.load_state_dict(torch.load('weights/_2024521171048unet_-4.pth'))
    class_model = ClassificationNetwork(7).to(device)
    for param in model.parameters():
        param.requires_grad = False
    
    # optimizer = optim.SGD(model.parameters(), lr=0.001)
    
    optimizer = optim.Adam(class_model.parameters(), lr=0.001)
    writer = SummaryWriter()
    localtime = time.localtime(time.time())
    log_name = '_'+ str(localtime.tm_year) + str(localtime.tm_mon) + str(localtime.tm_mday) + str(localtime.tm_hour) + str(localtime.tm_min) + str(localtime.tm_sec)

    num_epoch = 400
    best_loss = float('inf')
    best_model = None
    # 
    # x = torch.randn(10, 64, 18)  # 
    # y = torch.randint(0, 2, (10,))  # 0 or 1
    # dataset = physical_dataset_class(data_path, 1)# 1 for small move, 0 for stay
    dataset = physical_dataset_class_FER(data_path, 3,True)
    # dataset = physical_dataset_class(data_path, 0)
    train_size = int(0.8 * len(dataset))
    validation_size = len(dataset) - train_size
    # test_size = len(dataset) - train_size - validation_size
    train_dataset, validation_dataset = random_split(dataset, [train_size, validation_size])
    test_dataset = physical_dataset_class_FER(data_path, 3, False)

    train_dataset = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    validation_dataset = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    test_dataset = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    early_stop = 40
    num = 0
    for epoch in range(num_epoch):  
        model.eval()
        class_model.train()
        for i, (orientation,depth, tof_data, label) in enumerate(train_dataset):
            optimizer.zero_grad()
            mask = (depth != 0).float()
            tof_data = tof_data.permute(0,3,1,2)
            with torch.no_grad():
                out_orientation, out_depth = model(tof_data)
            out_orientation = out_orientation.reshape(-1, 16, 16)
            out_depth = out_depth.reshape(-1, 16, 16)
            out_depth = out_depth * mask
            out_orientation = out_orientation * mask
            # print(out_orientation.shape)
            # print(out_depth.shape)
            out_depth = out_depth.view(-1,4, 8,8)
            out_orientation = out_orientation.view(-1,4, 8,8)
            claass_output = class_model(out_orientation, out_depth)
            # normalize the tof_data along the last dimention, the sum of the 18 elements is 1
            # tof_data = tof_data/(torch.sum(tof_data, dim = 2, keepdim = True)+1e-6)
            # print(label)
            # print(claass_output)
            loss = entropy_loss(claass_output, label)
            
            loss.backward()
            optimizer.step()
            print('total_loss:', loss.item())
            print(f"Epoch {epoch+1}, Loss: {loss.item()}")
            writer.add_scalar('Loss/train', loss.item(), epoch * len(train_dataset) + i)
            

        val_loss, acc,_,_ = test(class_model, model, validation_dataset)
        writer.add_scalar('Loss/validation', val_loss, epoch)
        writer.add_scalar('Accuracy/validation', acc, epoch)
        num += 1
        if val_loss < best_loss:
            num = 0
            best_loss = val_loss
            best_model = class_model.state_dict()
            # save coefficient and signal(tensor) to npy
        if num > early_stop:
            break


    test_loss, acc,total_pred, total_true = test(class_model,model, test_dataset)
    with open('pred_true.pkl', 'wb') as f:
        pickle.dump([total_pred, total_true], f)
    writer.add_scalar('Loss/test', test_loss, 0)
    writer.add_scalar('Accuracy/test', acc, 0)
    save_path = 'weights/'+ log_name +model_name+'class_-4.pth'
    torch.save(best_model, save_path)
    writer.close()
    print(log_name)



if __name__ == '__main__':
    #python To_Face_run_class.py --data_folder inves_dataset/dataset_FER/ --model unet
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_folder', type=str, default='dataset/')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--model', type=str, default='unet')
    print(parser.parse_args())
    args = parser.parse_args()
    datafolder = args.data_folder
    epochs = args.epochs
    model_name= args.model
    train(datafolder, model_name)