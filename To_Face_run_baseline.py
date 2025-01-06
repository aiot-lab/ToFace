import torch
import torch.nn as nn
import torch.optim as optim
from ToFace_dataset import physical_dataset
from torch.utils.data import DataLoader, random_split

# use tensorboard
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import accuracy_score
import numpy as np
import time
import argparse
import re
from ToFace_1 import ToFace, Unet_baseline, Mobilenet_baseline, DualUNet



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

def test(model, test_dataset):
    model.eval()
    
    mae = nn.L1Loss()
    mse = nn.MSELoss()
    val_loss = 0
    mae_loss = 0
    L1 = 0
    L2 = 0
    L3 = 0
    l = 0
    with torch.no_grad():
        for i, (orientation,depth, tof_data) in enumerate(test_dataset):
            mask = (depth != 0).float()
            tof_data = tof_data.permute(0,3,1,2)
            out_orientation, out_depth = model(tof_data)
            # print(targets)
            out_orientation = out_orientation.reshape(-1, 16, 16)
            out_depth = out_depth.reshape(-1, 16, 16)
            out_depth = out_depth * mask
            out_orientation = out_orientation * mask
            # normalize the tof_data along the last dimention, the sum of the 18 elements is 1
            # tof_data = tof_data/(torch.sum(tof_data, dim = 2, keepdim = True)+1e-6)
            l1 = mse(out_orientation, orientation)
            l2 = mse(out_depth, depth)

            val_loss = l1 + l2 
            # print(f"Test recons Loss: {tmp_loss.item()}")
            val_loss += val_loss.item()
            L1 += l1.item()
            L2 += l2.item()
            
            l+=1
    print(f"Test Loss: {val_loss/l}")
    print(f"Test L1: {L1/l}")
    print(f"Test L2: {L2/l}")
    print(f"Test L3: {L3/l}")

    return val_loss/l

def relu_loss(x):
    return torch.relu(-x).sum()  # 惩罚所有小于0的元素

def train(data_path, modelname = 'unet'):
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
    if modelname == 'unet':
        model = Unet_baseline().to(device)
    # elif modelname == 'dualunet':
    #     model = DualUNet().to(device)
    else:
        model = Mobilenet_baseline().to(device)
    for param in model.parameters():
        print(1)
        print(param.name, param.shape)
    
    # optimizer = optim.SGD(model.parameters(), lr=0.001)
    
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    writer = SummaryWriter()
    localtime = time.localtime(time.time())
    log_name = '_'+ str(localtime.tm_year) + str(localtime.tm_mon) + str(localtime.tm_mday) + str(localtime.tm_hour) + str(localtime.tm_min) + str(localtime.tm_sec)

    num_epoch = 400
    best_loss = float('inf')
    best_model = None
    # 
    # x = torch.randn(10, 64, 18)  # 
    # y = torch.randint(0, 2, (10,))  # 0 or 1
    dataset = physical_dataset(data_path)
    train_size = int(0.8 * len(dataset))
    validation_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - validation_size
    train_dataset, validation_dataset, test_dataset = random_split(dataset, [train_size, validation_size, test_size])

    train_dataset = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    validation_dataset = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    test_dataset = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

    for epoch in range(num_epoch):  
        model.train()
        for i, (orientation,depth, tof_data) in enumerate(train_dataset):
            optimizer.zero_grad()
            mask = (depth != 0).float()
            # print(land3D)
            tof_data = tof_data.permute(0,3,1,2)
            out_orientation, out_depth = model(tof_data)
            out_orientation = out_orientation.reshape(-1, 16, 16)
            out_depth = out_depth.reshape(-1, 16, 16)
            out_depth = out_depth * mask
            out_orientation = out_orientation * mask
            # normalize the tof_data along the last dimention, the sum of the 18 elements is 1
            # tof_data = tof_data/(torch.sum(tof_data, dim = 2, keepdim = True)+1e-6)
            l1 = mse(out_orientation, orientation)
            l2 = mse(out_depth, depth)
            loss = l1 + l2 
            loss.backward()

           
            optimizer.step()
            print('total_loss:', loss.item())
            print('l1:', l1.item())
            print('l2:', l2.item())


            
            print(f"Epoch {epoch+1}, Loss: {loss.item()}")

            writer.add_scalar('Loss/train', loss.item(), epoch * len(train_dataset) + i)
            writer.add_scalar('Loss/l1', l1.item(), epoch * len(train_dataset) + i)
            writer.add_scalar('Loss/l2', l2.item(), epoch * len(train_dataset) + i)

        val_loss = test(model, validation_dataset)
        writer.add_scalar('Loss/validation', val_loss, epoch)

        if val_loss < best_loss:
            best_loss = val_loss
            best_model = model.state_dict()
            # save coefficient and signal(tensor) to npy


    test_loss = test(model, test_dataset)
    writer.add_scalar('Loss/test', test_loss, 0)

    save_path = 'weights/'+ log_name +model_name+'_-4.pth'
    torch.save(best_model, save_path)
    writer.close()
    print(log_name)



if __name__ == '__main__':
    #python To_Face_run_baseline.py --data_folder inves_dataset/dataset_facial1/ --model unet
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