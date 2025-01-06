import numpy as np
import torch
from torch.utils.data import Dataset
import os
import pickle
import math
import pickle
from tqdm import tqdm
from scipy.integrate import quad
from numba import jit
import cv2


# def find_zone(depth, h, w, H, W):
#     x = (W//2-w)/W*depth[h][w]*2*math.tan(80/2/180*math.pi)-33
#     y = (H//2-h)/H*depth[h][w]*2*math.tan(51/2/180*math.pi)-70
#     z= depth[h][w]-10
#     w = z*2*math.tan(45/2/180*math.pi)
#     h = z*2*math.tan(45/2/180*math.pi)
#     w_zone = w/8
#     x_zone = 4-(x/w_zone)
#     y_zone = 4-(y/w_zone)
    
#     return int(x_zone), int(y_zone),x,y
#     # there are 8*8 zones,FOV with 45*45, find the zone of the point of xyz
def find_zone(depth, h, w, H, W):
    cx, cy = W / 2, H / 2
    fx = W / (2 * np.tan(80 / 2 * np.pi / 180))
    fy = H / (2 * np.tan(51 / 2 * np.pi / 180))

    # 从像素坐标到世界坐标
    z = depth[h, w]
    x = (w - cx) * z / fx + 33  # X坐标调整，包含摄像头的水平位移
    y = (h - cy) * z / fy + 70  # Y坐标调整，包含摄像头的垂直位移
    # x = (w - cx) * z / fx + 20  # X坐标调整，包含摄像头的水平位移
    # y = (h - cy) * z / fy +0  # Y坐标调整，包含摄像头的垂直位移

    # 第二摄像头参数
    w_fov = 45
    w_zone_size = z * 2 * np.tan(w_fov / 2 * np.pi / 180) / 8

    # 计算落入的像素区域
    x_zone = int(4 + (x / w_zone_size))
    y_zone = int(4 + (y / w_zone_size))

    return x_zone, y_zone, x, y


def physical_dataprocess(data_path):
    dirs = os.listdir(data_path)
    for ii, dir in enumerate(tqdm(dirs)):
        for log in os.listdir(data_path + dir):
            depth_path = data_path + dir + '/' + log+'/depth/'
            img_path = data_path + dir + '/' + log+'/img/'
            physical_data_path = data_path + dir + '/' + log + '/physical_data/'
            if not os.path.exists(physical_data_path):
                os.makedirs(physical_data_path)
            files = os.listdir(depth_path)
            landmarks = pickle.load(open(data_path + dir + '/' + log + '/landmarks.pkl', 'rb'))

            for landmark in landmarks:
                if landmark[1] == []:
                    continue
                i = landmark[0][:-4]
                i = i[11:]
                print(i)
                with open(depth_path + 'depth_data'+i+'.npy', 'rb') as f:
                    depth = np.load(f)
                # print(depth.shape)
                #resize the depth image to 360*640
                # depth = cv2.resize(depth, (640, 360))
                depth = cv2.resize(depth, (128, 72))
                # print(depth.shape)
                # with open(img_path + 'color_image'+str(i+1)+'.npy', 'rb') as f:
                #     img = pickle.load(f)
                W = len(depth[0])
                H = len(depth)
                # print(landmark[1][0].shape)
                # h_min = int(min(landmark[1][0][1])//3)
                # h_max = int(max(landmark[1][0][1])//3)
                # w_min = int(min(landmark[1][0][0])//3)
                # w_max = int(max(landmark[1][0][0])//3)
                h_min = int(min(landmark[1][0][1])//15)
                h_max = int(max(landmark[1][0][1])//15)
                w_min = int(min(landmark[1][0][0])//15)
                w_max = int(max(landmark[1][0][0])//15)
                D3_map = np.zeros((H,W,8))
                physical_data = [[[] for _ in range(8)] for _ in range(8)]
                # print(len(physical_data))
                # print(len(physical_data[0]))
                for h in range(h_min, h_max+1):
                    for w in range(w_min, w_max+1):
                        if depth[h][w] > 262.5  and depth[h][w] < 937.5:
                            
                            zone = find_zone(depth, h, w, H, W)
                            D3_map[h][w] = np.array([zone[2],zone[3],depth[h][w], zone[0], zone[1],0,0,0])
                for h in range(h_min, h_max+1):
                    for w in range(w_min, w_max+1):
                        x = D3_map[h][w][0]
                        y = D3_map[h][w][1]
                        z = D3_map[h][w][2]
                        if int(D3_map[h][w][3])<0 or int(D3_map[h][w][3])>7 or int(D3_map[h][w][4])<0 or int(D3_map[h][w][4])>7 or D3_map[h][w][2] < 1:    
                            continue
                        #based on the the xyz of the D3_map[h-1][w], D3_map[h+1][w], D3_map[h][w-1], D3_map[h][w+1], calculate the orientation of the pointof D3_map[h][w]
                        if h>0 and h<H-1 and w>0 and w<W-1:
                            x1 = D3_map[h-1][w][0]
                            y1 = D3_map[h-1][w][1]
                            z1 = D3_map[h-1][w][2]
                            x2 = D3_map[h+1][w][0]
                            y2 = D3_map[h+1][w][1]
                            z2 = D3_map[h+1][w][2]
                            x3 = D3_map[h][w-1][0]
                            y3 = D3_map[h][w-1][1]
                            z3 = D3_map[h][w-1][2]
                            x4 = D3_map[h][w+1][0]
                            y4 = D3_map[h][w+1][1]
                            z4 = D3_map[h][w+1][2]
                            a = np.array([x1-x, y1-y, z1-z])
                            b = np.array([x2-x, y2-y, z2-z])
                            c = np.array([x3-x, y3-y, z3-z])
                            d = np.array([x4-x, y4-y, z4-z])
                            n1 = np.cross(a,b)
                            n2 = np.cross(b,c)
                            n3 = np.cross(c,d)
                            n4 = np.cross(d,a)
                            n = (n1+n2+n3+n4)/4
                            D3_map[h][w][5] = n[0]
                            D3_map[h][w][6] = n[1]
                            D3_map[h][w][7] = n[2]
                            # print(int(D3_map[h][w][3]))
                            # print(int(D3_map[h][w][4]))
                            physical_data[2][2]
                            physical_data[int(D3_map[h][w][4])][int(D3_map[h][w][3])].append([D3_map[h][w][:3], D3_map[h][w][5:8]])
                #save the physical_data
                with open(data_path + dir + '/' + log + '/physical_data/'+'physical'+str(i)+'.pkl', 'wb') as f:
                    pickle.dump(physical_data, f)

def calculate_integral(data_path):
    dirs = os.listdir(data_path)
    error_files = []
    for ii, dir in enumerate(tqdm(dirs)):
        for log in os.listdir(data_path + dir):
            
            physical_data_path = data_path + dir + '/' + log + '/physical_data/'
            
            files = os.listdir(physical_data_path)
            c = 0
            for file in files:
                print(c)
                c += 1
                try:
                    with open(physical_data_path + file, 'rb') as f:
                        physical_data = pickle.load(f)
                
                except:
                    print(physical_data_path + file)
                    error_files.append(physical_data_path + file)

                    continue
                # with open(img_path + 'color_image'+str(i+1)+'.npy', 'rb') as f:
                #     img = pickle.load(f)
                for h in range(8):
                    for w in range(8):
                        for i in range(len(physical_data[h][w])):
                            # length = torch.norm(physical_data[h][w][i][0]) use np
                            length = np.linalg.norm(physical_data[h][w][i][0])/10
                            d = physical_data[h][w][i][0][2]/10
                            integral = compute_integral(length, d)
                            physical_data[h][w][i].append(np.array([integral, 0,0]))
                
                with open(physical_data_path + file, 'wb') as f:
                    pickle.dump(physical_data, f)
    with open('error_files.pkl', 'wb') as f:
        pickle.dump(error_files, f)
@jit(nopython=True)
def integrand( r, length):
        return (length*length+r*r)**(-2.5)

def compute_integral( length, d):
    # 将torch张量转换为标量
    length_scalar = length.item()
    # 计算积分上限
    upper_limit = (d * math.tan(math.radians(51.0 / 2))) / 180
    # print(upper_limit)
    # 使用scipy的quad函数进行数值积分
    integral, _ = quad(integrand, 0, upper_limit, args=(length_scalar,))
    return integral

                
def physical_dataprocess_board(data_path):             
    dirs = os.listdir(data_path)
    for i, dir in enumerate(tqdm(dirs)):
        for log in os.listdir(data_path + dir):
            depth_path = data_path + dir + '/' + log+'/depth/'
            img_path = data_path + dir + '/' + log+'/img/'
            physical_data_path = data_path + dir + '/' + log + '/physical_data/'
            if not os.path.exists(physical_data_path):
                os.makedirs(physical_data_path)
            files = os.listdir(depth_path)
            # landmarks = pickle.load(open(data_path + dir + '/' + log + '/landmarks.pkl', 'rb'))

            for ii,file in enumerate(files):
                
                print(ii)
                with open(depth_path + 'depth_data'+str(ii+1)+'.npy', 'rb') as f:
                    depth = np.load(f)
                # with open(img_path + 'color_image'+str(i+1)+'.npy', 'rb') as f:
                #     img = pickle.load(f)
                depth = cv2.resize(depth, (640, 360))
                W = len(depth[0])
                H = len(depth)
                # print(landmark[1][0].shape)
                
                D3_map = np.zeros((H,W,8))
                physical_data = [[[] for _ in range(8)] for _ in range(8)]
                # print(len(physical_data))
                # print(len(physical_data[0]))
                for h in range(H):
                    for w in range(W):
                        if depth[h][w] > 240  and depth[h][w] < 700:
                            
                            zone = find_zone(depth, h, w, H, W)
                            D3_map[h][w] = np.array([zone[2],zone[3],depth[h][w], zone[0], zone[1],0,0,0])
                for h in range(H):
                    for w in range(W):
                        x = D3_map[h][w][0]
                        y = D3_map[h][w][1]
                        z = D3_map[h][w][2]
                        if int(D3_map[h][w][3])<0 or int(D3_map[h][w][3])>7 or int(D3_map[h][w][4])<0 or int(D3_map[h][w][4])>7 or D3_map[h][w][2] <1:    
                            continue
                        #based on the the xyz of the D3_map[h-1][w], D3_map[h+1][w], D3_map[h][w-1], D3_map[h][w+1], calculate the orientation of the pointof D3_map[h][w]
                        if h>0 and h<H-1 and w>0 and w<W-1:
                            x1 = D3_map[h-1][w][0]
                            y1 = D3_map[h-1][w][1]
                            z1 = D3_map[h-1][w][2]
                            x2 = D3_map[h+1][w][0]
                            y2 = D3_map[h+1][w][1]
                            z2 = D3_map[h+1][w][2]
                            x3 = D3_map[h][w-1][0]
                            y3 = D3_map[h][w-1][1]
                            z3 = D3_map[h][w-1][2]
                            x4 = D3_map[h][w+1][0]
                            y4 = D3_map[h][w+1][1]
                            z4 = D3_map[h][w+1][2]
                            a = np.array([x1-x, y1-y, z1-z])
                            b = np.array([x2-x, y2-y, z2-z])
                            c = np.array([x3-x, y3-y, z3-z])
                            d = np.array([x4-x, y4-y, z4-z])
                            n1 = np.cross(a,b)
                            n2 = np.cross(b,c)
                            n3 = np.cross(c,d)
                            n4 = np.cross(d,a)
                            n = (n1+n2+n3+n4)/4
                            D3_map[h][w][5] = n[0]
                            D3_map[h][w][6] = n[1]
                            D3_map[h][w][7] = n[2]
                            # print(int(D3_map[h][w][3]))
                            # print(int(D3_map[h][w][4]))
                            physical_data[2][2]
                            physical_data[int(D3_map[h][w][4])][int(D3_map[h][w][3])].append([D3_map[h][w][:3], D3_map[h][w][5:8]])
                #save the physical_data
                with open(data_path + dir + '/' + log + '/physical_data/'+'physical'+str(i)+'.pkl', 'wb') as f:
                    pickle.dump(physical_data, f)


                        






class physical_dataset(Dataset):
    def __init__(self, data_path):
        self.data_path = data_path
        self.data = []
        self.tof_data = []
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  
        ind = 0
        self.indexs = []
        for dir in os.listdir(data_path):
            for log in os.listdir(data_path + dir):
                physical_data_path = data_path + dir + '/' + log + '/physical_data/'

                files = os.listdir(physical_data_path)
                tof_data =  data_path + dir + '/' + log + '/tof_data.pkl'
                with open(tof_data, 'rb') as f:
                    tof_data = pickle.load(f)
                self.tof_data.append(tof_data)
                for file in files:
                    # with open(physical_data_path + file, 'rb') as f:
                    #     physical_data = pickle.load(f)
                    # self.data.append(physical_data)
                    file_path = physical_data_path + file
                    self.data.append([ind, file_path])
                ind += 1
        # random select 30 samples from the dataset, save the index in a list, the data is a list, so only choose the index
        for i in range(30):
            index = np.random.randint(0, len(self.data))
            while index in self.indexs:
                index = np.random.randint(0, len(self.data))
            self.indexs.append(index)
    def __len__(self):
        return len(self.data)
        # return 30
    def __getitem__(self, idx):
        # get the random index between 0 and len(self.data)
        # idx = np.random.randint(0, len(self.data))
        # idx = self.indexs[idx]
        file_path = self.data[idx][1]
        with open(file_path, 'rb') as f:
            physical_data = pickle.load(f)
        tof_frame_index = file_path.split('/')[-1]
        tof_frame_index = tof_frame_index.split('.')[0]
        tof_frame_index = int(tof_frame_index[8:])-1
        tof_data = self.tof_data[self.data[idx][0]][tof_frame_index]
        # print(tof_data[1].shape)
        tmp = np.flip(tof_data[1],0)
        # tmp = np.flip(tmp,1)
        tof_data = torch.tensor(tmp.copy())
        tof_data = tof_data.to(self.device)
        tof_data = tof_data.float()
        return physical_data, tof_data

        
if __name__ == '__main__':
    data_path = '/home/shared/SPAD_TOF/Data_preprocessing/physycal_model/inves_dataset/dataset_facial2/'
    physical_dataprocess(data_path)
    print('done')

    # data_path = '/home/shared/SPAD_TOF/Data_preprocessing/physycal_model/inves_dataset/dataset_physical_facial1/'
    # calculate_integral(data_path)
    # print('done')

    # data_path = '/home/shared/SPAD_TOF/Data_preprocessing/physycal_model/inves_dataset/dataset_physical/'
    # physical_dataprocess_board(data_path)
    # print('done')
    # calculate_integral(data_path)
    # print('done')
    
    # data_path = '/home/shared/SPAD_TOF/Data_preprocessing/physycal_model/dataset/'
    # dataset = physical_dataset(data_path)
    # print(len(dataset))
    # for i in range(len(dataset)):
    #     print(dataset[i][0])
    # # print(dataset[0])
