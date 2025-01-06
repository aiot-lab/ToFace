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
import re
import time


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
    w_zone_size = z * 2 * np.tan(w_fov / 2 * np.pi / 180) / 64

    # 计算落入的像素区域
    x_zone = int(32 + (x / w_zone_size))
    y_zone = int(32 + (y / w_zone_size))

    return x_zone, y_zone, x, y

def cal_orientation(eyebow1, eyebow2, mouse1, mouse2, depth_map):
    #depth: 1920*1080
    eyebow1_z = depth_map[int(eyebow1[0])][int(eyebow1[1])]
    eyebow2_z = depth_map[int(eyebow2[0])][int(eyebow2[1])]
    mouse1_z = depth_map[int(mouse1[0])][int(mouse1[1])]
    mouse2_z = depth_map[int(mouse2[0])][int(mouse2[1])]

    eyebow1_x = (960-int(eyebow1[1]))/960*eyebow1_z*2*math.tan(80/2/180*math.pi)
    eyebow1_y = (540-int(eyebow1[0]))/540*eyebow1_z*2*math.tan(51/2/180*math.pi)
    eyebow2_x = (960-int(eyebow2[1]))/960*eyebow2_z*2*math.tan(80/2/180*math.pi)
    eyebow2_y = (540-int(eyebow2[0]))/540*eyebow2_z*2*math.tan(51/2/180*math.pi)
    mouse1_x = (960-int(mouse1[1]))/960*mouse1_z*2*math.tan(80/2/180*math.pi)
    mouse1_y = (540-int(mouse1[0]))/540*mouse1_z*2*math.tan(51/2/180*math.pi)  
    mouse2_x = (960-int(mouse2[1]))/960*mouse2_z*2*math.tan(80/2/180*math.pi)
    mouse2_y = (540-int(mouse2[0]))/540*mouse2_z*2*math.tan(51/2/180*math.pi)
    eyebow1 = np.array([eyebow1_x, eyebow1_y, eyebow1_z])
    eyebow2 = np.array([eyebow2_x, eyebow2_y, eyebow2_z])
    mouse1 = np.array([mouse1_x, mouse1_y, mouse1_z])
    mouse2 = np.array([mouse2_x, mouse2_y, mouse2_z])

    mouse_mid = (mouse1+mouse2)/2

    v1 = eyebow1 - mouse_mid
    v2 = eyebow2 - mouse_mid
    n = np.cross(v1, v2)
    n = n/(np.linalg.norm(n)+1e-6)
    return n


# def physical_dataprocess(data_path):
#     dirs = os.listdir(data_path)
#     for ii, dir in enumerate(tqdm(dirs)):
#         for log in os.listdir(data_path + dir):
#             depth_path = data_path + dir + '/' + log+'/depth/'
#             img_path = data_path + dir + '/' + log+'/img/'
#             physical_data_path = data_path + dir + '/' + log + '/physical_data/'
#             if not os.path.exists(physical_data_path):
#                 os.makedirs(physical_data_path)
#             # else:
#             #     continue
#             files = os.listdir(depth_path)
#             landmarks = pickle.load(open(data_path + dir + '/' + log + '/landmarks.pkl', 'rb'))

#             for landmark in landmarks[7:]:
#                 if landmark[1] == []:
#                     continue
#                 i = landmark[0][:-4]
#                 i = i[11:]
#                 i = int(i)-6
#                 # print(i)
#                 with open(depth_path + 'depth_data'+i+'.npy', 'rb') as f:
#                     depth = np.load(f)
#                 # print(depth.shape)
#                 eyebow1 = [landmark[1][0][1][19], landmark[1][0][0][19]]
#                 eyebow2 = [landmark[1][0][1][24], landmark[1][0][0][24]]
#                 mouse1 = [landmark[1][0][1][48], landmark[1][0][0][48]]
#                 mouse2 = [landmark[1][0][1][54], landmark[1][0][0][54]]
#                 orientation = cal_orientation(eyebow1, eyebow2, mouse1, mouse2, depth)

#                 #resize the depth image to 360*640
#                 # depth = cv2.resize(depth, (640, 360))
#                 origin_H = len(depth)
#                 origin_W = len(depth[0])
#                 depth = cv2.resize(depth, (192, 108))
#                 # print(depth.shape)
#                 # with open(img_path + 'color_image'+str(i+1)+'.npy', 'rb') as f:
#                 #     img = pickle.load(f)
#                 W = len(depth[0])
#                 H = len(depth)
#                 scale = origin_H/H
#                 # print(landmark[1][0].shape)
#                 # h_min = int(min(landmark[1][0][1])//3)
#                 # h_max = int(max(landmark[1][0][1])//3)
#                 # w_min = int(min(landmark[1][0][0])//3)
#                 # w_max = int(max(landmark[1][0][0])//3)
#                 h_min = int(min(landmark[1][0][1])//scale)
#                 # print(h_min)
#                 h_max = int(max(landmark[1][0][1])//scale)
#                 w_min = int(min(landmark[1][0][0])//scale)
#                 w_max = int(max(landmark[1][0][0])//scale)
                

#                 D3_map = np.zeros((H,W,8))
#                 physical_data = [[[] for _ in range(16)] for _ in range(16)]
#                 # print(len(physical_data))
#                 # print(len(physical_data[0]))
#                 for h in range(h_min, h_max+1):
#                     for w in range(w_min, w_max+1):
#                         if depth[h][w] > 100.5  and depth[h][w] < 1300.5:
                            
#                             zone = find_zone(depth, h, w, H, W)
#                             D3_map[h][w] = np.array([zone[2],zone[3],depth[h][w], zone[0], zone[1],0,0,0])
#                 for h in range(h_min, h_max+1):
#                     for w in range(w_min, w_max+1):
#                         x = D3_map[h][w][0]
#                         y = D3_map[h][w][1]
#                         z = D3_map[h][w][2]
#                         if int(D3_map[h][w][3])<0 or int(D3_map[h][w][3])>15 or int(D3_map[h][w][4])<0 or int(D3_map[h][w][4])>15 or D3_map[h][w][2] < 1:    
#                             continue
#                         #based on the the xyz of the D3_map[h-1][w], D3_map[h+1][w], D3_map[h][w-1], D3_map[h][w+1], calculate the orientation of the pointof D3_map[h][w]
#                         if h>0 and h<H-1 and w>0 and w<W-1:
#                             x1 = D3_map[h-1][w][0]
#                             y1 = D3_map[h-1][w][1]
#                             z1 = D3_map[h-1][w][2]
#                             x2 = D3_map[h+1][w][0]
#                             y2 = D3_map[h+1][w][1]
#                             z2 = D3_map[h+1][w][2]
#                             x3 = D3_map[h][w-1][0]
#                             y3 = D3_map[h][w-1][1]
#                             z3 = D3_map[h][w-1][2]
#                             x4 = D3_map[h][w+1][0]
#                             y4 = D3_map[h][w+1][1]
#                             z4 = D3_map[h][w+1][2]
#                             a = np.array([x1-x, y1-y, z1-z])
#                             b = np.array([x2-x, y2-y, z2-z])
#                             c = np.array([x3-x, y3-y, z3-z])
#                             d = np.array([x4-x, y4-y, z4-z])
#                             n1 = np.cross(a,b)
#                             n2 = np.cross(b,c)
#                             n3 = np.cross(c,d)
#                             n4 = np.cross(d,a)
#                             n = (n1+n2+n3+n4)/4
#                             D3_map[h][w][5] = n[0]
#                             D3_map[h][w][6] = n[1]
#                             D3_map[h][w][7] = n[2]
#                             # print(int(D3_map[h][w][3]))
#                             # print(int(D3_map[h][w][4]))
#                             physical_data[2][2]
#                             physical_data[int(D3_map[h][w][4])][int(D3_map[h][w][3])].append([D3_map[h][w][:3], D3_map[h][w][5:8]])
#                 #save the physical_data
#                 with open(data_path + dir + '/' + log + '/physical_data/'+'physical'+str(i)+'.pkl', 'wb') as f:
#                     pickle.dump(physical_data, f)

# def physical_dataprocess(data_path):
#     dirs = os.listdir(data_path)
#     for ii, dir in enumerate(tqdm(dirs)):
#         for log in os.listdir(data_path + dir):
#             depth_path = data_path + dir + '/' + log+'/depth/'
#             img_path = data_path + dir + '/' + log+'/img/'
#             physical_data_path = data_path + dir + '/' + log + '/physical_data/'
#             if os.path.exists(physical_data_path):
#                 #delete the physical_data folder
#                 os.system('rm -r '+physical_data_path)
#             if not os.path.exists(physical_data_path):
#                 os.makedirs(physical_data_path)
#             # else:
#             #     continue
#             files = os.listdir(depth_path)
#             landmarks = pickle.load(open(data_path + dir + '/' + log + '/landmarks.pkl', 'rb'))

#             for landmark in landmarks:
#                 if landmark[1] == []:
#                     continue
#                 i = landmark[0][:-4]
#                 i = i[11:]
#                 i = int(i)-6
#                 if i <1:
#                     continue
#                 # print(i)
#                 with open(depth_path + 'depth_data'+str(i)+'.npy', 'rb') as f:
#                     depth = np.load(f)
#                 # print(depth.shape)
#                 eyebow1 = [landmark[1][0][1][19], landmark[1][0][0][19]]
#                 eyebow2 = [landmark[1][0][1][24], landmark[1][0][0][24]]
#                 mouse1 = [landmark[1][0][1][48], landmark[1][0][0][48]]
#                 mouse2 = [landmark[1][0][1][54], landmark[1][0][0][54]]
#                 orientation = cal_orientation(eyebow1, eyebow2, mouse1, mouse2, depth)

#                 #resize the depth image to 360*640
#                 # depth = cv2.resize(depth, (640, 360))
#                 origin_H = len(depth)
#                 origin_W = len(depth[0])
#                 depth = cv2.resize(depth, (192, 108))
#                 # print(depth.shape)
#                 # with open(img_path + 'color_image'+str(i+1)+'.npy', 'rb') as f:
#                 #     img = pickle.load(f)
#                 W = len(depth[0])
#                 H = len(depth)
#                 scale = origin_H/H
#                 # print(landmark[1][0].shape)
#                 # h_min = int(min(landmark[1][0][1])//3)
#                 # h_max = int(max(landmark[1][0][1])//3)
#                 # w_min = int(min(landmark[1][0][0])//3)
#                 # w_max = int(max(landmark[1][0][0])//3)
#                 h_min = int(min(landmark[1][0][1])//scale)
#                 # print(h_min)
#                 h_max = int(max(landmark[1][0][1])//scale)
#                 w_min = int(min(landmark[1][0][0])//scale)
#                 w_max = int(max(landmark[1][0][0])//scale)
                

#                 D3_map = np.zeros((H,W,8))
#                 physical_data = [[[] for _ in range(64)] for _ in range(64)]
#                 saved_data = []
#                 # print(len(physical_data))
#                 # print(len(physical_data[0]))
#                 for h in range(h_min, h_max+1):
#                     for w in range(w_min, w_max+1):
#                         if depth[h][w] > 100.5  and depth[h][w] < 1300.5:
                            
#                             zone = find_zone(depth, h, w, H, W)
#                             D3_map[h][w] = np.array([zone[2],zone[3],depth[h][w], zone[0], zone[1],0,0,0])
#                 for h in range(h_min, h_max+1):
#                     for w in range(w_min, w_max+1):
#                         x = D3_map[h][w][0]
#                         y = D3_map[h][w][1]
#                         z = D3_map[h][w][2]
#                         if int(D3_map[h][w][3])<0 or int(D3_map[h][w][3])>63 or int(D3_map[h][w][4])<0 or int(D3_map[h][w][4])>63 or D3_map[h][w][2] < 1:    
#                             continue
#                         #based on the the xyz of the D3_map[h-1][w], D3_map[h+1][w], D3_map[h][w-1], D3_map[h][w+1], calculate the orientation of the pointof D3_map[h][w]
#                         if h>0 and h<H-1 and w>0 and w<W-1:
#                             x1 = D3_map[h-1][w][0]
#                             y1 = D3_map[h-1][w][1]
#                             z1 = D3_map[h-1][w][2]
#                             x2 = D3_map[h+1][w][0]
#                             y2 = D3_map[h+1][w][1]
#                             z2 = D3_map[h+1][w][2]
#                             x3 = D3_map[h][w-1][0]
#                             y3 = D3_map[h][w-1][1]
#                             z3 = D3_map[h][w-1][2]
#                             x4 = D3_map[h][w+1][0]
#                             y4 = D3_map[h][w+1][1]
#                             z4 = D3_map[h][w+1][2]
#                             a = np.array([x1-x, y1-y, z1-z])
#                             b = np.array([x2-x, y2-y, z2-z])
#                             c = np.array([x3-x, y3-y, z3-z])
#                             d = np.array([x4-x, y4-y, z4-z])
#                             n1 = np.cross(a,b)
#                             n2 = np.cross(b,c)
#                             n3 = np.cross(c,d)
#                             n4 = np.cross(d,a)
#                             n = (n1+n2+n3+n4)/4
#                             D3_map[h][w][5] = n[0]
#                             D3_map[h][w][6] = n[1]
#                             D3_map[h][w][7] = n[2]
#                             # print(int(D3_map[h][w][3]))
#                             # print(int(D3_map[h][w][4]))
#                             physical_data[2][2]
#                             physical_data[int(D3_map[h][w][4])][int(D3_map[h][w][3])].append([D3_map[h][w][:3], D3_map[h][w][5:8]])
#                             # print(1)
#                 if len(physical_data[0]) == 0:
#                     print(i)
#                     continue          
#                 #save the physical_data
                
#                 with open(data_path + dir + '/' + log + '/physical_data/'+'physical'+str(i)+'.pkl', 'wb') as f:
#                     saved_data = [physical_data, orientation]
                    
#                     pickle.dump(saved_data, f)


def physical_dataprocess(data_path):
    dirs = os.listdir(data_path)
    for ii, dir in enumerate(tqdm(dirs)):
        for log in os.listdir(data_path + dir):
            depth_path = data_path + dir + '/' + log+'/depth/'
            img_path = data_path + dir + '/' + log+'/img/'
            physical_data_path = data_path + dir + '/' + log + '/physical_data/'
            if os.path.exists(physical_data_path):
                #delete the physical_data folder
                os.system('rm -r '+physical_data_path)
                # continue
            if not os.path.exists(physical_data_path):
                os.makedirs(physical_data_path)
            # else:
            #     continue
            files = os.listdir(depth_path)
            landmarks = pickle.load(open(data_path + dir + '/' + log + '/landmarks.pkl', 'rb'))

            for landmark in landmarks:
                # print   (landmark)
                if landmark[1] == []:
                    continue
                i = landmark[0][:-4]
                i = i[11:]
                i = int(i)-6
                # print(i)
                if i <1:
                    continue
                # print(i)
                with open(depth_path + 'depth_data'+str(i+3)+'.npy', 'rb') as f:
                    depth = np.load(f)
                # print(depth.shape)
                eyebow1 = [landmark[1][0][1][19], landmark[1][0][0][19]]
                eyebow2 = [landmark[1][0][1][24], landmark[1][0][0][24]]
                mouse1 = [landmark[1][0][1][48], landmark[1][0][0][48]]
                mouse2 = [landmark[1][0][1][54], landmark[1][0][0][54]]
                orientation = cal_orientation(eyebow1, eyebow2, mouse1, mouse2, depth)
                origin_H = len(depth)
                origin_W = len(depth[0])
                depth = cv2.resize(depth, (192, 108))
                W = len(depth[0])
                H = len(depth)
                scale = origin_H/H
                x_min = 63
                x_max = 0
                y_min = 63
                y_max = 0
                for index in range(len(landmark[1][0][0])):
                    h = int(landmark[1][0][1][index]//scale)
                    w = int(landmark[1][0][0][index]//scale)
                    if depth[h][w] > 100.5  and depth[h][w] < 1300.5:
                        zone = find_zone(depth, h, w, H, W)
                        x, y = zone[0], zone[1]
                        if x>=0 and x<64 and y>=0 and y<64:
                            if x < x_min:
                                x_min = x
                            if x > x_max:
                                x_max = x
                            if y < y_min:
                                y_min = y
                            if y > y_max:
                                y_max = y
                detection = [x_min, x_max, y_min, y_max]
                        
                #resize the depth image to 360*640
                # depth = cv2.resize(depth, (640, 360))
                
                # print(depth.shape)
                # with open(img_path + 'color_image'+str(i+1)+'.npy', 'rb') as f:
                #     img = pickle.load(f)
                
                # print(landmark[1][0].shape)
                # h_min = int(min(landmark[1][0][1])//3)
                # h_max = int(max(landmark[1][0][1])//3)
                # w_min = int(min(landmark[1][0][0])//3)
                # w_max = int(max(landmark[1][0][0])//3)
                # h_min = int(min(landmark[1][0][1])//scale)
                # # print(h_min)
                # h_max = int(max(landmark[1][0][1])//scale)
                # w_min = int(min(landmark[1][0][0])//scale)
                # w_max = int(max(landmark[1][0][0])//scale)
                h_min = 0
                w_min = 0
                h_max = H-1
                w_max = W-1

                D3_map = np.zeros((H,W,8))
                physical_data = [[[] for _ in range(64)] for _ in range(64)]
                saved_data = []
                # print(len(physical_data))
                # print(len(physical_data[0]))
                for h in range(h_min, h_max+1):
                    for w in range(w_min, w_max+1):
                        # if depth[h][w] > 100.5  and depth[h][w] < 1300.5:
                        if depth[h][w] > 20:
                            zone = find_zone(depth, h, w, H, W)
                            D3_map[h][w] = np.array([zone[2],zone[3],depth[h][w], zone[0], zone[1],0,0,0])
                for h in range(h_min, h_max+1):
                    for w in range(w_min, w_max+1):
                        x = D3_map[h][w][0]
                        y = D3_map[h][w][1]
                        z = D3_map[h][w][2]
                        if int(D3_map[h][w][3])<0 or int(D3_map[h][w][3])>63 or int(D3_map[h][w][4])<0 or int(D3_map[h][w][4])>63 or D3_map[h][w][2] < 1:    
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
                # print(i)
                            # print(1)
                if len(physical_data[0]) == 0:
                    print(i)
                    continue          
                #save the physical_data
                
                with open(data_path + dir + '/' + log + '/physical_data/'+'physical'+str(i)+'.pkl', 'wb') as f:
                    saved_data = [physical_data, orientation, detection]
                    
                    pickle.dump(saved_data, f)

def get_depth_orientation_files(data_path):
    for dir in tqdm(os.listdir(data_path)):

        for log in os.listdir(data_path + dir):
            physical_data_path = data_path + dir + '/' + log + '/physical_data/'
            files = os.listdir(physical_data_path)
            files.sort()
            depth_orintation_file_path = data_path + dir + '/' + log + '/depth_orientation/'
            # if os.path.exists(depth_orintation_file_path):
            #     continue
            if not os.path.exists(depth_orintation_file_path):
                os.makedirs(depth_orintation_file_path)
            for file in files:
                with open(physical_data_path + file, 'rb') as f:
                    physical_data = pickle.load(f)
                orientation, depth = get_detth_orientation(physical_data[0])
                index = file.split('.')[0]
                index = int(index[8:])
                with open(depth_orintation_file_path + 'depth_orientation'+str(index)+'.pkl', 'wb') as f:
                    pickle.dump([orientation, depth], f)
    return
                
        

def get_detth_orientation(physical_data):
    orientation = torch.zeros((64,64))
    depth = torch.zeros((64,64))
    for h in range(64):
        for w in range(64):
            if len(physical_data[h][w]) == 0:
                continue
            n = 0
            all_points = []
            if len(physical_data[h][w]) == 0:
                continue
            for points in physical_data[h][w]:
                # print(len(physical_data[h][w]))
                position = points[0]
                all_points.append(position)
                # orientation[h][w] = points[1][0]
                depth[h][w] += points[0][2]
                n+=1
            depth[h][w] = depth[h][w]/n
            all_points = np.array(all_points)
            center = all_points.mean(axis=0)
            points_centered = points - center
            cov_matrix = np.cov(points_centered, rowvar=False)
            eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
            normal_vector = eigenvectors[:, np.argmin(eigenvalues)]

            # calculate the angle between the normal vector and the center
            angle = np.arccos(np.abs(np.dot(normal_vector, center) / (np.linalg.norm(normal_vector) * np.linalg.norm(center)+1e-6)))
            orientation[h][w] = angle
    return orientation, depth




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
                files.sort()
                tof_data =  data_path + dir + '/' + log + '/tof_data.pkl'
                with open(tof_data, 'rb') as f:
                    tof_data = pickle.load(f)
                self.tof_data.append(tof_data)
                for file in files:
                    # with open(physical_data_path + file, 'rb') as f:
                    #     physical_data = pickle.load(f)
                    # self.data.append(physical_data)
                    file_path = physical_data_path + file
                    tof_frame_index = file_path.split('/')[-1]
                    tof_frame_index = tof_frame_index.split('.')[0]
                    tof_frame_index = int(tof_frame_index[8:])-1
                    tmp = tof_data[tof_frame_index]
                    if tmp[1] is None:
                        continue
                    
                    self.data.append([ind, file_path])
                ind += 1
        # random select 30 samples from the dataset, save the index in a list, the data is a list, so only choose the index
        # for i in range(30):
        #     index = np.random.randint(0, len(self.data))
        #     while index in self.indexs:
        #         index = np.random.randint(0, len(self.data))
        #     self.indexs.append(index)
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
        orientation, depth = self.get_ori_depth(physical_data)
        orientation = orientation.to(self.device).float()
        depth = depth.to(self.device).float()
        tof_frame_index = file_path.split('/')[-1]
        tof_frame_index = tof_frame_index.split('.')[0]
        tof_frame_index = int(tof_frame_index[8:])-1
        tof_data = self.tof_data[self.data[idx][0]][tof_frame_index]
        # print(len(tof_data[1]))

        tmp = np.flip(tof_data[1],0)
        # tmp = np.flip(tmp,1)
        tof_data = torch.tensor(tmp.copy())
        tof_data = tof_data.to(self.device)
        tof_data = tof_data.float()
        return orientation,depth, tof_data
    
    def get_ori_depth(self, physical_data):
        orientation = torch.zeros((16,16))
        depth = torch.zeros((16,16))
        for h in range(16):
            for w in range(16):
                if len(physical_data[h][w]) == 0:
                    continue
                n = 0
                all_points = []
                for points in physical_data[h][w]:
                    position = points[0]
                    all_points.append(position)
                    # orientation[h][w] = points[1][0]
                    depth[h][w] += points[0][2]
                    n+=1
                depth[h][w] = depth[h][w]/n
                all_points = np.array(all_points)
                center = all_points.mean(axis=0)
                points_centered = points - center
                cov_matrix = np.cov(points_centered, rowvar=False)
                eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
                normal_vector = eigenvectors[:, np.argmin(eigenvalues)]

                # calculate the angle between the normal vector and the center
                angle = np.arccos(np.abs(np.dot(normal_vector, center) / (np.linalg.norm(normal_vector) * np.linalg.norm(center)+1e-6)))
                orientation[h][w] = angle
        return orientation, depth
    

class physical_dataset_mask(Dataset):
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
                # files.sort()
                files = sorted(files, key=lambda x: int(re.search(r'\d+', x).group()))
                # print(files)
                label = dir.split('_')[2]
                label = int(label[1:])

                tof_data =  data_path + dir + '/' + log + '/tof_data.pkl'
                with open(tof_data, 'rb') as f:
                    tof_data = pickle.load(f)
                self.tof_data.append(tof_data)
                for file in files[7:]:
                    # with open(physical_data_path + file, 'rb') as f:
                    #     physical_data = pickle.load(f)
                    # self.data.append(physical_data)
                    file_path = physical_data_path + file
                    tof_frame_index = file_path.split('/')[-1]
                    tof_frame_index = tof_frame_index.split('.')[0]
                    tof_frame_index = int(tof_frame_index[8:])-1
                    tmp = tof_data[tof_frame_index]
                    if tmp[1] is None or tmp[0] is None or tmp[2] is None:
                        continue
                    
                    self.data.append([ind, file_path, label])
                ind += 1
        # random select 30 samples from the dataset, save the index in a list, the data is a list, so only choose the index
        # for i in range(30):
        #     index = np.random.randint(0, len(self.data))
        #     while index in self.indexs:
        #         index = np.random.randint(0, len(self.data))
        #     self.indexs.append(index)
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
        # print(len(physical_data))
        # print(file_path)
        orientation, depth = self.get_ori_depth(physical_data[0])
        face_orientation = physical_data[1]
        # print(face_orientation)
        orientation = orientation.to(self.device).float()
        depth = depth.to(self.device).float()
        tof_frame_index = file_path.split('/')[-1]
        tof_frame_index = tof_frame_index.split('.')[0]
        tof_frame_index = int(tof_frame_index[8:])-1
        tof_data = self.tof_data[self.data[idx][0]][tof_frame_index]
        # print(len(tof_data[1]))
        # class_label = torch.zeros(8)
        class_label = self.data[idx][2]
        # class_label = class_label.to(self.device).float()

        tmp = np.flip(tof_data[1],0)
        input_depth = np.flip(tof_data[0],0)
        input_depth = torch.tensor(input_depth.copy())
        input_depth = input_depth.to(self.device)
        input_depth = input_depth.float()
        input_reflectance = np.flip(tof_data[2],0)
        input_reflectance = torch.tensor(input_reflectance.copy())
        input_reflectance = input_reflectance.to(self.device)
        input_reflectance = input_reflectance.float()
        # tmp = np.flip(tmp,1)
        tof_data = torch.tensor(tmp.copy())
        tof_data = tof_data.to(self.device)
        tof_data = tof_data.float()
        empty = torch.zeros((8,8,10))
        empty = empty.to(self.device).float()
        tof_data = torch.cat((tof_data, empty), 2)

        mask = (depth != 0).float()
        # calculate the detection position(x,y,xl,yl) from the mask
        detect = torch.zeros(4)
        # print(depth)
        # print(mask)
        indices = torch.nonzero(mask == 1, as_tuple=False)
        # for h in range(16):
        #     for w in range(16):
        #         if mask[h][w] == 1:
        #             detect[0] = w
        #             detect[1] = h
        #             break
        # for h in range(15,-1,-1):
        #     for w in range(15,-1,-1):
        #         if mask[h][w] == 1:
        #             detect[2] = w-detect[0]
        #             detect[3] = h-detect[1]
        #             break
        # detect[0] = torch.min(indices[:, 1])
        # detect[1] = torch.min(indices[:, 0])
        # detect[2] = torch.max(indices[:, 1])-torch.min(indices[:, 1])
        # detect[3] = torch.max(indices[:, 0])-torch.min(indices[:, 0])
        detect[0] = torch.min(indices[:, 1])
        detect[1] = torch.min(indices[:, 0])
        detect[2] = torch.max(indices[:, 1])
        detect[3] = torch.max(indices[:, 0])
        detect  = detect/63*15
        target = {
            'boxes': detect.unsqueeze(0).float().to(self.device),
            'labels': torch.tensor([1], dtype=torch.int64).to(self.device)  # 标签1表示人脸
        }
        # print(detect)
        return orientation,depth, detect.to(self.device), input_depth, input_reflectance, target
    
    def get_ori_depth(self, physical_data):
        orientation = torch.zeros((64,64))
        depth = torch.zeros((64,64))
        for h in range(64):
            for w in range(64):
                if len(physical_data[h][w]) == 0:
                    continue
                n = 0
                all_points = []
                for points in physical_data[h][w]:
                    position = points[0]
                    all_points.append(position)
                    # orientation[h][w] = points[1][0]
                    depth[h][w] += points[0][2]
                    n+=1
                depth[h][w] = depth[h][w]/n
                all_points = np.array(all_points)
                center = all_points.mean(axis=0)
                points_centered = points - center
                cov_matrix = np.cov(points_centered, rowvar=False)
                eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
                normal_vector = eigenvectors[:, np.argmin(eigenvalues)]

                # calculate the angle between the normal vector and the center
                angle = np.arccos(np.abs(np.dot(normal_vector, center) / (np.linalg.norm(normal_vector) * np.linalg.norm(center)+1e-6)))
                orientation[h][w] = angle
        return orientation, depth
    
class physical_dataset_TransferLearning(Dataset):
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
                img_path = data_path + dir + '/' + log + '/img/'
                files = os.listdir(physical_data_path)
                # files.sort()
                files = sorted(files, key=lambda x: int(re.search(r'\d+', x).group()))
                # print(files)
                label = dir.split('_')[2]
                label = int(label[1:])

                tof_data =  data_path + dir + '/' + log + '/tof_data.pkl'
                with open(tof_data, 'rb') as f:
                    tof_data = pickle.load(f)
                self.tof_data.append(tof_data)
                for file in files[7:]:
                    # with open(physical_data_path + file, 'rb') as f:
                    #     physical_data = pickle.load(f)
                    # self.data.append(physical_data)
                    file_path = physical_data_path + file
                    tof_frame_index = file_path.split('/')[-1]
                    tof_frame_index = tof_frame_index.split('.')[0]
                    tof_frame_index = int(tof_frame_index[8:])-1
                    save_img_path = img_path + 'color_image'+str(tof_frame_index+1)+'.npy'
                    tmp = tof_data[tof_frame_index]
                    if tmp[1] is None or tmp[0] is None or tmp[2] is None:
                        continue
                    
                    self.data.append([ind, file_path, label, save_img_path])
                ind += 1
        # random select 30 samples from the dataset, save the index in a list, the data is a list, so only choose the index
        # for i in range(30):
        #     index = np.random.randint(0, len(self.data))
        #     while index in self.indexs:
        #         index = np.random.randint(0, len(self.data))
        #     self.indexs.append(index)
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
        # print(len(physical_data))
        # print(file_path)
        orientation, depth = self.get_ori_depth(physical_data[0])
        face_orientation = physical_data[1]
        # print(face_orientation)
        orientation = orientation.to(self.device).float()
        depth = depth.to(self.device).float()
        tof_frame_index = file_path.split('/')[-1]
        tof_frame_index = tof_frame_index.split('.')[0]
        tof_frame_index = int(tof_frame_index[8:])-1
        tof_data = self.tof_data[self.data[idx][0]][tof_frame_index]
        # print(len(tof_data[1]))
        # class_label = torch.zeros(8)
        class_label = self.data[idx][2]
        # class_label = class_label.to(self.device).float()

        image_path = self.data[idx][3]
        with open(image_path, 'rb') as f:
            img = np.load(f)

        tmp = np.flip(tof_data[1],0)
        input_depth = np.flip(tof_data[0],0)
        input_depth = torch.tensor(input_depth.copy())
        input_depth = input_depth.to(self.device)
        input_depth = input_depth.float()
        input_reflectance = np.flip(tof_data[2],0)
        input_reflectance = torch.tensor(input_reflectance.copy())
        input_reflectance = input_reflectance.to(self.device)
        input_reflectance = input_reflectance.float()
        # tmp = np.flip(tmp,1)
        tof_data = torch.tensor(tmp.copy())
        tof_data = tof_data.to(self.device)
        tof_data = tof_data.float()
        mask = (depth != 0).float()
        # calculate the detection position(x,y,xl,yl) from the mask
        detect = torch.zeros(4)
        # print(depth)
        # print(mask)
        indices = torch.nonzero(mask == 1, as_tuple=False)
        # for h in range(16):
        #     for w in range(16):
        #         if mask[h][w] == 1:
        #             detect[0] = w
        #             detect[1] = h
        #             break
        # for h in range(15,-1,-1):
        #     for w in range(15,-1,-1):
        #         if mask[h][w] == 1:
        #             detect[2] = w-detect[0]
        #             detect[3] = h-detect[1]
        #             break
        # detect[0] = torch.min(indices[:, 1])
        # detect[1] = torch.min(indices[:, 0])
        # detect[2] = torch.max(indices[:, 1])-torch.min(indices[:, 1])
        # detect[3] = torch.max(indices[:, 0])-torch.min(indices[:, 0])
        detect[0] = torch.min(indices[:, 1])
        detect[1] = torch.min(indices[:, 0])
        detect[2] = torch.max(indices[:, 1])
        detect[3] = torch.max(indices[:, 0])
        detect  = detect/15*7
        target = {
            'boxes': detect.unsqueeze(0).float().to(self.device),
            'labels': torch.tensor([1], dtype=torch.int64).to(self.device)  # 标签1表示人脸
        }
        # print(detect)
        return orientation,depth, detect.to(self.device), input_depth, input_reflectance, target, torch.tensor(img,dtype=torch.float).to(self.device)
    
    def get_ori_depth(self, physical_data):
        orientation = torch.zeros((16,16))
        depth = torch.zeros((16,16))
        for h in range(16):
            for w in range(16):
                if len(physical_data[h][w]) == 0:
                    continue
                n = 0
                all_points = []
                for points in physical_data[h][w]:
                    position = points[0]
                    all_points.append(position)
                    # orientation[h][w] = points[1][0]
                    depth[h][w] += points[0][2]
                    n+=1
                depth[h][w] = depth[h][w]/n
                all_points = np.array(all_points)
                center = all_points.mean(axis=0)
                points_centered = points - center
                cov_matrix = np.cov(points_centered, rowvar=False)
                eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
                normal_vector = eigenvectors[:, np.argmin(eigenvalues)]

                # calculate the angle between the normal vector and the center
                angle = np.arccos(np.abs(np.dot(normal_vector, center) / (np.linalg.norm(normal_vector) * np.linalg.norm(center)+1e-6)))
                orientation[h][w] = angle
        return orientation, depth
    
class physical_dataset_baseline(Dataset):
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
                # files.sort()
                files = sorted(files, key=lambda x: int(re.search(r'\d+', x).group()))
                # print(files)
                label = dir.split('_')[2]
                label = int(label[1:])

                tof_data =  data_path + dir + '/' + log + '/tof_data.pkl'
                with open(tof_data, 'rb') as f:
                    tof_data = pickle.load(f)
                self.tof_data.append(tof_data)
                for file in files[7:]:
                    # with open(physical_data_path + file, 'rb') as f:
                    #     physical_data = pickle.load(f)
                    # self.data.append(physical_data)
                    file_path = physical_data_path + file
                    tof_frame_index = file_path.split('/')[-1]
                    tof_frame_index = tof_frame_index.split('.')[0]
                    tof_frame_index = int(tof_frame_index[8:])-1
                    tmp = tof_data[tof_frame_index]
                    if tmp[1] is None or tmp[0] is None or tmp[2] is None:
                        continue
                    
                    self.data.append([ind, file_path, label])
                ind += 1
        # random select 30 samples from the dataset, save the index in a list, the data is a list, so only choose the index
        # for i in range(30):
        #     index = np.random.randint(0, len(self.data))
        #     while index in self.indexs:
        #         index = np.random.randint(0, len(self.data))
        #     self.indexs.append(index)
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
        # print(len(physical_data))
        # print(file_path)
        orientation, depth = self.get_ori_depth(physical_data[0])
        face_orientation = physical_data[1]
        # print(face_orientation)
        orientation = orientation.to(self.device).float()
        depth = depth.to(self.device).float()
        tof_frame_index = file_path.split('/')[-1]
        tof_frame_index = tof_frame_index.split('.')[0]
        tof_frame_index = int(tof_frame_index[8:])-1
        tof_data = self.tof_data[self.data[idx][0]][tof_frame_index]
        # print(len(tof_data[1]))
        # class_label = torch.zeros(8)
        class_label = self.data[idx][2]
        # class_label = class_label.to(self.device).float()

        tmp = np.flip(tof_data[1],0)
        input_depth = np.flip(tof_data[0],0)
        input_depth = torch.tensor(input_depth.copy())
        input_depth = input_depth.to(self.device)
        input_depth = input_depth.float()
        input_reflectance = np.flip(tof_data[2],0)
        input_reflectance = torch.tensor(input_reflectance.copy())
        input_reflectance = input_reflectance.to(self.device)
        input_reflectance = input_reflectance.float()
        # tmp = np.flip(tmp,1)
        tof_data = torch.tensor(tmp.copy())
        tof_data = tof_data.to(self.device)
        tof_data = tof_data.float()
        mask = (depth != 0).float()
        # calculate the detection position(x,y,xl,yl) from the mask
        detect = torch.zeros(4)
        # print(depth)
        # print(mask)
        indices = torch.nonzero(mask == 1, as_tuple=False)
        # for h in range(16):
        #     for w in range(16):
        #         if mask[h][w] == 1:
        #             detect[0] = w
        #             detect[1] = h
        #             break
        # for h in range(15,-1,-1):
        #     for w in range(15,-1,-1):
        #         if mask[h][w] == 1:
        #             detect[2] = w-detect[0]
        #             detect[3] = h-detect[1]
        #             break
        # detect[0] = torch.min(indices[:, 1])
        # detect[1] = torch.min(indices[:, 0])
        # detect[2] = torch.max(indices[:, 1])-torch.min(indices[:, 1])
        # detect[3] = torch.max(indices[:, 0])-torch.min(indices[:, 0])
        detect[0] = torch.min(indices[:, 1])
        detect[1] = torch.min(indices[:, 0])
        detect[2] = torch.max(indices[:, 1])
        detect[3] = torch.max(indices[:, 0])
        detect  = detect/15*7
        target = {
            'boxes': detect.unsqueeze(0).float().to(self.device),
            'labels': torch.tensor([class_label], dtype=torch.int64).to(self.device)  # 标签1表示人脸
        }
        # print(detect)
        return orientation,depth, detect.to(self.device), input_depth, input_reflectance, target
    
    def get_ori_depth(self, physical_data):
        orientation = torch.zeros((16,16))
        depth = torch.zeros((16,16))
        for h in range(16):
            for w in range(16):
                if len(physical_data[h][w]) == 0:
                    continue
                n = 0
                all_points = []
                for points in physical_data[h][w]:
                    position = points[0]
                    all_points.append(position)
                    # orientation[h][w] = points[1][0]
                    depth[h][w] += points[0][2]
                    n+=1
                depth[h][w] = depth[h][w]/n
                all_points = np.array(all_points)
                center = all_points.mean(axis=0)
                points_centered = points - center
                cov_matrix = np.cov(points_centered, rowvar=False)
                eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
                normal_vector = eigenvectors[:, np.argmin(eigenvalues)]

                # calculate the angle between the normal vector and the center
                angle = np.arccos(np.abs(np.dot(normal_vector, center) / (np.linalg.norm(normal_vector) * np.linalg.norm(center)+1e-6)))
                orientation[h][w] = angle
        return orientation, depth
    


class physical_dataset_class(Dataset):
    def __init__(self, data_path, type):
        self.data_path = data_path
        self.data = []
        self.tof_data = []
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  
        ind = 0
        self.indexs = []
        class_dict = {'dongsheng':0, 'hou':1, 'hrm':2, 'sijie':3, 'sst':4, 'xie':5, 'yjt':6, 'yuemin':7}
        for dir in os.listdir(data_path):
            name = dir.split('_')[0]
            number = int(dir.split('_')[1])
            if number != type:
                continue
            class_label = class_dict[name]
            for log in os.listdir(data_path + dir):
                physical_data_path = data_path + dir + '/' + log + '/physical_data/'

                files = os.listdir(physical_data_path)
                files.sort()
                tof_data =  data_path + dir + '/' + log + '/tof_data.pkl'
                with open(tof_data, 'rb') as f:
                    tof_data = pickle.load(f)
                self.tof_data.append(tof_data)
                for file in files:
                    # with open(physical_data_path + file, 'rb') as f:
                    #     physical_data = pickle.load(f)
                    # self.data.append(physical_data)
                    file_path = physical_data_path + file
                    tof_frame_index = file_path.split('/')[-1]
                    tof_frame_index = tof_frame_index.split('.')[0]
                    tof_frame_index = int(tof_frame_index[8:])-1
                    tmp = tof_data[tof_frame_index]
                    if tmp[1] is None:
                        continue
                    
                    self.data.append([ind, file_path, class_label])
                ind += 1
        # random select 30 samples from the dataset, save the index in a list, the data is a list, so only choose the index
        # for i in range(30):
        #     index = np.random.randint(0, len(self.data))
        #     while index in self.indexs:
        #         index = np.random.randint(0, len(self.data))
        #     self.indexs.append(index)
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
        orientation, depth = self.get_ori_depth(physical_data)
        orientation = orientation.to(self.device).float()
        depth = depth.to(self.device).float()
        tof_frame_index = file_path.split('/')[-1]
        tof_frame_index = tof_frame_index.split('.')[0]
        tof_frame_index = int(tof_frame_index[8:])-1
        tof_data = self.tof_data[self.data[idx][0]][tof_frame_index]
        # print(len(tof_data[1]))

        tmp = np.flip(tof_data[1],0)
        # tmp = np.flip(tmp,1)
        tof_data = torch.tensor(tmp.copy())
        tof_data = tof_data.to(self.device)
        tof_data = tof_data.float()
        class_label = self.data[idx][2]
        #generate the class label of one hot encoding
        class_label = torch.zeros(8)
        class_label[self.data[idx][2]] = 1
        class_label = class_label.to(self.device).float()
        return orientation,depth, tof_data, class_label
    
    def get_ori_depth(self, physical_data):
        orientation = torch.zeros((16,16))
        depth = torch.zeros((16,16))
        for h in range(16):
            for w in range(16):
                if len(physical_data[h][w]) == 0:
                    continue
                n = 0
                all_points = []
                for points in physical_data[h][w]:
                    position = points[0]
                    all_points.append(position)
                    # orientation[h][w] = points[1][0]
                    depth[h][w] += points[0][2]
                    n+=1
                depth[h][w] = depth[h][w]/n
                all_points = np.array(all_points)
                center = all_points.mean(axis=0)
                points_centered = points - center
                cov_matrix = np.cov(points_centered, rowvar=False)
                eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
                normal_vector = eigenvectors[:, np.argmin(eigenvalues)]

                # calculate the angle between the normal vector and the center
                angle = np.arccos(np.abs(np.dot(normal_vector, center) / (np.linalg.norm(normal_vector) * np.linalg.norm(center)+1e-6)))
                orientation[h][w] = angle
        return orientation, depth
    
# class physical_dataset_class_FER(Dataset):
#     def __init__(self, data_path, type,train_flag):
#         self.data_path = data_path
#         self.data = []
#         self.tof_data = []
#         self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  
#         ind = 0
#         self.indexs = []
#         # class_dict = {'dongsheng':0, 'hou':1, 'hrm':2, 'sijie':3, 'sst':4, 'xie':5, 'yjt':6, 'yuemin':7}
#         # for dir in os.listdir(data_path)):
#         for dir in tqdm(os.listdir(data_path)):
#             emotion = int(dir.split('_')[2][1])
#             number = int(dir.split('_')[1][1:])
#             if train_flag and number == type:
#                 continue
#             elif train_flag ==False and number!=type:
#                 continue
#             class_label = emotion
#             for log in os.listdir(data_path + dir):
#                 physical_data_path = data_path + dir + '/' + log + '/physical_data/'
#                 img_path = data_path + dir + '/' + log + '/img/'
#                 files = os.listdir(physical_data_path)
#                 files.sort()
#                 tof_data =  data_path + dir + '/' + log + '/tof_data.pkl'
#                 with open(tof_data, 'rb') as f:
#                     tof_data = pickle.load(f)
#                 self.tof_data.append(tof_data)
#                 for file in files:
#                     # with open(physical_data_path + file, 'rb') as f:
#                     #     physical_data = pickle.load(f)
#                     # self.data.append(physical_data)
#                     file_path = physical_data_path + file
#                     with open(file_path, 'rb') as f:
#                         physical_data = pickle.load(f)
#                     orientation, depth = self.get_ori_depth(physical_data[0])
#                     tof_frame_index = file_path.split('/')[-1]
#                     tof_frame_index = tof_frame_index.split('.')[0]
#                     tof_frame_index = int(tof_frame_index[8:])-1
#                     save_img_path = img_path + 'color_image'+str(tof_frame_index+1)+'.npy'
#                     tmp = tof_data[tof_frame_index]
#                     if tmp[1] is None or tmp[0] is None or tmp[2] is None:
#                         continue
                    
#                     self.data.append([ind, file_path, class_label, orientation, depth,save_img_path])
#                     # self.data.append([ind, file_path, class_label, 1,1,save_img_path])
#                 ind += 1
#         # random select 30 samples from the dataset, save the index in a list, the data is a list, so only choose the index
#         # for i in range(30):
#         #     index = np.random.randint(0, len(self.data))
#         #     while index in self.indexs:
#         #         index = np.random.randint(0, len(self.data))
#         #     self.indexs.append(index)
#     def __len__(self):
#         return len(self.data)
#         # return 30
#     def __getitem__(self, idx):
#         # get the random index between 0 and len(self.data)
#         # idx = np.random.randint(0, len(self.data))
#         # idx = self.indexs[idx]
#         file_path = self.data[idx][1]
#         with open(file_path, 'rb') as f:
#             physical_data = pickle.load(f)
#         # orientation, depth = self.get_ori_depth(physical_data[0])
#         orientation = self.data[idx][3]
#         depth = self.data[idx][4]
#         image_path = self.data[idx][5]

#         face_orientation = physical_data[1]
#         orientation = orientation.to(self.device).float()
#         depth = depth.to(self.device).float()
#         with open(image_path, 'rb') as f:
#             img = np.load(f)
#         img = np.resize(img, (540, 960, 3))
#         tof_frame_index = file_path.split('/')[-1]
#         tof_frame_index = tof_frame_index.split('.')[0]
#         tof_frame_index = int(tof_frame_index[8:])-1
#         tof_data = self.tof_data[self.data[idx][0]][tof_frame_index]
#         # print(len(tof_data[1]))

#         tmp = np.flip(tof_data[1],0)
#         input_depth = np.flip(tof_data[0],0)
#         input_depth = torch.tensor(input_depth.copy())
#         input_depth = input_depth.to(self.device)
#         input_depth = input_depth.float()
#         input_reflectance = np.flip(tof_data[2],0)
#         input_reflectance = torch.tensor(input_reflectance.copy())
#         input_reflectance = input_reflectance.to(self.device)
#         input_reflectance = input_reflectance.float()
#         # tmp = np.flip(tmp,1)
#         tof_data = torch.tensor(tmp.copy())
#         tof_data = tof_data.to(self.device)
#         tof_data = tof_data.float()
#         empty = torch.zeros((8,8,10))
#         empty = empty.to(self.device).float()
#         tof_data = torch.cat((tof_data, empty), 2)
#         class_label = self.data[idx][2]
#         #generate the class label of one hot encoding
#         class_label = torch.zeros(7)
#         class_label[self.data[idx][2]] = 1
#         class_label = class_label.to(self.device).float()
#         detect = torch.zeros(4)
#         mask = (depth != 0).float()
#         indices = torch.nonzero(mask == 1, as_tuple=False)
#         detect[0] = torch.min(indices[:, 1])
#         detect[1] = torch.min(indices[:, 0])
#         detect[2] = torch.max(indices[:, 1])
#         detect[3] = torch.max(indices[:, 0])
#         detect  = detect/15*7
#         target = {
#             'boxes': detect.unsqueeze(0).float().to(self.device),
#             'labels': torch.tensor([1], dtype=torch.int64).to(self.device)  # 标签1表示人脸
#         }


#         return orientation,depth, tof_data, class_label, face_orientation, input_depth,input_reflectance,target,torch.tensor(img,dtype=torch.float).to(self.device)
    
#     def get_ori_depth(self, physical_data):
#         # print(len(physical_data))
#         # print(physical_data)
#         orientation = torch.zeros((64,64))
#         depth = torch.zeros((64,64))
#         for h in range(64):
#             for w in range(64):
#                 if len(physical_data[h][w]) == 0:
#                     continue
#                 n = 0
#                 all_points = []
#                 if len(physical_data[h][w]) == 0:
#                     continue
#                 for points in physical_data[h][w]:
#                     # print(len(physical_data[h][w]))
#                     position = points[0]
#                     all_points.append(position)
#                     # orientation[h][w] = points[1][0]
#                     depth[h][w] += points[0][2]
#                     n+=1
#                 depth[h][w] = depth[h][w]/n
#                 all_points = np.array(all_points)
#                 center = all_points.mean(axis=0)
#                 points_centered = points - center
#                 cov_matrix = np.cov(points_centered, rowvar=False)
#                 eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
#                 normal_vector = eigenvectors[:, np.argmin(eigenvalues)]

#                 # calculate the angle between the normal vector and the center
#                 angle = np.arccos(np.abs(np.dot(normal_vector, center) / (np.linalg.norm(normal_vector) * np.linalg.norm(center)+1e-6)))
#                 orientation[h][w] = angle
#         return orientation, depth
    
class physical_dataset_class_FER(Dataset):
    def __init__(self, data_path, type,train_flag,environ = [], distance = [],in_or_out = False, single = False):
        self.data_path = data_path
        self.single = single
        self.data = []
        self.tof_data = []
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  
        ind = 0
        self.indexs = []
        # class_dict = {'dongsheng':0, 'hou':1, 'hrm':2, 'sijie':3, 'sst':4, 'xie':5, 'yjt':6, 'yuemin':7}
        # for dir in os.listdir(data_path)):
        dirs = os.listdir(data_path)
        sorted_dirs = sorted(dirs, key=lambda x: int(re.search(r'\d+', x).group()))
        self.indecator = []
        number_data = 0
        for dir in tqdm(os.listdir(data_path)):
            emotion = int(dir.split('_')[2][1])
            number = int(dir.split('_')[1][1:])
            dis = dir.split('_')[3]
            env = dir.split('_')[0]

            is_long = False
            if distance == '70-100':
                is_long = True
            if train_flag and number in type:
                continue
            elif train_flag ==False and number not in type:
                continue
            if env not in environ:
                # print(env)
                continue
            if in_or_out:
                if dis in distance:
                    continue
            if not in_or_out:
                if dis not in distance:
                    continue
            # if dis in distance:
            #     continue
            class_label = emotion
            # files = os.listdir(data_path + dir)
            # files = sorted(files, key=lambda x: int(re.search(r'\d+', x).group()))
            for log in os.listdir(data_path + dir):
                physical_data_path = data_path + dir + '/' + log + '/physical_data/'
                img_path = data_path + dir + '/' + log + '/img/'
                depth_orientation_path = data_path + dir + '/' + log + '/depth_orientation/'
                files = os.listdir(physical_data_path)
                # files.sort()\
                files = sorted(files, key=lambda x: int(re.search(r'\d+', x).group()))
                tof_data =  data_path + dir + '/' + log + '/tof_data.pkl'
                #preload the depth and orientation and the target_data, range
                with open(tof_data, 'rb') as f:
                    tof_data = pickle.load(f)
                self.tof_data.append(tof_data)
                tmp_i = 0
                for file in files:
                    # with open(physical_data_path + file, 'rb') as f:
                    #     physical_data = pickle.load(f)
                    # self.data.append(physical_data)
                    
                    file_path = physical_data_path + file
                    # with open(file_path, 'rb') as f:
                    #     physical_data = pickle.load(f)
                    # orientation, depth = self.get_ori_depth(physical_data[0])
                    tof_frame_index = file_path.split('/')[-1]
                    tof_frame_index = tof_frame_index.split('.')[0]
                    tof_frame_index = int(tof_frame_index[8:])-1
                    if tof_frame_index < 7:
                        continue 
                    tem_flag = True
                    for jj in range(0,4):
                        tmp_physical_data_path = physical_data_path + 'physical'+str(tof_frame_index+1-jj)+'.pkl'
                        if tof_frame_index-jj < 0 :
                            tem_flag = False
                            break
                        tmp_tof = tof_data[tof_frame_index-jj]
                        if tmp_tof[1] is None or tmp_tof[0] is None or tmp_tof[2] is None:
                            tem_flag = False
                            break
                        if not os.path.exists(tmp_physical_data_path):
                            tem_flag = False
                            break
                    # if tem_flag:
                    #     self.indecator.append(number_data)
                    #     tmp_i += 1
                    # if tmp_i > 10:
                    #     break
                    save_img_path = img_path + 'color_image'+str(tof_frame_index+1)+'.npy'
                    save_depth_orientation_path = depth_orientation_path + 'depth_orientation'+str(tof_frame_index+1)+'.pkl'
                    if not os.path.exists(save_depth_orientation_path):
                        continue
                    with open(save_depth_orientation_path, 'rb') as f:
                        depth_orientation = pickle.load(f)
                    orientation = depth_orientation[0]
                    depth = depth_orientation[1]
                    tmp = tof_data[tof_frame_index]
                    if tmp[1] is None or tmp[0] is None or tmp[2] is None:
                        continue
                    with open(file_path, 'rb') as f:
                        physical_data = pickle.load(f)
                    face_orientation = physical_data[1]
                    detection = physical_data[2]
                    if tem_flag:
                        self.indecator.append(number_data)
                        self.data.append([ind, file_path, class_label, orientation, depth,save_img_path, face_orientation,detection,number,is_long])
                        number_data += 1
                        tmp_i += 1
                    # if tmp_i > 10:
                    #     break
                    # self.data.append([ind, file_path, class_label, 1,1,save_img_path])
                ind += 1
        # random select 30 samples from the dataset, save the index in a list, the data is a list, so only choose the index
        # for i in range(30):
        #     index = np.random.randint(0, len(self.data))
        #     while index in self.indexs:
        #         index = np.random.randint(0, len(self.data))
        #     self.indexs.append(index)
    def __len__(self):
        # return len(self.data)
        return len(self.indecator)
        # return 30
    def __getitem__(self, idx):
        # get the random index between 0 and len(self.data)
        # idx = np.random.randint(0, len(self.data))
        # idx = self.indexs[idx]
        idx = self.indecator[idx]
        start_time = time.time()
        # print(idx)
        # print(len(self.data))
        file_path = self.data[idx][1]
        # with open(file_path, 'rb') as f:
        #     physical_data = pickle.load(f)
        # orientation, depth = self.get_ori_depth(physical_data[0])
        orientation = self.data[idx][3]
        depth = self.data[idx][4]
        image_path = self.data[idx][5]
        is_long = self.data[idx][9]

        # face_orientation = physical_data[1]
        face_orientation = self.data[idx][6]
        face_orientation = torch.tensor(face_orientation).to(self.device).float()
        orientation = orientation.to(self.device).float()
        depth = depth.to(self.device).float()
        with open(image_path, 'rb') as f:
            img = np.load(f)
        img = np.resize(img, (540, 960, 3))
        tof_frame_index = file_path.split('/')[-1]
        tof_frame_index = tof_frame_index.split('.')[0]
        tof_frame_index = int(tof_frame_index[8:])-1
        tof_data = self.tof_data[self.data[idx][0]][tof_frame_index-3:tof_frame_index+1]
        # print(len(self.tof_data[self.data[idx][0]]))
        # print(tof_frame_index)
        # print(len(tof_data[1]))
        # end_time = time.time()
        # print('load data time1:', end_time-start_time)
        # start_time = time.time()

        # tmp = np.flip(tof_data[1],0)
        # print(len(tof_data))
        input_depth = np.flip(tof_data[-1][0],0)
        input_depth = torch.tensor(input_depth.copy())
        input_depth = input_depth.to(self.device)
        input_depth = input_depth.float()
        input_reflectance = np.flip(tof_data[-1][2],0)
        input_reflectance = torch.tensor(input_reflectance.copy())
        input_reflectance = input_reflectance.to(self.device)
        input_reflectance = input_reflectance.float()
        # tmp = np.flip(tmp,1)

        only_tof = []
        empty = torch.zeros((8,8,10))
        empty = empty.to(self.device).float()
        for i in range(len(tof_data)):
            # print(i)
            # print(tof_data[i][1].shape)
            tmp = np.flip(tof_data[i][1],0)
            tmp = torch.tensor(tmp.copy())
            tmp = tmp.to(self.device).float()
            if is_long:
                tmp = torch.cat((empty, tmp), 2)
            else:
                tmp = torch.cat((tmp, empty), 2)
            

            only_tof.append(tmp)
        # tof_data = torch.tensor(tmp.copy())
        # tof_data = tof_data.to(self.device)
        # tof_data = tof_data.float()
        tof_data = torch.stack(only_tof,0)
        
        
        
        class_label = self.data[idx][2]
        #generate the class label of one hot encoding
        class_label = torch.tensor([class_label]).to(self.device)
        # class_label = torch.zeros(7)
        # class_label[self.data[idx][2]] = 1
        # class_label = class_label.to(self.device).float()
        # detect = torch.zeros(4)
        # mask = (depth != 0).float()
        # indices = torch.nonzero(mask == 1, as_tuple=False)
        # detect[0] = torch.min(indices[:, 1])
        # detect[1] = torch.min(indices[:, 0])
        # detect[2] = torch.max(indices[:, 1])
        # detect[3] = torch.max(indices[:, 0])

        # detect  = detect/15*7
        # end_time = time.time()
        # print('load data time2:', end_time-start_time)
        # start_time = time.time()
        # detection = physical_data[2]
        detection = self.data[idx][7]
        uid = self.data[idx][8]-1
        # print(uid)
        uid = torch.tensor(uid).to(self.device)

        # generate mask based on the detection
        mask = torch.zeros((64,64)).to(self.device)
        #将detection四舍五入为整数
        detection = torch.tensor(detection).to(self.device).float()
        detection_tmp = detection.round().int()

        mask[detection_tmp[2]:detection_tmp[3],detection_tmp[0]:detection_tmp[1]] = 1
        detect = torch.tensor(detection).to(self.device).float()
        # detect = detect[[2,0,3,1]]# xmin,ymin,xmax,ymax
        detect = detect[[0,2,1,3]]
        # if detect[2] -detect[0] < 0.001:
            #detect[2] 向上取整，detect[0]向下取整，这两个都是浮点数现在
        detect = detect/63*7
        detect[2] = torch.ceil(detect[2])
        detect[0] = torch.floor(detect[0])
        detect[3] = torch.ceil(detect[3])
        detect[1] = torch.floor(detect[1])
        # print(detect)
        # detect = detect.permute
        
        target = {
            'boxes': detect.unsqueeze(0),
            'labels': torch.tensor([1], dtype=torch.int64).to(self.device)  # 标签1表示人脸
        }
        # end_time = time.time()
        # print('load data time3:', end_time-start_time)
        # start_time = time.time()
        # print(image_path)
        # print(class_label)
        if self.single:
            tof_data = tof_data[-1]


        return orientation,depth, tof_data, class_label, face_orientation, input_depth,input_reflectance,target,torch.tensor(img,dtype=torch.float).to(self.device), mask.float(), uid
    
    def get_ori_depth(self, physical_data):
        # print(len(physical_data))
        # print(physical_data)
        orientation = torch.zeros((64,64))
        depth = torch.zeros((64,64))
        for h in range(64):
            for w in range(64):
                if len(physical_data[h][w]) == 0:
                    continue
                n = 0
                all_points = []
                if len(physical_data[h][w]) == 0:
                    continue
                for points in physical_data[h][w]:
                    # print(len(physical_data[h][w]))
                    position = points[0]
                    all_points.append(position)
                    # orientation[h][w] = points[1][0]
                    depth[h][w] += points[0][2]
                    n+=1
                depth[h][w] = depth[h][w]/n
                all_points = np.array(all_points)
                center = all_points.mean(axis=0)
                points_centered = points - center
                cov_matrix = np.cov(points_centered, rowvar=False)
                eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
                normal_vector = eigenvectors[:, np.argmin(eigenvalues)]

                # calculate the angle between the normal vector and the center
                angle = np.arccos(np.abs(np.dot(normal_vector, center) / (np.linalg.norm(normal_vector) * np.linalg.norm(center)+1e-6)))
                orientation[h][w] = angle
        return orientation, depth


class physical_dataset_class_FER_infer(Dataset):
    def __init__(self, data_path, type,train_flag,environ = [], distance = [],in_or_out = False, single = False):
        self.data_path = data_path
        self.single = single
        self.data = []
        self.tof_data = []
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  
        ind = 0
        self.indexs = []
        # class_dict = {'dongsheng':0, 'hou':1, 'hrm':2, 'sijie':3, 'sst':4, 'xie':5, 'yjt':6, 'yuemin':7}
        # for dir in os.listdir(data_path)):
        dirs = os.listdir(data_path)
        sorted_dirs = sorted(dirs, key=lambda x: int(re.search(r'\d+', x).group()))
        self.indecator = []
        number_data = 0
        for dir in tqdm(os.listdir(data_path)):
            emotion = int(dir.split('_')[2][1])
            number = int(dir.split('_')[1][1:])
            dis = dir.split('_')[3]
            env = dir.split('_')[0]

            is_long = False
            if distance == '70-100':
                is_long = True
            if train_flag and number in type:
                continue
            elif train_flag ==False and number not in type:
                continue
            if env not in environ:
                # print(env)
                continue
            if in_or_out:
                if dis in distance:
                    continue
            if not in_or_out:
                if dis not in distance:
                    continue
            # if dis in distance:
            #     continue
            class_label = emotion
            # files = os.listdir(data_path + dir)
            # files = sorted(files, key=lambda x: int(re.search(r'\d+', x).group()))
            for log in os.listdir(data_path + dir):
                physical_data_path = data_path + dir + '/' + log + '/physical_data/'
                img_path = data_path + dir + '/' + log + '/img/'
                depth_orientation_path = data_path + dir + '/' + log + '/depth_orientation/'
                files = os.listdir(physical_data_path)
                # files.sort()\
                files = sorted(files, key=lambda x: int(re.search(r'\d+', x).group()))
                tof_data =  data_path + dir + '/' + log + '/tof_data.pkl'
                #preload the depth and orientation and the target_data, range
                with open(tof_data, 'rb') as f:
                    tof_data = pickle.load(f)
                self.tof_data.append(tof_data)
                tmp_i = 0
                for file in files:
                    # with open(physical_data_path + file, 'rb') as f:
                    #     physical_data = pickle.load(f)
                    # self.data.append(physical_data)
                    
                    file_path = physical_data_path + file
                    # with open(file_path, 'rb') as f:
                    #     physical_data = pickle.load(f)
                    # orientation, depth = self.get_ori_depth(physical_data[0])
                    tof_frame_index = file_path.split('/')[-1]
                    tof_frame_index = tof_frame_index.split('.')[0]
                    tof_frame_index = int(tof_frame_index[8:])-1
                    if tof_frame_index < 7:
                        continue 
                    tem_flag = True
                    for jj in range(0,4):
                        tmp_physical_data_path = physical_data_path + 'physical'+str(tof_frame_index+1-jj)+'.pkl'
                        if tof_frame_index-jj < 0 :
                            tem_flag = False
                            break
                        tmp_tof = tof_data[tof_frame_index-jj]
                        if tmp_tof[1] is None or tmp_tof[0] is None or tmp_tof[2] is None:
                            tem_flag = False
                            break
                        if not os.path.exists(tmp_physical_data_path):
                            tem_flag = False
                            break
                    # if tem_flag:
                    #     self.indecator.append(number_data)
                    #     tmp_i += 1
                    # if tmp_i > 10:
                    #     break
                    save_img_path = img_path + 'color_image'+str(tof_frame_index+1)+'.npy'
                    save_depth_orientation_path = depth_orientation_path + 'depth_orientation'+str(tof_frame_index+1)+'.pkl'
                    if not os.path.exists(save_depth_orientation_path):
                        continue
                    with open(save_depth_orientation_path, 'rb') as f:
                        depth_orientation = pickle.load(f)
                    orientation = depth_orientation[0]
                    depth = depth_orientation[1]
                    tmp = tof_data[tof_frame_index]
                    if tmp[1] is None or tmp[0] is None or tmp[2] is None:
                        continue
                    with open(file_path, 'rb') as f:
                        physical_data = pickle.load(f)
                    face_orientation = physical_data[1]
                    detection = physical_data[2]
                    detection_tmp= torch.tensor(detection).to(self.device).float().round().int()
                    # detection_tmp = detection
                    min_depth = 10000
                    for h in range(detection_tmp[2],detection_tmp[3]):
                        for w in range(detection_tmp[0],detection_tmp[1]):
                            if h < 0 or h >= 64 or w < 0 or w >= 64:
                                continue
                            if depth[h][w] < min_depth and depth[h][w] >300:
                                min_depth = depth[h][w]
                    if tem_flag:
                        self.indecator.append(number_data)
                        self.data.append([ind, file_path, class_label, orientation, depth,save_img_path, face_orientation,detection,number,is_long, min_depth])
                        number_data += 1
                        tmp_i += 1
                    if tmp_i > 100:
                        break
                    # self.data.append([ind, file_path, class_label, 1,1,save_img_path])
                ind += 1
        # random select 30 samples from the dataset, save the index in a list, the data is a list, so only choose the index
        # for i in range(30):
        #     index = np.random.randint(0, len(self.data))
        #     while index in self.indexs:
        #         index = np.random.randint(0, len(self.data))
        #     self.indexs.append(index)
    def __len__(self):
        # return len(self.data)
        return len(self.indecator)
        # return 30
    def __getitem__(self, idx):
        # get the random index between 0 and len(self.data)
        # idx = np.random.randint(0, len(self.data))
        # idx = self.indexs[idx]
        idx = self.indecator[idx]
        start_time = time.time()
        # print(idx)
        # print(len(self.data))
        file_path = self.data[idx][1]
        # with open(file_path, 'rb') as f:
        #     physical_data = pickle.load(f)
        # orientation, depth = self.get_ori_depth(physical_data[0])
        orientation = self.data[idx][3]
        depth = self.data[idx][4]
        image_path = self.data[idx][5]
        is_long = self.data[idx][9]

        # face_orientation = physical_data[1]
        face_orientation = self.data[idx][6]
        face_orientation = torch.tensor(face_orientation).to(self.device).float()
        orientation = orientation.to(self.device).float()
        depth = depth.to(self.device).float()
        with open(image_path, 'rb') as f:
            img = np.load(f)
        img = np.resize(img, (540, 960, 3))
        tof_frame_index = file_path.split('/')[-1]
        tof_frame_index = tof_frame_index.split('.')[0]
        tof_frame_index = int(tof_frame_index[8:])-1
        tof_data = self.tof_data[self.data[idx][0]][tof_frame_index-3:tof_frame_index+1]
        # print(len(self.tof_data[self.data[idx][0]]))
        # print(tof_frame_index)
        # print(len(tof_data[1]))
        # end_time = time.time()
        # print('load data time1:', end_time-start_time)
        # start_time = time.time()

        # tmp = np.flip(tof_data[1],0)
        # print(len(tof_data))
        input_depth = np.flip(tof_data[-1][0],0)
        input_depth = torch.tensor(input_depth.copy())
        input_depth = input_depth.to(self.device)
        input_depth = input_depth.float()
        input_reflectance = np.flip(tof_data[-1][2],0)
        input_reflectance = torch.tensor(input_reflectance.copy())
        input_reflectance = input_reflectance.to(self.device)
        input_reflectance = input_reflectance.float()
        # tmp = np.flip(tmp,1)

        only_tof = []
        empty = torch.zeros((8,8,10))
        empty = empty.to(self.device).float()
        for i in range(len(tof_data)):
            # print(i)
            # print(tof_data[i][1].shape)
            tmp = np.flip(tof_data[i][1],0)
            tmp = torch.tensor(tmp.copy())
            tmp = tmp.to(self.device).float()
            if is_long:
                tmp = torch.cat((empty, tmp), 2)
            else:
                tmp = torch.cat((tmp, empty), 2)
            

            only_tof.append(tmp)
        # tof_data = torch.tensor(tmp.copy())
        # tof_data = tof_data.to(self.device)
        # tof_data = tof_data.float()
        tof_data = torch.stack(only_tof,0)
        
        
        
        class_label = self.data[idx][2]
        #generate the class label of one hot encoding
        class_label = torch.tensor([class_label]).to(self.device)
        # class_label = torch.zeros(7)
        # class_label[self.data[idx][2]] = 1
        # class_label = class_label.to(self.device).float()
        # detect = torch.zeros(4)
        # mask = (depth != 0).float()
        # indices = torch.nonzero(mask == 1, as_tuple=False)
        # detect[0] = torch.min(indices[:, 1])
        # detect[1] = torch.min(indices[:, 0])
        # detect[2] = torch.max(indices[:, 1])
        # detect[3] = torch.max(indices[:, 0])

        # detect  = detect/15*7
        # end_time = time.time()
        # print('load data time2:', end_time-start_time)
        # start_time = time.time()
        # detection = physical_data[2]
        detection = self.data[idx][7]
        uid = self.data[idx][8]-1
        # print(uid)
        uid = torch.tensor(uid).to(self.device)

        # generate mask based on the detection
        mask = torch.zeros((64,64)).to(self.device)
        #将detection四舍五入为整数
        detection = torch.tensor(detection).to(self.device).float()
        detection_tmp = detection.round().int()

        mask[detection_tmp[2]:detection_tmp[3],detection_tmp[0]:detection_tmp[1]] = 1
        detect = torch.tensor(detection).to(self.device).float()
        # detect = detect[[2,0,3,1]]# xmin,ymin,xmax,ymax
        detect = detect[[0,2,1,3]]
        # if detect[2] -detect[0] < 0.001:
            #detect[2] 向上取整，detect[0]向下取整，这两个都是浮点数现在
        detect = detect/63*7
        detect[2] = torch.ceil(detect[2])
        detect[0] = torch.floor(detect[0])
        detect[3] = torch.ceil(detect[3])
        detect[1] = torch.floor(detect[1])
        # print(detect)
        # detect = detect.permute
        
        target = {
            'boxes': detect.unsqueeze(0),
            'labels': torch.tensor([1], dtype=torch.int64).to(self.device)  # 标签1表示人脸
        }
        # end_time = time.time()
        # print('load data time3:', end_time-start_time)
        # start_time = time.time()
        # print(image_path)
        # print(class_label)
        if self.single:
            tof_data = tof_data[-1]
        min_depth = self.data[idx][-1]
        min_depth = torch.tensor(min_depth).to(self.device).float()

        return orientation,depth, tof_data, class_label, face_orientation, input_depth,input_reflectance,target,torch.tensor(img,dtype=torch.float).to(self.device), mask.float(), uid, min_depth
    
    def get_ori_depth(self, physical_data):
        # print(len(physical_data))
        # print(physical_data)
        orientation = torch.zeros((64,64))
        depth = torch.zeros((64,64))
        for h in range(64):
            for w in range(64):
                if len(physical_data[h][w]) == 0:
                    continue
                n = 0
                all_points = []
                if len(physical_data[h][w]) == 0:
                    continue
                for points in physical_data[h][w]:
                    # print(len(physical_data[h][w]))
                    position = points[0]
                    all_points.append(position)
                    # orientation[h][w] = points[1][0]
                    depth[h][w] += points[0][2]
                    n+=1
                depth[h][w] = depth[h][w]/n
                all_points = np.array(all_points)
                center = all_points.mean(axis=0)
                points_centered = points - center
                cov_matrix = np.cov(points_centered, rowvar=False)
                eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
                normal_vector = eigenvectors[:, np.argmin(eigenvalues)]

                # calculate the angle between the normal vector and the center
                angle = np.arccos(np.abs(np.dot(normal_vector, center) / (np.linalg.norm(normal_vector) * np.linalg.norm(center)+1e-6)))
                orientation[h][w] = angle
        return orientation, depth


def meen_data(data_path):
    dirs = os.listdir(data_path)
    for dir in dirs:
        for log in os.listdir(data_path + dir):
            physical_data_path = data_path + dir + '/' + log + '/physical_data/'
            files = os.listdir(physical_data_path)
            for file in files:
                with open(physical_data_path + file, 'rb') as f:
                    physical_data = pickle.load(f)
                for h in range(16):
                    for w in range(16):
                        for i in range(len(physical_data[h][w])):
                            physical_data[h][w][i][0] = np.mean(physical_data[h][w][i][0], axis = 0)
                with open(physical_data_path + file, 'wb') as f:
                    pickle.dump(physical_data, f)
                




        
if __name__ == '__main__':
    data_path = '/data/share/SPAD_TOF/Data_preprocessing/physycal_model/inves_dataset/dataset_FER/'
    # data_path   = '/home/shared/SPAD_TOF/Data_preprocessing/physycal_model/inves_dataset/dataset_mask_generate/'
    physical_dataprocess(data_path)
    get_depth_orientation_files(data_path)
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
    # for i in range(5):
    #     print(dataset[i][0])
    # print(dataset[0])
