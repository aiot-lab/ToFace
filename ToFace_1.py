import torch
import torch.nn as nn
import torch.optim as optim
from scipy.integrate import quad
import math
import numpy as np
import time
from scipy.interpolate import CubicSpline
import torch.nn.functional as F
from torch.autograd import Function
import pywt
from pytorch_wavelets import DWT1DForward, DWT1DInverse
from torchvision.models import mobilenet_v2
from fvcore.nn import FlopCountAnalysis



class Upsample1(nn.Module):
    def __init__(self):
        super(Upsample1, self).__init__()
        self.conv1 = nn.Conv2d(28, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 560, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(560)

    def forward(self, x):
        #x:batch*18*8*8
        x = F.relu(self.bn1(self.conv1(x)))#batch*64*8*8
        x = F.relu(self.bn2(self.conv2(x)))#batch*128*8*8
        x = self.conv3(x)#batch*360*8*8
        return x
    
def pad_1d_constant(x, pad, value=0):
    left_pad, right_pad = pad
    batch_size = x.shape[0]
    left_pad_val = torch.full((batch_size,left_pad), value, dtype=x.dtype, device=x.device)
    right_pad_val = torch.full((batch_size,right_pad+1), value, dtype=x.dtype, device=x.device)
    return torch.cat([left_pad_val, x, right_pad_val], dim=-1)
def pad_1d_reflect(x, pad):
    left_pad, right_pad = pad
    left_reflect = x[..., :left_pad].flip(-1)
    right_reflect = x[..., -right_pad:].flip(-1)
    return torch.cat([left_reflect, x, right_reflect], dim=-1)
def pad_1d_replicate(x, pad):
    left_pad, right_pad = pad
    left_replicate = x[..., :1].expand(-1, left_pad)
    right_replicate = x[..., -1:].expand(-1, right_pad)
    return torch.cat([left_replicate, x, right_replicate], dim=-1)
def pad_1d_circular(x, pad):
    left_pad, right_pad = pad
    left_circular = x[..., -left_pad-1:]
    right_circular = x[..., :right_pad]
    return torch.cat([left_circular, x, right_circular], dim=-1)

def swt_pytorch(data, wavelet = 'db2', level=1, start_level=0, axis=-1, trim_approx=False, norm=False):
    data = data.permute(0,2,3,1)
    batch_size, height, width, channels = data.shape
    data = data.reshape(batch_size*height*width,channels)
    def pad_1d(x, pad, mode='constant', value=0):
        if mode == 'reflect':
            return pad_1d_reflect(x, pad)
        elif mode == 'constant':
            return pad_1d_constant(x, pad, value)
        elif mode == 'replicate':
            return pad_1d_replicate(x, pad)
        elif mode == 'circular':
            return pad_1d_circular(x, pad)
        
        else:
            raise NotImplementedError(f"Padding mode {mode} is not implemented for 1D data.")
    
    def prep_filt_afb1d(h, device=None):
        t = torch.get_default_dtype()
        h = torch.tensor(h[::-1], device=device, dtype=t).reshape(1, 1, -1)
        return h
    
    if not isinstance(data, torch.Tensor):
        raise ValueError("Input data should be a PyTorch tensor")
    
    if level is None:
        level = pywt.swt_max_level(data.shape[axis])

    wavelet = pywt.Wavelet(wavelet)
    if norm:
        if not wavelet.orthogonal:
            raise ValueError("Wavelet must be orthogonal if norm=True")
        wavelet = pywt.Wavelet(wavelet.name, [c / np.sqrt(2) for c in wavelet.filter_bank])

    axis = axis % data.ndim

    coeffs = []
    original_len = data.shape[axis]
    # print(original_len)
    sqrt2 = np.sqrt(2)

    for i in range(start_level, start_level + level):
        step_size = 2 ** i
        h0, h1 = wavelet.dec_lo, wavelet.dec_hi

        # if norm:
        #     h0 = [val / sqrt2 for val in h0]
        #     h1 = [val / sqrt2 for val in h1]

        h0 = prep_filt_afb1d(h0, device=data.device)
        h1 = prep_filt_afb1d(h1, device=data.device)
        
        pad_size = step_size * (h0.shape[2]) // 2
        # print(pad_size)
        pad_width = (pad_size, pad_size)
        data = pad_1d(data, pad_width, mode='circular')
        
        d = data.unsqueeze(1)
        lo = F.conv1d(d, h0, stride=1,dilation=step_size)
        hi = F.conv1d(d, h1, stride=1,dilation=step_size)
        
        lo = lo.squeeze(1)
        hi = hi.squeeze(1)
        # print(lo)
        lo = lo[..., pad_size:pad_size + original_len]
        hi = hi[..., pad_size:pad_size + original_len]
        
        coeffs.append((lo, hi))
        data = lo

    if trim_approx:
        coeffs = [coeffs[-1][0]] + [d for _, d in reversed(coeffs)]
    if level == 1:
        # print(coeffs)
        coeffs = torch.cat(coeffs[0], dim=-1)
        # print(coeffs.shape)
        coeffs = coeffs.reshape(batch_size,height,width,channels*2)
        # print(coeffs.shape)
        coeffs = coeffs.permute(0,3,1,2)
    return coeffs

class SWTFunction(Function):
    @staticmethod
    def forward(ctx, input, wavelet, level):
        ctx.save_for_backward(input)
        ctx.wavelet = wavelet
        ctx.level = level

        batch_size, channels, height, width = input.shape
        # 输出的通道数将是原通道数的两倍，因为SWT包含近似和细节部分
        output = torch.empty(batch_size, channels * 2, height, width, device=input.device)

        # 处理每个批次
        for b in range(batch_size):
            # 处理每一个8x8区域的信号
            for i in range(height):
                for j in range(width):
                    # 提取每个位置的360通道长度的信号
                    data = input[b, :, i, j]
                    # 在CPU上进行SWT处理，因为PyWavelets不支持GPU操作
                    coeffs = pywt.swt(data.cpu().numpy(), wavelet=wavelet, level=level, trim_approx=True, norm=True)
                    # 只使用一级小波变换的输出（一个近似部分和一个细节部分）
                    # print(coeffs)
                    cA, cD = coeffs
                    # 将结果放回对应的位置
                    output[b, :channels, i, j] = torch.from_numpy(cA).to(input.device)
                    output[b, channels:, i, j] = torch.from_numpy(cD).to(input.device)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        wavelet = ctx.wavelet
        level = ctx.level

        batch_size, channels, height, width = grad_output.shape
        channels //= 2  # 分别为近似和细节部分
        grad_input = torch.zeros_like(input)

        for b in range(batch_size):
            for i in range(height):
                for j in range(width):
                    grad_data = grad_output[b, :, i, j]
                    grad_cA = grad_data[:channels].cpu().numpy()
                    # grad_cA = np.zeros_like(grad_data[:channels])
                    grad_cD = grad_data[channels:].cpu().numpy()
                    reconstructed = pywt.iswt([(grad_cA, grad_cD)], wavelet=wavelet)
                    grad_input[b, :, i, j] = torch.from_numpy(reconstructed).to(input.device)

        return grad_input, None, None

class SWTLayer(nn.Module):
    def __init__(self, wavelet='db2', level=1):
        super(SWTLayer, self).__init__()
        self.wavelet = wavelet
        self.level = level

    def forward(self, x):
        return SWTFunction.apply(x, self.wavelet, self.level)
    
class WaveletConvMLP(nn.Module):
    def __init__(self):
        super(WaveletConvMLP, self).__init__()
        self.swt_layer = SWTLayer()  # SWT 层
        # self.conv1d = nn.Conv1d(360, 90, kernel_size=6)  # 使用90个输出通道
        self.conv2d = nn.Conv2d(1, 4, kernel_size=(2,7), padding=(0,3))
        
        self.fc1 = nn.Linear(1 * 360, 90)
        self.conv2d1 = nn.Conv2d(4, 4, kernel_size=(2,7), padding=(0,3))
        self.fc2 = nn.Linear(90, 16)
        self.conv2d2 = nn.Conv2d(4, 4, kernel_size=(2,7), padding=(0,3))
        self.fc3 = nn.Linear(16, 1)

    def forward(self, x):
        # SWT 变换
        #x:batch*360*8*8
        old_coeffs = self.swt_layer(x)
        coeffs = old_coeffs[:,360:,:,:]#只取细节部分
        coeffs = self.swt_layer(coeffs)

        coeffs = coeffs.reshape(-1, 2,360)  #

        # 一维卷积
        coeffs = self.conv2d(coeffs.unsqueeze(1))  # 为了适配卷积层的输入形状，增加一个维度
        coeffs = F.relu(coeffs)#batch*8*8,4,2,360


        # MLP
        coeffs = coeffs.view(coeffs.size(0),4, -1)
        coeffs = F.relu(self.fc1(coeffs))
        output = F.relu(self.fc2(coeffs))
        output = self.fc3(output)
        output = output.view(-1, 8, 8, 4)  # 最终输出形状

        return output

class DWTLayer(nn.Module):
    def __init__(self, wavelet='db2'):
        super(DWTLayer, self).__init__()
        # 初始化DWT前向变换层，注意只保留第一级的变换
        self.dwt = DWT1DForward(wave=wavelet, J=1)

    def forward(self, x):
        # x 的形状应该是 (batch_size, channels, height, width)
        # 在这里，channels = 360，我们将对每个8x8的block的每个通道进行变换
        batch_size, channels, height, width = x.shape
        # 初始化输出tensor，每个通道变为两倍（近似和细节），宽度减半
        transformed = torch.zeros(batch_size, channels, height, width, device=x.device)

        # 应用DWT
        for i in range(height):
            for j in range(width):
                # 从所有批次中抽取同一位置的数据，形状 [batch_size, channels]
                data = x[:, :, i, j]
                # 在最后一个维度（width）上增加一个维度以满足dwt的输入要求
                # data = data.unsqueeze(-1)
                data = data.unsqueeze(1)
                # print(data.shape)
                Yl, Yh = self.dwt(data)
                # print(Yl.shape)
                # print(Yh[0].shape)
                # Yl, Yh 都是列表，列表中包含不同级别的变换结果
                # 我们只用第一级的结果
                Yl = Yl.squeeze(1)  # 去掉最后一个维度，它是冗余的
                Yh = Yh[0].squeeze(1)  # 只取第一级细节系数，并去掉最后一个维度
                # 将近似和细节系数合并，放入输出tensor
                
                transformed[:, :channels//2, i, j] = Yl[:,:channels//2]
                transformed[:, channels//2:, i, j] = Yh[:,:channels//2]

        return transformed

class RecursiveDWT(nn.Module):
    def __init__(self, wavelet='db1', max_levels=3):
        super(RecursiveDWT, self).__init__()
        self.max_levels = max_levels
        self.wavelet = wavelet
        self.dwt_forward = DWT1DForward(wave=wavelet, J=1)

    def forward(self, x):
        batch_size, channels, height, width = x.shape
        x = x.reshape(batch_size*height*width,-1)
        x = x.unsqueeze(1)
        results = self._recursive_dwt(x, 0, batch_size, channels, height, width)
        return results

    def _recursive_dwt(self, x, level, batch_size, channels, height, width):
        if level >= self.max_levels:
            return [x.reshape(batch_size,-1,height,width)]  # 最终层级返回当前结果
        # print(x.shape)
        
        Yl, Yh = self.dwt_forward(x)
        
        # 递归处理近似系数和细节系数
        approx_results = self._recursive_dwt(Yl, level + 1, batch_size, channels, height, width)
        detail_results = self._recursive_dwt(Yh[0], level + 1, batch_size, channels, height, width)
        return approx_results + detail_results

class Dwt_CnnMLP(nn.Module):
    def __init__(self, num_outputs):
        super(Dwt_CnnMLP, self).__init__()
        self.dwt_layer = DWTLayer()  # DWT 层
        self.conv2d = nn.Conv2d(1, 4, (1, 3))  # 使用特殊的卷积核大小和组卷积
        self.fc1 = nn.Linear(64, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 8)
        self.fc4 = nn.Linear(8, num_outputs)
        self.num_outputs = num_outputs
        self.multi_dwt = RecursiveDWT(max_levels=5)

    def forward(self, x):
        # x 的形状是 (batch_size, 8, 8, 360)
        x = x.permute(0, 3, 1, 2)  # 改变维度为 (batch_size, 360, 8, 8)
        x = self.dwt_layer(x)
        cd1 = x[:, 180:, :, :]
        cd1 = self.dwt_layer(cd1)
        cda=cd1[:, :90, :,:]
        cd = self.multi_dwt(cda)

        ca1 = x[:, :180, :, :]
        ca1 = self.dwt_layer(ca1)
        cad = ca1[:, 90:, :, :]
        # c = torch.cat((cda,cad),1)
        ca = self.multi_dwt(cad)
        _,n,_,_ = cd[0].shape
        # print(ca[0].shape)
        length = len(cd)
        cd = torch.cat(cd,1)
        # print(cd.shape)
        cd = cd.view(-1,length,n)
        # print(cd.shape)
        ca = torch.cat(ca,1)
        ca = ca.view(-1,length,n)
        x = torch.cat((cd,ca),1)
        x = torch.unsqueeze(x,1)
        # print(x.shape)
        x = self.conv2d(x)#batch*8*8,4,198,1
        # print(x.shape)
        x = F.relu(x)
        x = x.view(x.size(0), 4, -1)
        # print(x.shape)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        # x = self.fc4(x)

        x = x.view(-1, 4,8, 8, 8)  # 改变维度为 (batch_size, 4, 8, 8)
        return x

class Intense_branch(nn.Module):
    def __init__(self):
        super(Intense_branch, self).__init__()
        self.conv1 = nn.Conv2d(1, 4, kernel_size=(1,7), padding=(0,3))
        self.bn1 = nn.BatchNorm2d(4)
        self.fc1 = nn.Linear(360, 90)
        self.fc2 = nn.Linear(90, 32)
        self.fc3 = nn.Linear(32, 16)
        self.fc4 = nn.Linear(16, 8)



    def forward(self, x):
        #x:batch*360*8*8
        batch,channel,height,width = x.shape
        # x = x.unsqueeze(1)
        x = x.reshape(batch*height*width,channel)
        x = x.unsqueeze(1)
        x = x.unsqueeze(1)
        # print(x.shape)
        x = F.relu(self.bn1(self.conv1(x)))#batch*8*8,4,1,360
        x = x.view(batch,height,width,4,360)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        return x



class ToFace(nn.Module):
    def __init__(self):
        super(ToFace, self).__init__()
        self.dwt_cnn_mlp = Dwt_CnnMLP(1)
        self.upsample = Upsample1()
        self.swt = WaveletConvMLP()
        self.intense = Intense_branch()
        self.depth_fc = nn.Linear(4, 360)
        self.orientation_fc = nn.Linear(8, 4)
        self.orientation_fc2 = nn.Linear(4, 2)
        self.orientation_fc3 = nn.Linear(2, 1)


    def forward(self, x):
        
        x = self.upsample(x)
        upsampled = x
        x = self.swt(x)
        depth = x
        # print(x.shape)
        
        # position = x
        # position -=26.25
        # position /= 3.75
        batch,height,width,channel = x.shape
        tmp = x.view(-1,4)
        tmp = tmp**4
        position = self.depth_fc(tmp)
        position = position.view(batch,height,width,-1)
        position = position.permute(0,3,1,2)
        # print(position.shape)
        # print(upsampled.shape)
        depth_upsampled = position*upsampled
        # print(depth_upsampled.shape)
        intense = self.intense(depth_upsampled)
        depth_upsampled = depth_upsampled.permute(0,2,3,1)
        peak_width = self.dwt_cnn_mlp(depth_upsampled)
        peak_width = peak_width.permute(0,3,4,1,2)
        # print(intense.shape)
        # print(peak_width.shape)
        # orientation = torch.cat((intense,peak_width),-1)
        orientation = intense+peak_width
        orientation = F.relu(self.orientation_fc(orientation))
        orientation = F.relu(self.orientation_fc2(orientation))
        orientation = self.orientation_fc3(orientation)

        orientation = orientation.squeeze(-1)
        # print(orientation.shape)
        # print(depth.shape)
        # print(upsampled.shape)

        
        return orientation, depth, upsampled
    



class MobileNetBranch(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MobileNetBranch, self).__init__()
        mobilenet = mobilenet_v2(pretrained=False)
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            mobilenet.features[1:7]  # Adding more layers by taking more features
        )
        self.middle = mobilenet.features[7:14]
        self.classifier = nn.Sequential(
            nn.Conv2d(96, out_channels, kernel_size=1),
            nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)  # Upsample to match 8x8 size
        )

    def forward(self, x):
        x = self.features(x)
        x = self.middle(x)
        x = self.classifier(x)
        return x

class Mobilenet_baseline(nn.Module):
    def __init__(self):
        super(Mobilenet_baseline, self).__init__()
        self.branch1 = MobileNetBranch(28, 16)
        self.branch2 = MobileNetBranch(28, 16)

    def forward(self, x):
        out1 = self.branch1(x)
        out2 = self.branch2(x)
        return out1, out2


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class UNetBranch(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNetBranch, self).__init__()
        self.encoder1 = DoubleConv(in_channels, 64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.middle = DoubleConv(128, 256)

        self.upconv1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.decoder1 = DoubleConv(256, 128)
        self.upconv2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.decoder2 = DoubleConv(128, 64)

        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        e1 = self.encoder1(x)
        p1 = self.pool1(e1)
        e2 = self.encoder2(p1)
        p2 = self.pool2(e2)

        m = self.middle(p2)

        u1 = self.upconv1(m)
        d1 = self.decoder1(torch.cat([u1, e2], dim=1))
        u2 = self.upconv2(d1)
        d2 = self.decoder2(torch.cat([u2, e1], dim=1))

        out = self.final_conv(d2)
        return out

class Unet_baseline(nn.Module):
    def __init__(self):
        super(Unet_baseline, self).__init__()
        self.branch1 = UNetBranch(28, 16)
        self.branch2 = UNetBranch(28, 16)

    def forward(self, x):
        out1 = self.branch1(x)
        out2 = self.branch2(x)
        return out1, out2





class Encoder(nn.Module):
    def __init__(self, in_channels):
        super(Encoder, self).__init__()
        self.encoder1 = DoubleConv(in_channels, 64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.middle = DoubleConv(128, 256)

    def forward(self, x):
        e1 = self.encoder1(x)
        p1 = self.pool1(e1)
        e2 = self.encoder2(p1)
        p2 = self.pool2(e2)

        m = self.middle(p2)
        return m, e1, e2

class Decoder(nn.Module):
    def __init__(self, out_channels):
        super(Decoder, self).__init__()
        self.upconv1 = nn.ConvTranspose2d(256*2, 128, kernel_size=2, stride=2)
        self.decoder1 = DoubleConv(256, 128)
        self.upconv2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.decoder2 = DoubleConv(128, 64)

        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, m, e1, e2):
        u1 = self.upconv1(m)
        d1 = self.decoder1(torch.cat([u1, e2], dim=1))
        u2 = self.upconv2(d1)
        d2 = self.decoder2(torch.cat([u2, e1], dim=1))

        out = self.final_conv(d2)
        return out

class DualUNet(nn.Module):
    def __init__(self):
        super(DualUNet, self).__init__()
        self.encoder = Encoder(18)
        self.branch1 = Decoder(4)
        self.branch2 = Decoder(4)

    def forward(self, x):
        encoded, e1, e2 = self.encoder(x)
        out1 = self.branch1(encoded, e1, e2)
        out2 = self.branch2(encoded, e1, e2)
        return out1, out2

class ClassificationNetwork(nn.Module):
    def __init__(self, num_classes, input_size=8):
        super(ClassificationNetwork, self).__init__()
        # in_channels = input_size/8
        self.conv = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        # self.fc1 = nn.Sequential(
        #     nn.Linear(64 * input_size * input_size, 768),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(768, num_classes-3)
        #     # 
        # )
        self.fc1 = nn.Linear(64 * input_size * input_size, 768)
            # nn.ReLU(inplace=True),
        self.fc11= nn.Linear(768, num_classes-3)
            # 
        
        self.fc2 = nn.Sequential(
            nn.Linear(64 * input_size * input_size, 768),
            nn.ReLU(inplace=True),
            nn.Linear(768, 3)
        )
        self.fc3 = nn.Sequential(
            nn.Linear(64 * input_size * input_size, 768),
            nn.ReLU(inplace=True),
            nn.Linear(768, 13)
        )

    def forward(self, x):
        # x = torch.cat([out1, out2], dim=1)
        x = self.conv(x)
        x = x.view(x.size(0), -1)  # Flatten
        x1 = self.fc1(x)
        # add relu
        x11 = nn.functional.relu(x1)
        x1 = self.fc11(x11)
        x2 = self.fc2(x)
        x3 = self.fc3(x)
        return x1, x2,x3, x11
    
class ClassificationNetwork_mobilenet(nn.Module):
    def __init__(self, num_classes, input_size=8):
        super(ClassificationNetwork_mobilenet, self).__init__()
        # in_channels = input_size/8
        self.conv = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        mobilenet = mobilenet_v2(pretrained=True)
        self.features = mobilenet.features
        self.features[0][0] = nn.Conv2d(2, 32, kernel_size=3, stride=2, padding=1, bias=False)

        self.fc1 = nn.Sequential(
            nn.Linear(mobilenet.last_channel, 768),
            nn.ReLU(inplace=True),
            nn.Linear(768, num_classes-3)
            # 
        )
        self.fc2 = nn.Sequential(
            nn.Linear(mobilenet.last_channel, 768),
            nn.ReLU(inplace=True),
            nn.Linear(768, 3)
        )

    def forward(self, x):
        # x = torch.cat([out1, out2], dim=1)
        x = self.features(x)
        # x = x.view(x.size(0), -1)  # Flatten
        # print(x.shape)
        x = nn.functional.adaptive_avg_pool2d(x, 1).reshape(x.shape[0], -1)
        # print(x.shape)
        x1 = self.fc1(x)
        x2 = self.fc2(x)
        return x1, x2


class ToFace_Encoder(nn.Module):
    def __init__(self, in_channels):
        super(ToFace_Encoder, self).__init__()
        self.encoder1 = DoubleConv(in_channels, 64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.middle = DoubleConv(128, 256)

    def forward(self, x):
        e1 = self.encoder1(x)
        # p1 = self.pool1(e1)
        e2 = self.encoder2(e1)
        # p2 = self.pool2(e2)

        m = self.middle(e2)
        return m, e1, e2

class Toface_Decoder(nn.Module):
    def __init__(self, out_channels, scale_factor=2):
        super(Toface_Decoder, self).__init__()
        self.upconv1 = nn.Conv2d(int(256*scale_factor), 128, kernel_size=3, padding=1)
        self.decoder1 = DoubleConv(256, 128)
        self.upconv2 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.decoder2 = DoubleConv(128, 64)

        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, m, e1, e2):
        u1 = self.upconv1(m)
        d1 = self.decoder1(torch.cat([u1, e2], dim=1))
        u2 = self.upconv2(d1)
        d2 = self.decoder2(torch.cat([u2, e1], dim=1))

        out = self.final_conv(d2)
        return out
    

class ToFace_Unet_swt(nn.Module):
    def __init__(self):
        super(ToFace_Unet_swt, self).__init__()
        # self.swt_layer = SWTLayer()  # SWT 层
        # self.swt_layer = swt_pytorch()
        # self.conv1d = nn.Conv1d(360, 90, kernel_size=6)  # 使用90个输出通道
        self.conv2d = nn.Conv2d(1, 1, kernel_size=(2,7), padding=(0,3))
        
        self.fc1 = nn.Linear(1 * 560, 256)
        self.fc2 = nn.Linear(256, 256)
        # self.conv2d1 = nn.Conv2d(4, 4, kernel_size=(2,7), padding=(0,3))
        # self.fc2 = nn.Linear(90, 16)
        # self.conv2d2 = nn.Conv2d(4, 4, kernel_size=(2,7), padding=(0,3))
        # self.fc3 = nn.Linear(16, 1)

    def forward(self, x):
        # SWT 变换
        #x:batch*360*8*8
        old_coeffs = swt_pytorch(x)
        coeffs = old_coeffs[:,560:,:,:]#只取细节部分
        # print(coeffs.shape)
        coeffs = swt_pytorch(coeffs)

        coeffs = coeffs.reshape(-1, 2,560)  #

        # 一维卷积
        coeffs = self.conv2d(coeffs.unsqueeze(1))  # 为了适配卷积层的输入形状，增加一个维度
        coeffs = F.relu(coeffs)#batch*8*8,1,1,360


        # MLP
        coeffs = coeffs.view(coeffs.size(0),1, -1)
        coeffs = F.relu(self.fc1(coeffs))
        output = self.fc2(coeffs)
        output = output.view(-1, 256,8, 8)  # 最终输出形状
        

        return output

class ToFace_Unet_Dwt(nn.Module):
    def __init__(self):
        super(ToFace_Unet_Dwt, self).__init__()
        self.dwt_layer = DWTLayer()  # DWT 层
        self.conv2d = nn.Conv2d(1, 1, (1, 3))  # 使用特殊的卷积核大小和组卷积
        self.fc1 = nn.Linear(64, 128)
        # self.fc2 = nn.Linear(128, 256)
        
        
        self.multi_dwt = RecursiveDWT(max_levels=5)

    def forward(self, x):
        # x 的形状是 (batch_size, 8, 8, 360)
        # x = x.permute(0, 3, 1, 2)  # 改变维度为 (batch_size, 360, 8, 8)
        x = self.dwt_layer(x)
        cd1 = x[:, 180:, :, :]
        cd1 = self.dwt_layer(cd1)
        cda=cd1[:, :90, :,:]
        cd = self.multi_dwt(cda)

        ca1 = x[:, :180, :, :]
        ca1 = self.dwt_layer(ca1)
        cad = ca1[:, 90:, :, :]
        # c = torch.cat((cda,cad),1)
        ca = self.multi_dwt(cad)
        _,n,_,_ = cd[0].shape
        # print(ca[0].shape)
        length = len(cd)
        cd = torch.cat(cd,1)
        # print(cd.shape)
        cd = cd.view(-1,length,n)
        # print(cd.shape)
        ca = torch.cat(ca,1)
        ca = ca.view(-1,length,n)
        x = torch.cat((cd,ca),1)
        x = torch.unsqueeze(x,1)
        # print(x.shape)
        x = self.conv2d(x)#batch*8*8,4,198,1
        # print(x.shape)
        x = F.relu(x)
        x = x.view(x.size(0), 1, -1)
        # print(x.shape)
        x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        x = x.view(-1, 128,8, 8)  # 改变维度为 (batch_size, 4, 8, 8)
        # x = self.fc4(x)

        # x = x.view(-1, 4,8, 8, 8)  
        return x


class ToFace_Unet_Dwt_locate(nn.Module):
    def __init__(self):
        super(ToFace_Unet_Dwt_locate, self).__init__()
        self.dwt_layer = DWTLayer()  # DWT 层
        self.conv2d = nn.Conv2d(1, 1, kernel_size=(2,7), padding=(0,3))
        self.fc1 = nn.Linear(90, 128)
        self.fc2 = nn.Linear(128, 256)
        

    def forward(self, x):
        # x 的形状是 (batch_size, 8, 8, 360)
        # x = x.permute(0, 3, 1, 2)  # 改变维度为 (batch_size, 360, 8, 8)
        x = self.dwt_layer(x)
        cd1 = x[:, 180:, :, :]
        cd1 = self.dwt_layer(cd1)
        cd1 = cd1.view(-1, 2,90)  #
        cd1 = self.conv2d(cd1.unsqueeze(1))  # 为了适配卷积层的输入形状，增加一个维度
        cd1 = F.relu(cd1)
        cd1 = cd1.view(cd1.size(0),1, -1)
        cd1 = F.relu(self.fc1(cd1))
        output = self.fc2(cd1)
        output = output.view(-1, 256,8, 8)
        
        return output



class ToFace_Unet_Intense(nn.Module):
    def __init__(self):
        super(ToFace_Unet_Intense, self).__init__()
        self.conv1 = nn.Conv2d(1, 1, kernel_size=(1,7), padding=(0,3))
        # self.bn1 = nn.BatchNorm2d(4)
        self.fc1 = nn.Linear(560, 256)
        # self.fc2 = nn.Linear(90, 32)
        # self.fc3 = nn.Linear(32, 16)
        # self.fc4 = nn.Linear(16, 8)



    def forward(self, x):
        #x:batch*360*8*8
        batch,channel,height,width = x.shape
        # x = x.unsqueeze(1)
        x = x.permute(0,2,3,1)
        x = x.reshape(batch*height*width,channel)
        x = x.unsqueeze(1)
        x = x.unsqueeze(1)
        # print(x.shape)
        x = F.relu(self.conv1(x))#batch*8*8,1,1,360
        x = x.view(batch,height,width,1,560)
        x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        # x = F.relu(self.fc3(x))
        # x = F.relu(self.fc4(x))
        return x

class FER_Classifier(nn.Module):
    def __init__(self,in_channels=768):
        super(FER_Classifier, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 256, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(256, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(32, 16)
        self.fc2 = nn.Linear(16, 7)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2)
        # print(x.shape)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class ToFace_Unet(nn.Module):
    def __init__(self):
        super(ToFace_Unet, self).__init__()
        self.upsample = Upsample1()
        self.encoder = ToFace_Encoder(18)
        self.branch1 = Toface_Decoder(4,1)
        self.branch2 = Toface_Decoder(4)
        self.swt = ToFace_Unet_swt()
        # self.swt = ToFace_Unet_Dwt_locate()
        self.dwt = ToFace_Unet_Dwt()
        self.intense = ToFace_Unet_Intense()
        self.depth_fc = nn.Linear(4, 360)
        self.classifier = FER_Classifier()


    def forward(self, x):
        upsample = self.upsample(x)
        swt = self.swt(upsample)
        width = self.dwt(upsample)
        encoded, e1, e2 = self.encoder(x)
        # print(swt.shape)
        # print(encoded.shape)
        # depth_encoded = torch.cat([swt, encoded], dim=1)
        depth_encoded = swt+encoded
        depth = self.branch1(depth_encoded, e1, e2)
        # print(depth_encoded.shape)
        # print   (depth.shape)
        tmp = depth.view(-1,4)
        position = self.depth_fc(tmp)
        position = position.view(-1,360,8,8)
        position_up = position*upsample
        intense = self.intense(position_up)
        intense = intense.squeeze(-2)
        intense = intense.permute(0,3,1,2)
        
        # print(intense.shape)
        # print(width.shape)
        # print(encoded.shape)
        width_encoded = torch.cat([width, encoded, intense], dim=1)
        orientation = self.branch2(width_encoded, e1, e2)
        classifier_encoded = torch.cat([encoded, intense,swt,width], dim=1)
        # print(classifier_encoded.shape)
        classifier_output = self.classifier(classifier_encoded)
        return orientation, depth, upsample, classifier_output
    

class ToFace_Unet_Peak_cnn(nn.Module):
    def __init__(self):
        super(ToFace_Unet_Peak_cnn, self).__init__()
        self.conv1 = nn.Conv2d(1, 1, kernel_size=(1,7), padding=(0,3))
        self.conv2 = nn.Conv2d(1, 1, kernel_size=(1,7), padding=(0,3))
        # self.bn1 = nn.BatchNorm2d(4)
        self.fc1 = nn.Linear(560, 280)
        self.fc2 = nn.Linear(280, 128)
        # self.fc2 = nn.Linear(90, 32)
        # self.fc3 = nn.Linear(32, 16)
        # self.fc4 = nn.Linear(16, 8)



    def forward(self, x):
        #x:batch*360*8*8
        batch,channel,height,width = x.shape
        # x = x.unsqueeze(1)
        x = x.permute(0,2,3,1)
        x = x.reshape(batch*height*width,channel)
        x = x.unsqueeze(1)
        x = x.unsqueeze(1)
        # print(x.shape)
        x = F.relu(self.conv1(x))#batch*8*8,1,1,360
        x = F.relu(self.conv2(x))
        x = x.view(batch,height,width,1,560)
        x = F.relu(self.fc1(x))
        
        x = F.relu(self.fc2(x))
        
        # x = F.relu(self.fc2(x))
        # x = F.relu(self.fc3(x))
        # x = F.relu(self.fc4(x))
        return x
class ToFace_Unet_cnn(nn.Module):
    def __init__(self, out_size):
        super(ToFace_Unet_cnn, self).__init__()
        self.upsample = Upsample1()
        self.encoder = ToFace_Encoder(28)
        out_channels = out_size/8 #eg:16*16 output should has 4 output channels
        self.out_size = out_size
        
        out_channels = int(out_channels**2)
        self.out_channels = out_channels
        # print(out_channels)
        self.branch1 = Toface_Decoder(out_channels,1)
        self.branch2 = Toface_Decoder(out_channels,2)
        self.swt = ToFace_Unet_swt()
        # self.swt = ToFace_Unet_Dwt_locate()
        # self.dwt = ToFace_Unet_Dwt()
        self.peak = ToFace_Unet_Intense()
        # self.peak = ToFace_Unet_Peak_cnn()
        self.depth_fc = nn.Linear(560, 560)
        self.classifier = FER_Classifier(768)


    def forward(self, x):
        start_time = time.time()
        upsample = self.upsample(x)
        swt = self.swt(upsample)
        print('swt:',time.time()-start_time)
        start_time = time.time()
        # width = self.dwt(upsample)
        encoded, e1, e2 = self.encoder(x)
        # print(swt.shape)
        # print(encoded.shape)
        depth_encoded = swt+encoded
        # peak = self.peak(upsample)
        # peak = peak.permute(0,3,1,2)
        # depth_encoded = torch.cat([encoded,peak], dim=1)
        depth = self.branch1(depth_encoded, e1, e2)
        # print(depth_encoded.shape)
        # print   (depth.shape)
        depth = depth.permute(0,2,3,1)
        tmp = depth.reshape(-1,self.out_channels)
        tmp = (tmp-262.5)/1.875
        indicators = torch.zeros(tmp.shape[0],560,device=tmp.device,dtype=torch.float32)
        tmp = tmp.long().clamp(0, 560 - 1)
        # for i in range(tmp.shape[0]):
        #     indicators[i,tmp[i]] = 1
        indicators.scatter_(1,tmp,1)
        print('depth:',time.time()-start_time)
        position = self.depth_fc(indicators)
        position = position.view(-1,8,8,560)
        position = position.permute(0,3,1,2)
        position_up = position*upsample
        peak = self.peak(position_up)
        # print(peak.shape)
        peak = peak.squeeze(-2)
        peak = peak.permute(0,3,1,2)
        
        
        # print(intense.shape)
        # print(width.shape)
        # print(encoded.shape)
        width_encoded = torch.cat([peak, encoded], dim=1)
        orientation = self.branch2(width_encoded, e1, e2)
        classifier_encoded = torch.cat([encoded, peak,swt], dim=1)
        # print(classifier_encoded.shape)
        classifier_output = self.classifier(classifier_encoded)
        return orientation, depth, upsample, classifier_output


class ConvBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, stride=1, dilation=1):
        super(ConvBlock3D, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        return x

class UNet3D(nn.Module):
    def __init__(self):
        super(UNet3D, self).__init__()

        # Encoder
        self.enc1 = ConvBlock3D(1, 16, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.enc2 = ConvBlock3D(16, 32, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.enc3 = ConvBlock3D(32, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool = nn.MaxPool3d(kernel_size=(1, 1, 2), stride=(1, 1, 2))

        # Bottleneck
        self.bottleneck = ConvBlock3D(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1))

        # Decoder with dilated convolutions
        self.upconv3 = nn.ConvTranspose3d(128, 64, kernel_size=(1, 1, 2), stride=(1, 1, 2))
        self.dec3 = ConvBlock3D(128, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1), dilation=1)
        self.upconv2 = nn.ConvTranspose3d(64, 32, kernel_size=(1, 1, 2), stride=(1, 1, 2))
        self.dec2 = ConvBlock3D(64, 32, kernel_size=(3, 3, 3), padding=(1, 1, 1), dilation=1)
        self.upconv1 = nn.ConvTranspose3d(32, 16, kernel_size=(1, 1, 2), stride=(1, 1, 2))
        self.dec1 = ConvBlock3D(32, 16, kernel_size=(3, 3, 3), padding=(1, 1, 1), dilation=1)

        # Output layer
        self.conv_out = nn.Conv3d(16, 20, kernel_size=1)  # 输出通道为20，匹配最终输出尺寸

    def forward(self, x):
        # Encoder
        batch,channel,height,width,d = x.shape
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))
        pooled = self.pool(enc3)
        # print(enc1.shape)
        # print(enc2.shape)
        # print(enc3.shape)
        e1 = enc1.permute(0,2,3,4,1).reshape(batch,height,width,-1)
        e2 = enc2.permute(0,2,3,4,1).reshape(batch,height,width,-1)
        e3 = enc3.permute(0,2,3,4,1).reshape(batch,height,width,-1)
        # Bottleneck
        bottleneck = self.bottleneck(pooled)

        # Decoder
        up3 = self.upconv3(bottleneck)
        up3 = F.pad(up3, [0, enc3.size(4) - up3.size(4), 0, enc3.size(3) - up3.size(3), 0, enc3.size(2) - up3.size(2)])
        dec3 = self.dec3(torch.cat((up3, enc3), dim=1))
        
        up2 = self.upconv2(dec3)
        up2 = F.pad(up2, [0, enc2.size(4) - up2.size(4), 0, enc2.size(3) - up2.size(3), 0, enc2.size(2) - up2.size(2)])
        dec2 = self.dec2(torch.cat((up2, enc2), dim=1))
        
        up1 = self.upconv1(dec2)
        up1 = F.pad(up1, [0, enc1.size(4) - up1.size(4), 0, enc1.size(3) - up1.size(3), 0, enc1.size(2) - up1.size(2)])
        dec1 = self.dec1(torch.cat((up1, enc1), dim=1))

        # Output
        out = self.conv_out(dec1)
        out = out.permute(0,2,3,4,1).reshape(batch,height,width,-1)
        return out,e1,e2,e3
    
class Toface_Decoder_final(nn.Module):
    def __init__(self, out_channels):
        super(Toface_Decoder_final, self).__init__()
        self.upconv1 = nn.Conv2d(int(256), 448, kernel_size=3, padding=1)
        self.decoder1 = DoubleConv(448*2, 448)
        self.upconv2 = nn.Conv2d(448, 448, kernel_size=3, padding=1)
        self.decoder2 = DoubleConv(448*2, 448)
        self.upconv3 = nn.Conv2d(448, 448, kernel_size=3, padding=1)
        self.decoder3 = DoubleConv(448*2, 448)

        self.final_conv = nn.Conv2d(448, out_channels, kernel_size=1)

    def forward(self, m, e1, e2,e3):
        u1 = self.upconv1(m)
        d1 = self.decoder1(torch.cat([u1, e3], dim=1))
        u2 = self.upconv2(d1)
        d2 = self.decoder2(torch.cat([u2, e2], dim=1))
        u3 = self.upconv3(d2)
        d3 = self.decoder3(torch.cat([u3, e1], dim=1))

        out = self.final_conv(d3)
        return out


# class ToFace_Unet_3Dcnn(nn.Module):
#     def __init__(self, out_size):
#         super(ToFace_Unet_3Dcnn, self).__init__()
#         # self.upsample = Upsample1()
#         # self.encoder = ToFace_Encoder(28)
#         self.up = UNet3D()
#         out_channels = out_size/8 #eg:16*16 output should has 4 output channels
#         self.out_size = out_size
        
#         out_channels = int(out_channels**2)
#         self.out_channels = out_channels
#         # print(out_channels)
#         self.branch1 = Toface_Decoder_final(out_channels)
#         self.branch2 = Toface_Decoder_final(out_channels)
#         self.swt = ToFace_Unet_swt()
#         # self.swt = ToFace_Unet_Dwt_locate()
#         # self.dwt = ToFace_Unet_Dwt()
#         self.peak = ToFace_Unet_Intense()
#         # self.peak = ToFace_Unet_Peak_cnn()
#         self.depth_fc = nn.Linear(560, 560)
#         # self.classifier = FER_Classifier(640)


#     def forward(self, x):
#         start_time = time.time()
#         # upsample = self.upsample(x)
#         upsample,e1,e2,e3 = self.up(x)
#         upsample = upsample.permute(0,3,1,2)
#         e1 = e1.permute(0,3,1,2)
#         e2 = e2.permute(0,3,1,2)
#         e3 = e3.permute(0,3,1,2)
#         swt = self.swt(upsample)
#         print('swt:',time.time()-start_time)
#         start_time = time.time()
#         # width = self.dwt(upsample)
#         # encoded, e1, e2 = self.encoder(x)
#         # print(swt.shape)
#         # print(encoded.shape)
#         # print(swt.shape)
#         # print(e3.shape)
#         depth_encoded = torch.cat([swt, e3], dim=1)
#         # print   (depth_encoded.shape)   
#         # peak = self.peak(upsample)
#         # peak = peak.permute(0,3,1,2)
#         # depth_encoded = torch.cat([encoded,peak], dim=1)
#         depth = self.branch1(depth_encoded, e1, e2)
#         # print(depth_encoded.shape)
#         # print   (depth.shape)
#         depth = depth.permute(0,2,3,1)
#         tmp = depth.reshape(-1,self.out_channels)
#         tmp = (tmp-262.5)/1.875
#         indicators = torch.zeros(tmp.shape[0],560,device=tmp.device,dtype=torch.float32)
#         tmp = tmp.long().clamp(0, 560 - 1)
#         # for i in range(tmp.shape[0]):
#         #     indicators[i,tmp[i]] = 1
#         indicators.scatter_(1,tmp,1)
#         print('depth:',time.time()-start_time)
#         position = self.depth_fc(indicators)
#         position = position.view(-1,8,8,560)
#         position = position.permute(0,3,1,2)
#         position_up = position*upsample
#         peak = self.peak(position_up)
#         # print(peak.shape)
#         peak = peak.squeeze(-2)
#         peak = peak.permute(0,3,1,2)
        
        
#         # print(intense.shape)
#         # print(width.shape)
#         # print(encoded.shape)
#         width_encoded = torch.cat([peak, e3], dim=1)
#         orientation = self.branch2(width_encoded, e1, e2)

#         # classifier_encoded = torch.cat([encoded, peak,swt], dim=1)
#         # print(classifier_encoded.shape)
#         # classifier_output = self.classifier(classifier_encoded)
#         return orientation, depth, upsample
    
class ToFace_Unet_3Dcnn(nn.Module):
    def __init__(self, out_size):
        super(ToFace_Unet_3Dcnn, self).__init__()
        # self.upsample = Upsample1()
        # self.encoder = ToFace_Encoder(28)
        self.up = UNet3D()
        out_channels = out_size/8 #eg:16*16 output should has 4 output channels
        self.out_size = out_size
        
        out_channels = int(out_channels**2)
        self.out_channels = out_channels
        # print(out_channels)
        self.branch1 = Toface_Decoder_final(out_channels)
        self.branch2 = Toface_Decoder_final(out_channels)
        self.swt = ToFace_Unet_swt()
        # self.swt = ToFace_Unet_Dwt_locate()
        # self.dwt = ToFace_Unet_Dwt()
        self.peak = ToFace_Unet_Intense()
        # self.peak = ToFace_Unet_Peak_cnn()
        self.depth_fc = nn.Linear(560, 560)
        self.classifier = FER_Classifier(448)


    def forward(self, x):
        start_time = time.time()
        # upsample = self.upsample(x)
        upsample,e1,e2,e3 = self.up(x)
        upsample = upsample.permute(0,3,1,2)
        e1 = e1.permute(0,3,1,2)
        e2 = e2.permute(0,3,1,2)
        e3 = e3.permute(0,3,1,2)
        swt = self.swt(upsample)
        # print('swt:',time.time()-start_time)
        start_time = time.time()
        # width = self.dwt(upsample)
        # encoded, e1, e2 = self.encoder(x)
        # print(swt.shape)
        # print(encoded.shape)
        # print(swt.shape)
        # print(e3.shape)
        # depth_encoded = torch.cat([swt, e3], dim=1)
        depth_encoded = swt
        # print   (depth_encoded.shape)   
        # peak = self.peak(upsample)
        # peak = peak.permute(0,3,1,2)
        # depth_encoded = torch.cat([encoded,peak], dim=1)
        depth = self.branch1(depth_encoded, e1, e2,e3)
        # print(depth_encoded.shape)
        # print   (depth.shape)
        depth = depth.permute(0,2,3,1)
        tmp = depth.reshape(-1,self.out_channels)
        tmp = (tmp-262.5)/1.875
        indicators = torch.zeros(tmp.shape[0],560,device=tmp.device,dtype=torch.float32)
        tmp = tmp.long().clamp(0, 560 - 1)
        # for i in range(tmp.shape[0]):
        #     indicators[i,tmp[i]] = 1
        indicators.scatter_(1,tmp,1)
        # print('depth:',time.time()-start_time)
        position = self.depth_fc(indicators)
        position = position.view(-1,8,8,560)
        position = position.permute(0,3,1,2)
        position_up = position*upsample
        peak = self.peak(position_up)
        # print(peak.shape)
        peak = peak.squeeze(-2)
        peak = peak.permute(0,3,1,2)
        
        
        # print(intense.shape)
        # print(width.shape)
        # print(encoded.shape)
        # width_encoded = torch.cat([peak, e3], dim=1)
        width_encoded = peak
        orientation = self.branch2(width_encoded, e1, e2,e3)

        # classifier_encoded = torch.cat([e3, peak,swt], dim=1)
        classifier_encoded = e3
        # print(classifier_encoded.shape)
        classifier_output = self.classifier(classifier_encoded)
        return orientation, depth, upsample,classifier_output


class ConvBlock3D_complex(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, stride=1, dilation=1,residual = False):
        super(ConvBlock3D_complex, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)
        self.residual = residual

    def forward(self, x):
        residual = x if self.residual else 0
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = x + residual
        return x

class UNet3D_complex(nn.Module):
    def __init__(self):
        super(UNet3D_complex, self).__init__()

        # Encoder
        self.enc0_1 = ConvBlock3D_complex(4, 8, kernel_size=(3, 3, 3), padding=(1, 1, 1))

        self.enc1 = ConvBlock3D_complex(8, 16, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.enc1_1 = ConvBlock3D_complex(16, 16, kernel_size=(3, 3, 3), padding=(1, 1, 1),residual=True)
        self.enc1_2 = ConvBlock3D_complex(16, 16, kernel_size=(3, 3, 3), padding=(1, 1, 1),residual=True)
        self.enc1_3 = ConvBlock3D_complex(16, 16, kernel_size=(3, 3, 3), padding=(1, 1, 1),residual=True)
        self.enc2 = ConvBlock3D_complex(16, 32, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.enc2_1 = ConvBlock3D_complex(32, 32, kernel_size=(3, 3, 3), padding=(1, 1, 1),residual=True)
        self.enc2_2 = ConvBlock3D_complex(32, 32, kernel_size=(3, 3, 3), padding=(1, 1, 1),residual=True)
        self.enc2_3 = ConvBlock3D_complex(32, 32, kernel_size=(3, 3, 3), padding=(1, 1, 1),residual=True)
        self.enc3 = ConvBlock3D_complex(32, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.enc3_1 = ConvBlock3D_complex(64, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1),residual=True)
        self.enc3_2 = ConvBlock3D_complex(64, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1),residual=True)
        self.enc3_3 = ConvBlock3D_complex(64, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1),residual=True)
        self.pool = nn.MaxPool3d(kernel_size=(1, 1, 2), stride=(1, 1, 2))

        # Bottleneck
        self.bottleneck = ConvBlock3D(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1))

        # Decoder with dilated convolutions
        self.upconv3 = nn.ConvTranspose3d(128, 64, kernel_size=(1, 1, 2), stride=(1, 1, 2))
        self.dec3 = ConvBlock3D(128, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1), dilation=1)
        self.upconv2 = nn.ConvTranspose3d(64, 32, kernel_size=(1, 1, 2), stride=(1, 1, 2))
        self.dec2 = ConvBlock3D(64, 32, kernel_size=(3, 3, 3), padding=(1, 1, 1), dilation=1)
        self.upconv1 = nn.ConvTranspose3d(32, 16, kernel_size=(1, 1, 2), stride=(1, 1, 2))
        self.dec1 = ConvBlock3D(32, 16, kernel_size=(3, 3, 3), padding=(1, 1, 1), dilation=1)

        # Output layer
        self.conv_out = nn.Conv3d(16, 20, kernel_size=1)  # 输出通道为20，匹配最终输出尺寸

    def forward(self, x):
        # Encoder
        batch,channel,height,width,d = x.shape
        enc0_1 = self.enc0_1(x)
        enc1 = self.enc1(enc0_1)
        enc1 = self.enc1_1(enc1)
        enc1 = self.enc1_2(enc1)
        enc1 = self.enc1_3(enc1)
        enc2 = self.enc2(self.pool(enc1))
        enc2 = self.enc2_1(enc2)
        enc2 = self.enc2_2(enc2)
        enc2 = self.enc2_3(enc2)
        enc3 = self.enc3(self.pool(enc2))
        enc3 = self.enc3_1(enc3)
        enc3 = self.enc3_2(enc3)
        enc3 = self.enc3_3(enc3)
        pooled = self.pool(enc3)
        # print(enc1.shape)
        # print(enc2.shape)
        # print(enc3.shape)
        e1 = enc1.permute(0,2,3,4,1).reshape(batch,height,width,-1)
        e2 = enc2.permute(0,2,3,4,1).reshape(batch,height,width,-1)
        e3 = enc3.permute(0,2,3,4,1).reshape(batch,height,width,-1)
        # Bottleneck
        bottleneck = self.bottleneck(pooled)

        # Decoder
        up3 = self.upconv3(bottleneck)
        up3 = F.pad(up3, [0, enc3.size(4) - up3.size(4), 0, enc3.size(3) - up3.size(3), 0, enc3.size(2) - up3.size(2)])
        dec3 = self.dec3(torch.cat((up3, enc3), dim=1))
        
        up2 = self.upconv2(dec3)
        up2 = F.pad(up2, [0, enc2.size(4) - up2.size(4), 0, enc2.size(3) - up2.size(3), 0, enc2.size(2) - up2.size(2)])
        dec2 = self.dec2(torch.cat((up2, enc2), dim=1))
        
        up1 = self.upconv1(dec2)
        up1 = F.pad(up1, [0, enc1.size(4) - up1.size(4), 0, enc1.size(3) - up1.size(3), 0, enc1.size(2) - up1.size(2)])
        dec1 = self.dec1(torch.cat((up1, enc1), dim=1))

        # Output
        out = self.conv_out(dec1)
        out = out.permute(0,2,3,4,1).reshape(batch,height,width,-1)
        return out,e1,e2,e3

# class Toface_Decoder_final_complex(nn.Module):
#     def __init__(self, out_channels):
#         super(Toface_Decoder_final_complex, self).__init__()
#         self.upconv1 = nn.Conv2d(int(256), 448, kernel_size=3, padding=1)
#         self.decoder1 = DoubleConv(448*2, 448)
#         self.upconv2 = nn.Conv2d(448, 448, kernel_size=3, padding=1)
#         self.decoder2 = DoubleConv(448*2, 448)
#         self.decoder2_1 = DoubleConv(448, 448)
#         self.decoder2_2 = DoubleConv(448, 448)
#         self.decoder2_3 = DoubleConv(448, 448)
#         self.upconv3 = nn.Conv2d(448, 448, kernel_size=3, padding=1)
#         self.decoder3 = DoubleConv(448*2, 448)
#         self.decoder3_1 = DoubleConv(448, 448)
#         self.decoder3_2 = DoubleConv(448, 448)
#         self.decoder3_3 = DoubleConv(448, 448)

#         self.final_conv = nn.Conv2d(448, out_channels, kernel_size=1)

#     def forward(self, m, e1, e2,e3):
#         u1 = self.upconv1(m)
#         d1 = self.decoder1(torch.cat([u1, e3], dim=1))
#         u2 = self.upconv2(d1)
#         d2 = self.decoder2(torch.cat([u2, e2], dim=1))
#         d2_1 = self.decoder2_1(d2)
#         d2_1 = d2_1 + d2
#         d2_2 = self.decoder2_2(d2_1)
#         d2_2 = d2_2 + d2_1
#         d2_3 = self.decoder2_3(d2_2)
#         d2_3 = d2_3 + d2_2
#         u3 = self.upconv3(d2_3)
#         d3 = self.decoder3(torch.cat([u3, e1], dim=1))
#         d3_1 = self.decoder3_1(d3)
#         d3_1 = d3_1 + d3
#         d3_2 = self.decoder3_2(d3_1)
#         d3_2 = d3_2 + d3_1
#         d3_3 = self.decoder3_3(d3_2)
#         d3_3 = d3_3 + d3_2

#         out = self.final_conv(d3_3)
#         return out
    
class Toface_Decoder_final_complex(nn.Module):
    def __init__(self, out_channels):
        super(Toface_Decoder_final_complex, self).__init__()

        # Using ConvTranspose2d to upsample spatial dimensions
        self.upconv1 = nn.Conv2d(256, 448, kernel_size=3, padding=1)
        self.decoder1 = DoubleConv(448 * 2, 448)
        self.upconv2 = nn.ConvTranspose2d(448, 112, kernel_size=2, stride=2)
        self.decoder2 = DoubleConv(448+112, 448)
        self.decoder2_1 = DoubleConv(448, 224)
        self.decoder2_2 = DoubleConv(224, 224)
        self.decoder2_3 = DoubleConv(224, 224)
        self.upconv3 = nn.ConvTranspose2d(224, 56, kernel_size=2, stride=2)
        self.decoder3 = DoubleConv(448+56, 448)
        self.decoder3_1 = DoubleConv(448, 224)
        self.decoder3_2 = DoubleConv(224, 112)
        self.decoder3_3 = DoubleConv(112, 56)

        self.final_conv = nn.Conv2d(56, out_channels, kernel_size=1)

    def forward(self, m, e1, e2, e3):
        # Upsampling and merging features
        u1 = self.upconv1(m)
        d1 = self.decoder1(torch.cat([u1, e3], dim=1))
        u2 = self.upconv2(d1)
        e2 = F.interpolate(e2, size=(16, 16), mode='bilinear', align_corners=False)
        d2 = self.decoder2(torch.cat([u2, e2], dim=1))
        d2_1 = self.decoder2_1(d2)
        # d2_1 = d2_1 + d2
        d2_2 = self.decoder2_2(d2_1)
        # d2_2 = d2_2 + d2_1
        d2_3 = self.decoder2_3(d2_2)
        # d2_3 = d2_3 + d2_2
        u3 = self.upconv3(d2_3)
        e1 = F.interpolate(e1, size=(32, 32), mode='bilinear', align_corners=False)
        d3 = self.decoder3(torch.cat([u3, e1], dim=1))
        d3_1 = self.decoder3_1(d3)
        # d3_1 = d3_1 + d3
        d3_2 = self.decoder3_2(d3_1)
        # d3_2 = d3_2 + d3_1
        d3_3 = self.decoder3_3(d3_2)
        # d3_3 = d3_3 + d3_2

        out = self.final_conv(d3_3)
        return out

class ToFace_Unet_3Dcnn_complex(nn.Module):
    def __init__(self, out_size):
        super(ToFace_Unet_3Dcnn_complex, self).__init__()
        # self.upsample = Upsample1()
        # self.encoder = ToFace_Encoder(28)
        self.up = UNet3D_complex()
        out_channels = out_size/8 #eg:16*16 output should has 4 output channels
        self.out_size = out_size
        
        out_channels = int(out_channels**2)
        self.out_channels = out_channels
        # print(out_channels)
        self.branch1 = Toface_Decoder_final_complex(1)
        self.branch2 = Toface_Decoder_final_complex(1)
        self.swt = ToFace_Unet_swt()
        # self.swt = ToFace_Unet_Dwt_locate()
        # self.dwt = ToFace_Unet_Dwt()
        self.peak = ToFace_Unet_Intense()
        # self.peak = ToFace_Unet_Peak_cnn()
        self.depth_fc = nn.Linear(560, 560)
        self.classifier = FER_Classifier(448)


    def forward(self, x):
        start_time = time.time()
        # upsample = self.upsample(x)
        upsample,e1,e2,e3 = self.up(x)
        upsample = upsample.permute(0,3,1,2)
        # print(e1.shape)
        e1 = e1.permute(0,3,1,2)
        e2 = e2.permute(0,3,1,2)
        e3 = e3.permute(0,3,1,2)
        swt = self.swt(upsample)
        # print('swt:',time.time()-start_time)
        start_time = time.time()
        # width = self.dwt(upsample)
        # encoded, e1, e2 = self.encoder(x)
        # print(swt.shape)
        # print(encoded.shape)
        # print(swt.shape)
        # print(e3.shape)
        # depth_encoded = torch.cat([swt, e3], dim=1)
        depth_encoded = swt
        # print   (depth_encoded.shape)   
        # peak = self.peak(upsample)
        # peak = peak.permute(0,3,1,2)
        # depth_encoded = torch.cat([encoded,peak], dim=1)
        depth = self.branch1(depth_encoded, e1, e2,e3)
        # print(depth_encoded.shape)
        # print   (depth.shape)
        depth = depth.permute(0,2,3,1)
        tmp = depth.reshape(-1,self.out_channels)
        # stop the gradient of tmp
        tmp = tmp.detach()
        tmp = (tmp-262.5)/1.875
        indicators = torch.zeros(tmp.shape[0],560,device=tmp.device,dtype=torch.float32)
        tmp = tmp.long().clamp(0, 560 - 1)
        # for i in range(tmp.shape[0]):
        #     indicators[i,tmp[i]] = 1
        indicators.scatter_(1,tmp,1)
        # print('depth:',time.time()-start_time)
        position = self.depth_fc(indicators)
        position = position.view(-1,8,8,560)
        position = position.permute(0,3,1,2)
        # print(position.shape)
        # print(upsample.shape)
        position_up = position*upsample
        peak = self.peak(position_up)
        # print(peak.shape)
        peak = peak.squeeze(-2)
        peak = peak.permute(0,3,1,2)
        
        
        # print(intense.shape)
        # print(width.shape)
        # print(encoded.shape)
        # width_encoded = torch.cat([peak, e3], dim=1)
        width_encoded = peak
        orientation = self.branch2(width_encoded, e1, e2,e3)
        orientation = orientation.permute(0,2,3,1)
        # classifier_encoded = torch.cat([e3, peak,swt], dim=1)
        classifier_encoded = e3
        # print(classifier_encoded.shape)
        classifier_output = self.classifier(classifier_encoded)
        return orientation, depth, upsample,classifier_output


class Toface_Decoder_final_complex_class(nn.Module):
    def __init__(self, out_channels):
        super(Toface_Decoder_final_complex_class, self).__init__()

        # Using ConvTranspose2d to upsample spatial dimensions
        self.upconv1 = nn.Conv2d(256*2, 448, kernel_size=3, padding=1)
        self.decoder1 = DoubleConv(448 * 2, 448)
        # self.upconv2 = nn.ConvTranspose2d(448, 112, kernel_size=2, stride=2)
        self.decoder2 = DoubleConv(448*2, 448)
        self.decoder2_1 = DoubleConv(448, 112)
        self.decoder2_2 = DoubleConv(112, 112)
        self.decoder2_3 = DoubleConv(112, 56)

        # self.upconv3 = nn.ConvTranspose2d(224, 56, kernel_size=2, stride=2)
        self.decoder3 = DoubleConv(448+56, 448)
        self.decoder3_1 = DoubleConv(448, 112)
        self.decoder3_2 = DoubleConv(112, 112)
        self.decoder3_3 = DoubleConv(112, 28)

        # self.final_conv = nn.Conv2d(56, out_channels, kernel_size=1)
        self.classification = nn.Sequential(
            nn.Linear(28*8*8,768),
            nn.ReLU(inplace=True),
            nn.Linear(768, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 7)
        )

    def forward(self, m, e1, e2, e3):
        # Upsampling and merging features
        u1 = self.upconv1(m)
        d1 = self.decoder1(torch.cat([u1, e3], dim=1))
        # u2 = self.upconv2(d1)
        # e2 = F.interpolate(e2, size=(16, 16), mode='bilinear', align_corners=False)
        d2 = self.decoder2(torch.cat([d1, e2], dim=1))
        d2_1 = self.decoder2_1(d2)
        # d2_1 = d2_1 + d2
        d2_2 = self.decoder2_2(d2_1)
        d2_2 = d2_2 + d2_1
        d2_3 = self.decoder2_3(d2_2)
        # d2_3 = d2_3 + d2_2
        # u3 = self.upconv3(d2_3)
        # e1 = F.interpolate(e1, size=(32, 32), mode='bilinear', align_corners=False)
        d3 = self.decoder3(torch.cat([d2_3, e1], dim=1))
        d3_1 = self.decoder3_1(d3)
        # d3_1 = d3_1 + d3
        d3_2 = self.decoder3_2(d3_1)
        d3_2 = d3_2 + d3_1
        d3_3 = self.decoder3_3(d3_2)
        # d3_3 = d3_3 + d3_2
        d3_3 = d3_3.permute(0,2,3,1)
        d3_3 = d3_3.reshape(-1,28*8*8)
        out = self.classification(d3_3)
        return out


class ToFace_Unet_3Dcnn_complex_class(nn.Module):
    def __init__(self, out_size):
        super(ToFace_Unet_3Dcnn_complex_class, self).__init__()
        # self.upsample = Upsample1()
        # self.encoder = ToFace_Encoder(28)
        self.up = UNet3D_complex()
        # out_channels = out_size/8 #eg:16*16 output should has 4 output channels
        # self.out_size = out_size
        
        # out_channels = int(out_channels**2)
        # self.out_channels = out_channels
        # print(out_channels)
        self.branch1 = Toface_Decoder_final_complex_class(1)
        # self.branch2 = Toface_Decoder_final_complex_class(1)
        self.swt = ToFace_Unet_swt()
        # self.swt = ToFace_Unet_Dwt_locate()
        # self.dwt = ToFace_Unet_Dwt()
        self.peak = ToFace_Unet_Intense()
        # self.peak = ToFace_Unet_Peak_cnn()
        # self.depth_fc = nn.Linear(560, 560)
        # self.classifier = FER_Classifier(448)


    def forward(self, x):
        start_time = time.time()
        # upsample = self.upsample(x)
        upsample,e1,e2,e3 = self.up(x)
        upsample = upsample.permute(0,3,1,2)
        # print(e1.shape)
        e1 = e1.permute(0,3,1,2)
        e2 = e2.permute(0,3,1,2)
        e3 = e3.permute(0,3,1,2)
        swt = self.swt(upsample)
        # print('swt:',time.time()-start_time)
        start_time = time.time()
        depth_encoded = swt
        # depth = self.branch1(depth_encoded, e1, e2,e3)
        # depth = depth.permute(0,2,3,1)
        # tmp = depth.reshape(-1,self.out_channels)
        # # stop the gradient of tmp
        # tmp = tmp.detach()
        # tmp = (tmp-262.5)/1.875
        # indicators = torch.zeros(tmp.shape[0],560,device=tmp.device,dtype=torch.float32)
        # tmp = tmp.long().clamp(0, 560 - 1)
        # indicators.scatter_(1,tmp,1)
        # position = self.depth_fc(indicators)
        # position = position.view(-1,8,8,560)
        # position = position.permute(0,3,1,2)
        
        # position_up = position*upsample
        peak = self.peak(upsample)
        # print(peak.shape)
        peak = peak.squeeze(-2)
        peak = peak.permute(0,3,1,2)
        
        
        # print(intense.shape)
        # print(width.shape)
        # print(encoded.shape)
        # width_encoded = torch.cat([peak, e3], dim=1)
        width_encoded = peak
        encode = torch.cat([depth_encoded, width_encoded], dim=1)
        classification = self.branch1(encode, e1, e2,e3)
        # orientation = self.branch2(width_encoded, e1, e2,e3)
        # orientation = orientation.permute(0,2,3,1)
        # classifier_encoded = torch.cat([e3, peak,swt], dim=1)
        # classifier_encoded = e3
        # # print(classifier_encoded.shape)
        # classifier_output = self.classifier(classifier_encoded)
        return upsample,classification





if __name__ == '__main__':
    # dwt = Dwt_CnnMLP(1)
    # print(dwt)
    # tmp = torch.randn(2, 8, 8, 360)
    # output = dwt(tmp)
    # out = torch.randn(2, 4,8, 8)
    # loss = nn.MSELoss()
    # l = loss(output, out)
    # l.backward()
    # print(l)


    # swt = WaveletConvMLP()
    # print(swt)
    # tmp = torch.randn(2, 360, 8, 8 )
    # output = swt(tmp)
    # out = torch.randn(2, 8,8, 4)
    # loss = nn.MSELoss()
    # l = loss(output, out)
    # l.backward()
    # print(l)

    # upsample = Upsample1()
    # print(upsample)
    # tmp = torch.randn(2, 18, 8, 8 )
    # output = upsample(tmp)
    # out = torch.randn(2, 360,8, 8)
    # loss = nn.MSELoss()
    # l = loss(output, out)
    # l.backward()
    # print(l)

    # toface = ToFace()
    # print(toface)
    # tmp = torch.randn(2, 18, 8, 8 )
    # output = toface(tmp)
    # out3 = torch.randn(2, 360,8, 8)
    # out2 = torch.randn(2, 8,8, 4)
    # out1 = torch.randn(2, 8,8, 4)
    # loss = nn.MSELoss()
    # l1 = loss(output[0], out1)
    # l2 = loss(output[1], out2)
    # l3 = loss(output[2], out3)
    # l = l1+l2+l3
    # l.backward()
    # print(l)

    # toface = ToFace_Unet_cnn()
    # print(toface)
    # tmp = torch.randn(2, 18, 8, 8 )
    # output = toface(tmp)
    # print(output[1].shape)

    # toface = ToFace_Unet_3Dcnn_complex(32)
    # print(toface)
    # tmp = torch.randn(1, 1, 8, 8, 28 )
    # flop_analyzer = FlopCountAnalysis(toface, tmp)
    # # output = toface(tmp)
    # total_params = sum(p.numel() for p in toface.parameters())
    # flops = flop_analyzer.total()
    # # print(output[0].shape)
    # # print(output[1].shape)
    # # print(output[2].shape)
    # print(f"Model has {total_params:,} parameters.")
    # print(f"Model size is approximately {total_params * 4 / (1024**2):.2f} MB.")
    # print(f"Model has {flops/1e9:.2f} GFLOPs.")

    # Toface = ToFace_Unet_cnn(32)
    # print(Toface)
    # tmp = torch.randn(1, 28, 8, 8)
    # flop_analyzer = FlopCountAnalysis(Toface, tmp)
    # # output = toface(tmp)
    # total_params = sum(p.numel() for p in Toface.parameters())
    # flops = flop_analyzer.total()
    # print(f"Model has {total_params:,} parameters.")
    # print(f"Model size is approximately {total_params * 4 / (1024**2):.2f} MB.")
    # print(f"Model has {flops/1e9:.2f} GFLOPs.")


    # unet_baseline = Unet_baseline()
    # print(unet_baseline)
    # tmp = torch.randn(1, 28, 8, 8)
    # flop_analyzer = FlopCountAnalysis(unet_baseline, tmp)
    # # output = toface(tmp)
    # total_params = sum(p.numel() for p in unet_baseline.parameters())
    # flops = flop_analyzer.total()
    # print(f"Model has {total_params:,} parameters.")
    # print(f"Model size is approximately {total_params * 4 / (1024**2):.2f} MB.")
    # print(f"Model has {flops/1e9:.2f} GFLOPs.")

    D3_cnn = ToFace_Unet_3Dcnn(32)
    print(D3_cnn)
    tmp = torch.randn(1, 1, 8, 8, 28)
    flop_analyzer = FlopCountAnalysis(D3_cnn, tmp)
    # output = toface(tmp)
    total_params = sum(p.numel() for p in D3_cnn.parameters())
    flops = flop_analyzer.total()
    print(f"Model has {total_params:,} parameters.")
    print(f"Model size is approximately {total_params * 4 / (1024**2):.2f} MB.")
    print(f"Model has {flops/1e9:.2f} GFLOPs.")


    # toface = ToFace_Unet_3Dcnn_complex_class(32)
    # print(toface)
    # tmp = torch.randn(2, 4, 8, 8, 28 )
    # output = toface(tmp)
    # print(output[0].shape)
    # print(output[1].shape)
    # # print(output[2].shape)