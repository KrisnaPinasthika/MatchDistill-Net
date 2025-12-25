import torch
import torch.nn as nn
from pytorch3d.loss import chamfer_distance
from torch.nn.utils.rnn import pad_sequence
from torchmetrics.functional import image_gradients, structural_similarity_index_measure
import torch.nn.functional as F 
import numpy as np 

class LogRMSE(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target, mask=None, interpolate=True):
        
        if interpolate:
            input = nn.functional.interpolate(input, target.shape[-2:], mode='bilinear', align_corners=True)

        if mask is not None:
            input = input[mask]
            target = target[mask]
        
        eps = 1e-8
        pred = torch.log(input + eps)
        gt = torch.log(target + eps)

        calc = torch.sqrt(torch.mean((pred - gt) ** 2))
        return calc
    
class SoftDeltaLoss(nn.Module):
    def __init__(self, threshold=1.25, k=10.0):
        super().__init__()
        self.name = 'SoftDeltaLoss'
        self.tau = np.log(threshold)
        self.k = k
        self.berhu = BerHuLoss()

    def forward(self, input, target, mask=None, interpolate=True):
        berhu_calc = self.berhu(torch.log(input), torch.log(target), mask=mask, interpolate=interpolate)
        if interpolate:
            input = nn.functional.interpolate(input, target.shape[-2:], mode='bilinear', align_corners=True)

        if mask is not None:
            input = input[mask]
            target = target[mask]
        
        g = torch.log(input) - torch.log(target)

        Dg = torch.var(g) + 0.15 * berhu_calc
        
        # print(g)
        deltaloss = torch.relu(torch.abs(g) - self.tau).mean()
        
        return self.k * (torch.sqrt(Dg) + deltaloss)
        # return self.k * torch.sqrt(Dg) + 0.3 * deltaloss 
        # return self.k * torch.sqrt(Dg) + deltaloss #  OG


class BerHuLoss(nn.Module):  # Main loss function used in AdaBins paper
    
    def __init__(self):
        super(BerHuLoss, self).__init__()
        self.name = 'BerHuLoss'

    def forward(self, input, target, mask=None, interpolate=True):
        if interpolate:
            input = nn.functional.interpolate(input, target.shape[-2:], mode='bilinear', align_corners=True)

        if mask is not None:
            input = input[mask]
            target = target[mask]
        
        abs_value = (input-target).abs()
        max_val = abs_value.max()
        
        condition = 0.2 * max_val
        
        L2_loss = ((abs_value**2+condition**2)/(2*condition))
        
        return torch.where(abs_value > condition, L2_loss, abs_value).mean()



class SILogLoss(nn.Module):  # Main loss function used in AdaBins paper
    
    def __init__(self):
        super(SILogLoss, self).__init__()
        self.name = 'SILog'

    def forward(self, input, target, mask=None, interpolate=True):
        if interpolate:
            input = nn.functional.interpolate(input, target.shape[-2:], mode='bilinear', align_corners=True)

        if mask is not None:
            input = input[mask]
            target = target[mask]
        
        g = torch.log(input) - torch.log(target)

        Dg = torch.var(g) + 0.15 * torch.pow(torch.mean(g), 2)
        return 10 * torch.sqrt(Dg)
    
class BinsChamferLoss(nn.Module):  # Bin centers regularizer used in AdaBins paper
    def __init__(self):
        super().__init__()
        self.name = "ChamferLoss"

    def forward(self, bins, target_depth_maps):
        bin_centers = 0.5 * (bins[:, 1:] + bins[:, :-1])
        n, p = bin_centers.shape
        input_points = bin_centers.view(n, p, 1)  # .shape = n, p, 1
        # n, c, h, w = target_depth_maps.shape

        target_points = target_depth_maps.flatten(1)  # n, hwc
        mask = target_points.ge(1e-3)  # only valid ground truth points
        target_points = [p[m] for p, m in zip(target_points, mask)]
        target_lengths = torch.Tensor([len(t) for t in target_points]).long().to(target_depth_maps.device)
        target_points = pad_sequence(target_points, batch_first=True).unsqueeze(2)  # .shape = n, T, 1

        loss, _ = chamfer_distance(x=input_points, y=target_points, y_lengths=target_lengths)
        return loss

class L1Loss(nn.Module):  # Main loss function used in AdaBins paper
    
    def __init__(self):
        super(L1Loss, self).__init__()
        self.name = 'l1_loss'

    def forward(self, input, target, mask=None, interpolate=True):
        if interpolate:
            input = nn.functional.interpolate(input, target.shape[-2:], mode='bilinear', align_corners=True)

        if mask is not None:
            input = input[mask]
            target = target[mask]
        
        calc = torch.mean(torch.abs(input - target))
        
        return calc
    
class DepthSmoothness(nn.Module):  # Main loss function used in AdaBins paper
    def __init__(self):
        super(DepthSmoothness, self).__init__()
        self.name = 'DepthSmoothness'

    def forward(self, input, target, mask=None, interpolate=True):
        if interpolate:
            input = nn.functional.interpolate(input, target.shape[-2:], mode='bilinear', align_corners=True)

        if mask is not None:
            # input = input * mask # input[mask] 
            # target = target * mask # target[mask]
            input = input[mask]
            target = target[mask]
            
        eps = 1e-8
        input = torch.log(input + eps)
        target = torch.log(target + eps)

        dy_true, dx_true = image_gradients(target)
        dy_pred, dx_pred = image_gradients(input)

        # weights_x = torch.exp(torch.mean(torch.abs(target)))
        # weights_y = torch.exp(torch.mean(torch.abs(target)))

        # Depth smoothness
        # smoothness_x = dx_pred * weights_x
        # smoothness_y = dy_pred * weights_y

        depth_smoothness_loss = torch.mean((dy_pred - dy_true)**2) + torch.mean((dx_pred - dx_true)**2)
        # print(f"Depth smoothness : {depth_smoothness_loss.item()}")
        return depth_smoothness_loss

class LossSSIM(nn.Module):  # Main loss function used in AdaBins paper3
    
    def __init__(self, max_val):
        super(LossSSIM, self).__init__()
        self.name = 'LSSIM'
        self.max_val = max_val

    def forward(self, input, target, mask=None, interpolate=True):
        if interpolate:
            input = nn.functional.interpolate(input, target.shape[-2:], mode='bilinear', align_corners=True)
            

        if mask is not None:
            # input = input * mask # input[mask] 
            # target = target * mask # target[mask]
            input = input[mask]
            target = target[mask]
        
        eps = 1e-8
        input = torch.log(input + eps)
        target = torch.log(target + eps)

        return self.loss_ssim(input, target, self.max_val, kernel_size=11, size_average=True, full=False, k1=0.01, k2=0.03)
    
    def gaussian(self, window_size, sigma):
        gauss = torch.Tensor([ np.exp(-(x - window_size//2)**2 / float(2*sigma**2)) for x in range(window_size)])
        return gauss/gauss.sum()

    def create_window(self, window_size, channel=1):
        _1D_window = self.gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return window

    def loss_ssim(self, img1, img2, max_val, kernel_size=11, size_average=True, full=False, k1=0.01, k2=0.03):
        padd = 0
        (batch, channel, height, width) = img1.size()

        real_size = min(kernel_size, height, width)
        window = self.create_window(real_size, channel=channel).to(img1.device)

        mu1 = F.conv2d(img1, window, padding=padd, groups=channel)
        mu2 = F.conv2d(img2, window, padding=padd, groups=channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        # print(f"{mu1}\n{mu2}\n{mu1_mu2}")

        sigma1_sq = F.conv2d(img1 * img1, window, padding=padd, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, window, padding=padd, groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, padding=padd, groups=channel) - mu1_mu2

        C1 = (k1 * max_val) ** 2
        C2 = (k2 * max_val) ** 2

        v1 = 2.0 * sigma12 + C2
        v2 = sigma1_sq + sigma2_sq + C2
        cs = torch.mean(v1 / v2)  # contrast sensitivity

        ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)

        if size_average:
            ret = ssim_map.mean()
        else:
            ret = ssim_map.mean(1).mean(1).mean(1)

        if full:
            return ret, cs
        # print(f"L SSIM : {(1 - ret).item()}")
        # print()
        return torch.clamp((1 - ret), 0, 1)