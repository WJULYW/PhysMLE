""" Utilities """
import os
import shutil
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Function
import numpy as np
import argparse
from torch.autograd import Variable
from numpy import random
import math
import utils
from scipy.signal import find_peaks, welch
from scipy import signal
from scipy.fft import fft

args = utils.get_args()


class P_loss3(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, gt_lable, pre_lable):
        if len(gt_lable.shape) == 3:
            M, N, A = gt_lable.shape
            gt_lable = gt_lable - torch.mean(gt_lable, dim=2).view(M, N, 1)
            pre_lable = pre_lable - torch.mean(pre_lable, dim=2).view(M, N, 1)
        aPow = torch.sqrt(torch.sum(torch.mul(gt_lable, gt_lable), dim=-1))
        bPow = torch.sqrt(torch.sum(torch.mul(pre_lable, pre_lable), dim=-1))
        pearson = torch.sum(torch.mul(gt_lable, pre_lable), dim=-1) / (aPow * bPow + 0.001)
        loss = 1 - torch.sum(torch.sum(pearson, dim=1), dim=0) / (gt_lable.shape[0] * gt_lable.shape[1])
        return loss


class SP_loss(nn.Module):
    def __init__(self, device, clip_length=256, delta=3, loss_type=1, use_wave=False):
        super(SP_loss, self).__init__()

        self.clip_length = clip_length
        self.time_length = clip_length
        self.device = device
        self.delta = delta
        self.delta_distribution = [0.4, 0.25, 0.05]
        self.low_bound = 40
        self.high_bound = 150

        self.bpm_range = torch.arange(self.low_bound, self.high_bound, dtype=torch.float).to(self.device)
        self.bpm_range = self.bpm_range / 60.0

        self.pi = 3.14159265
        two_pi_n = Variable(2 * self.pi * torch.arange(0, self.time_length, dtype=torch.float))
        hanning = Variable(torch.from_numpy(np.hanning(self.time_length)).type(torch.FloatTensor),
                           requires_grad=True).view(1, -1)

        self.two_pi_n = two_pi_n.to(self.device)
        self.hanning = hanning.to(self.device)

        self.cross_entropy = nn.CrossEntropyLoss()
        self.nll = nn.NLLLoss()
        self.l1 = nn.L1Loss()

        self.loss_type = loss_type
        self.eps = 0.0001

        self.lambda_l1 = 0.1
        self.use_wave = use_wave

    def forward(self, wave, gt, pred=None, flag=None):  # all variable operation
        fps = 30

        hr = gt.clone()

        hr[hr.ge(self.high_bound)] = self.high_bound - 1
        hr[hr.le(self.low_bound)] = self.low_bound

        if pred is not None:
            pred = torch.mul(pred, fps)
            pred = pred * 60 / self.clip_length

        batch_size = wave.shape[0]

        f_t = self.bpm_range / fps
        preds = wave * self.hanning

        preds = preds.view(batch_size, 1, -1)
        f_t = f_t.repeat(batch_size, 1).view(batch_size, -1, 1)

        tmp = self.two_pi_n.repeat(batch_size, 1)
        tmp = tmp.view(batch_size, 1, -1)

        complex_absolute = torch.sum(preds * torch.sin(f_t * tmp), dim=-1) ** 2 \
                           + torch.sum(preds * torch.cos(f_t * tmp), dim=-1) ** 2

        target = hr - self.low_bound
        target = target.type(torch.long).view(batch_size)

        whole_max_val, whole_max_idx = complex_absolute.max(1)
        whole_max_idx = whole_max_idx + self.low_bound

        if self.loss_type == 1:
            loss = self.cross_entropy(complex_absolute, target)

        elif self.loss_type == 7:
            norm_t = (torch.ones(batch_size).to(self.device) / torch.sum(complex_absolute, dim=1))
            norm_t = norm_t.view(-1, 1)
            complex_absolute = complex_absolute * norm_t

            loss = self.cross_entropy(complex_absolute, target)

            idx_l = target - self.delta
            idx_l[idx_l.le(0)] = 0
            idx_r = target + self.delta
            idx_r[idx_r.ge(self.high_bound - self.low_bound - 1)] = self.high_bound - self.low_bound - 1;

            loss_snr = 0.0
            for i in range(0, batch_size):
                loss_snr = loss_snr + 1 - torch.sum(complex_absolute[i, idx_l[i]:idx_r[i]])

            loss_snr = loss_snr / batch_size

            loss = loss + loss_snr

        return loss, whole_max_idx


class MergeLoss(nn.Module):

    def __init__(self):
        super(MergeLoss, self).__init__()
        pass

    def forward(self, weight_c, weight_s, device):
        if len(weight_c.shape) == 2:
            H, W = weight_c.shape
            # weight_c = weight_c.view(H, W)
            # weight_s = weight_s.view(H, W)
            M = torch.mm(weight_c, weight_s.permute(1, 0)).cpu()
            ones = torch.ones(H, H, dtype=torch.float32)
            diag = torch.eye(H, dtype=torch.float32)
            tmp_extaFnorm = M - (ones - diag)
            uni_nosie = torch.randn(H, 1)
            u = torch.mm(tmp_extaFnorm, uni_nosie)
            v = torch.mm(tmp_extaFnorm, u)
            return (v.norm(p=2).sum() / (u.norm(p=2).sum() + 1e-12)).to(device)
        else:
            N, C, H, W = weight_c.shape
            weight_c = weight_c.view(N * C, H, W)
            weight_s = weight_s.view(N * C, H, W)
            M = torch.bmm(weight_c, weight_s.permute(0, 2, 1)).cpu()
            ones = torch.ones(N * C, H, H, dtype=torch.float32)  # .to(device)  # (N * C) * H * H
            diag = torch.eye(H, dtype=torch.float32)  # .to(device)  # (N * C) * H * H
            tmp_extaFnorm = M - (ones - diag)

            uni_nosie = torch.randn(N * C, H, 1)  # .to(device)

            u = torch.bmm(tmp_extaFnorm, uni_nosie)
            v = torch.bmm(tmp_extaFnorm, u)

            return (v.norm(p=2).sum() / (u.norm(p=2).sum() + 1e-12)).to(device)


class temporal_loss(nn.Module):
    def __init__(self, thr=2):
        super(temporal_loss, self).__init__()
        self.L1Loss = nn.L1Loss()
        self.thr = thr

    def forward(self, pre, pre_aug):  # all variable operation
        temp = torch.abs(pre - pre_aug)
        return self.L1Loss(torch.squeeze(torch.where(temp >= self.thr, pre_aug, pre)), torch.squeeze(pre_aug))


class bvp_hr_loss(nn.Module):
    def __init__(self):
        super(bvp_hr_loss, self).__init__()
        self.L1Loss = nn.L1Loss()

    def forward(self, pre_bvp, gt_hr):  # all variable operation
        pre_bvp = pre_bvp.squeeze()
        batch_size = pre_bvp.shape[0]
        device = pre_bvp.device
        pre_bvp = pre_bvp.cpu().detach().numpy()
        hrs = torch.zeros(batch_size)

        for i in range(batch_size):
            bvp = pre_bvp[i]
            bvp = (bvp - np.min(bvp)) / (np.max(bvp) - np.min(bvp))
            bvp = bvp.astype('float32')
            hr, _, _ = utils.hr_fft(bvp, fs=30, harmonics_removal=True)
            hr = np.array(hr)
            hr = hr.astype('float32')
            hrs[i] = hr.item()
        hrs = hrs.to(device)
        return self.L1Loss(hrs, gt_hr)


class bvp_rr_loss(nn.Module):
    def __init__(self, thr=5):
        super(bvp_rr_loss, self).__init__()
        self.L1Loss = nn.L1Loss()
        self.thr = thr

    def forward(self, gt_bvp, pre_rr):  # all variable operation
        device = gt_bvp.device
        if len(gt_bvp.shape) < 3:
            return torch.tensor(0).to(device)
        batch_size = gt_bvp.shape[0]
        gt_bvp = gt_bvp.squeeze()
        gt_bvp = gt_bvp.cpu().detach().numpy()
        rrs = torch.zeros(batch_size)

        for i in range(batch_size):
            bvp = gt_bvp[i]
            bvp = (bvp - np.min(bvp)) / (np.max(bvp) - np.min(bvp))
            bvp = bvp.astype('float32')
            peaks, _ = signal.find_peaks(bvp)

            rr_intervals = np.diff(peaks) / 30  # 30是BVP信号的采样率

            frequencies, power_spectrum = signal.welch(rr_intervals)

            respiratory_band = (0.12, 0.4)  # 呼吸频率带的范围
            mask = np.logical_and(frequencies >= respiratory_band[0], frequencies <= respiratory_band[1])
            if np.sum(mask) == 0:
                rrs[i] = 0
                continue
            respiratory_power_spectrum = power_spectrum[mask]
            respiratory_frequencies = frequencies[mask]
            respiratory_frequencies = respiratory_frequencies[np.argmax(respiratory_power_spectrum)]
            rr = 60 / 1 * respiratory_frequencies
            rrs[i] = rr.item()
        rrs = rrs.to(device)
        temp = torch.abs(rrs - pre_rr)
        return self.L1Loss(torch.squeeze(torch.where(temp >= self.thr, pre_rr, rrs)), torch.squeeze(rrs))


class Asp_loss(nn.Module):
    def __init__(self):
        super(Asp_loss, self).__init__()
        self.L1Loss = nn.L1Loss()

    def generate_aug(self, feat_spo, spo, num=5):
        if spo[0] == 0:
            return None, None
        self.batch_size = feat_spo.shape[0]
        device = feat_spo.device
        sample = 1 - 0.1 * torch.rand(dtype=float, size=(self.batch_size, int(num))).to(device)
        spo_aug = spo.unsqueeze(1).repeat(1, int(num))
        spo_aug = torch.mul(spo_aug, sample)
        sample = sample.unsqueeze(-1)
        sample.expand(self.batch_size, int(num), feat_spo.shape[1])
        feat_spo = feat_spo.unsqueeze(1).repeat(1, int(num), 1)
        feat_aug = torch.mul(feat_spo, sample)
        return feat_aug.float(), spo_aug.float()

    def forward(self, spo_pred, spo):  # all variable operation
        spo_pred = spo_pred.squeeze()
        return self.L1Loss(torch.where(spo >= 80, spo_pred, spo).view(-1, ), spo.view(-1, ))


class SpatialConsistencyLoss(nn.Module):
    def __init__(self, ratio=0.5):
        super().__init__()
        self.ratio = ratio
        self.Loss = nn.CosineSimilarity(dim=-1)

    def forward(self, FeatureMap):
        if len(FeatureMap.shape) == 3:
            B, W, L = FeatureMap.shape
            randm_size = int(self.ratio * W)
            if randm_size <= 0:
                randm_size = 1
            Index1 = np.random.randint(W, size=randm_size)
            Index2 = np.random.randint(W, size=randm_size)
            gt_lable = FeatureMap[:, Index1, :].clone().requires_grad_()
            pre_lable = FeatureMap[:, Index2, :].clone().requires_grad_()
            loss = self.Loss(gt_lable, pre_lable)
            loss = -1 * torch.mean(torch.mean(loss, dim=1), dim=0)
            return loss
        else:
            B, C, W, L = FeatureMap.shape
            randm_size = int(self.ratio * W)
            if randm_size <= 0:
                randm_size = 1
            Index1 = np.random.randint(W, size=randm_size)
            Index2 = np.random.randint(W, size=randm_size)
            gt_lable = FeatureMap[:, :, Index1, :].clone().requires_grad_()
            pre_lable = FeatureMap[:, :, Index2, :].clone().requires_grad_()
            loss = self.Loss(gt_lable, pre_lable)
            loss = -1 * torch.mean(torch.mean(torch.mean(loss, dim=1), dim=1), dim=0)
            return loss




def get_loss(bvp_pre, hr_pre, bvp_gt, hr_gt, dataName, loss_sig, loss_hr, args, inter_num, spo_pre=None, spo_gt=None,
              loss_spo=None, rf_pre=None, rf_gt=None, loss_rf=None):
    k = 2.0 / (1.0 + np.exp(-10.0 * inter_num / args.max_iter)) - 1.0
    if dataName == 'PURE':
        loss = (loss_sig(bvp_pre, bvp_gt) + loss_spo(torch.squeeze(spo_pre), spo_gt) / 10 + k * loss_hr(
            torch.squeeze(hr_pre), hr_gt) / 10) / 2
    elif dataName == 'UBFC':
        loss = (loss_sig(bvp_pre, bvp_gt) + k * loss_hr(torch.squeeze(hr_pre), hr_gt) / 10) / 2
    elif dataName == 'BUAA':
        loss = (loss_sig(bvp_pre, bvp_gt) + k * loss_hr(torch.squeeze(hr_pre), hr_gt) / 10) / 2
    elif dataName == 'VIPL':
        loss = (k * loss_hr(torch.squeeze(hr_pre), hr_gt) + k * loss_spo(torch.squeeze(spo_pre), spo_gt) / 10) / 2
    elif dataName == 'V4V':
        loss = (k * loss_hr(torch.squeeze(hr_pre), hr_gt) + k * loss_rf(torch.squeeze(rf_pre), rf_gt) / 10) / 2
        # loss = k * loss_hr(torch.squeeze(hr_pre), hr_gt)
    elif dataName == 'HCW':
        loss = (loss_sig(bvp_pre, bvp_gt) + k * loss_hr(torch.squeeze(hr_pre), hr_gt) / 10 + k * loss_rf(
            torch.squeeze(rf_pre), rf_gt) / 10) / 2

    if torch.sum(torch.isnan(loss)) > 0:
        print('Tere in nan loss found in ' + dataName)
    return loss



