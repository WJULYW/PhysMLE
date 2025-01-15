# -*- coding: UTF-8 -*-
import numpy as np
import os
from torch.utils.data import Dataset
import cv2
import csv
import scipy.io as scio
from scipy.signal import find_peaks, butter, filtfilt
import torchvision.transforms.functional as transF
import torchvision.transforms as transforms
from PIL import Image
from numpy.fft import fft, ifft, rfft, irfft
from torch.autograd import Variable
from utils import rr_cal
import random
import utils
import biosppy.signals as signals

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])


class Data_DG(Dataset):
    def __init__(self, root_dir, dataName, STMap, frames_num, args, transform=None, domain_label=None, jump=0):
        self.root_dir = root_dir
        self.dataName = dataName
        self.STMap_Name = STMap
        self.frames_num = int(frames_num)
        self.datalist = os.listdir(root_dir)
        self.datalist = sorted(self.datalist)
        self.num = len(self.datalist)
        self.domain_label = domain_label
        self.transform = transform
        self.args = args
        self.jump = jump

        self.transform = transforms.Compose([transforms.Resize(size=(64, 256)),
                                             transforms.ToTensor(),
                                             transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                  std=[0.229, 0.224, 0.225])
                                             ])
        self.transform_aug = transforms.Compose([
            transforms.Resize(size=(64, 256)),
            # transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.1),
            # transforms.RandomGrayscale(p=0.1),
            # transforms.RandomApply([transforms.GaussianBlur(kernel_size=7, sigma=(0.1, 2.0))], p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])



    def __len__(self):
        return self.num

    def getLabel(self, nowPath, Step_Index):
        # 读取每个数据集的心率值和BVP信号
        # nowPath=
        if self.dataName == 'COH':
            bvp_name = 'Label/BVP.mat'
            bvp_path = os.path.join(nowPath, bvp_name)
            bvp = scio.loadmat(bvp_path)['BVP']
            bvp = np.array(bvp.astype('float32')).reshape(-1)
            bvp = bvp[Step_Index:Step_Index + self.frames_num]
            bvp = (bvp - np.min(bvp)) / (np.max(bvp) - np.min(bvp))
            bvp = bvp.astype('float32')
            gt = np.array(0.0)
            gt = gt.astype('float32')

        elif self.dataName == 'BUAA':
            bvp_name = 'Label/BVP.mat'
            bvp_path = os.path.join(nowPath, bvp_name)
            bvp = scio.loadmat(bvp_path)['BVP']
            bvp = np.array(bvp.astype('float32')).reshape(-1)
            bvp = bvp[Step_Index:Step_Index + self.frames_num]
            bvp = (bvp - np.min(bvp)) / (np.max(bvp) - np.min(bvp))
            bvp = bvp.astype('float32')

            gt_name = 'Label/HR_256.mat'
            gt_path = os.path.join(nowPath, gt_name)
            gt = scio.loadmat(gt_path)['HR']
            gt = np.array(gt.astype('float32')).reshape(-1)
            gt = gt[int(Step_Index / 10)]
            gt = gt.astype('float32')

        elif self.dataName == 'VIPL':
            bvp_name = 'Label_CSI/BVP_Filt.mat'
            bvp_path = os.path.join(nowPath, bvp_name)
            bvp = scio.loadmat(bvp_path)['BVP']
            bvp = np.array(bvp.astype('float32')).reshape(-1)
            bvp = bvp[Step_Index:Step_Index + self.frames_num]
            bvp = (bvp - np.min(bvp)) / (np.max(bvp) - np.min(bvp))
            bvp = bvp.astype('float32')

            gt_name = 'Label_CSI/HR.mat'
            gt_path = os.path.join(nowPath, gt_name)
            gt = scio.loadmat(gt_path)['HR']
            gt = np.array(gt.astype('float32')).reshape(-1)
            gt = np.nanmean(gt[Step_Index:Step_Index + self.frames_num])
            gt = gt.astype('float32')

            sp_name = 'Label_CSI/SPO2.mat'
            sp_path = os.path.join(nowPath, sp_name)
            sp = scio.loadmat(sp_path)['SPO2']
            sp = np.array(sp.astype('float32')).reshape(-1)
            sp = np.nanmean(sp[Step_Index:Step_Index + self.frames_num])
            sp = sp.astype('float32')

            return gt, bvp, sp

        elif self.dataName == 'V4V':
            gt_name = 'Label/HR.mat'
            gt_path = os.path.join(nowPath, gt_name)
            gt = scio.loadmat(gt_path)['HR']
            gt = np.array(gt.astype('float32')).reshape(-1)
            gt = np.nanmean(gt[Step_Index:Step_Index + self.frames_num])
            gt = gt.astype('float32')
            bvp = np.array(0.0)
            bvp = bvp.astype('float32')

            rf_name = 'Label/RF.mat'
            rf_path = os.path.join(nowPath, rf_name)
            rf = scio.loadmat(rf_path)['RF']
            rf = np.array(rf.astype('float32')).reshape(-1)
            rf = np.nanmean(rf[Step_Index:Step_Index + self.frames_num])
            rf = rf.astype('float32')

            return gt, bvp, rf

        elif self.dataName == 'HCW':
            bvp_name = 'Label/BVP_Filt.mat'
            bvp_path = os.path.join(nowPath, bvp_name)
            bvp = scio.loadmat(bvp_path)['BVP']
            bvp = np.array(bvp.astype('float32')).reshape(-1)
            bvp = bvp[Step_Index:Step_Index + self.frames_num]
            bvp = (bvp - np.min(bvp)) / (np.max(bvp) - np.min(bvp))
            bvp = bvp.astype('float32')

            '''
            gt_name = 'Label/HR.mat'
            gt_path = os.path.join(nowPath, gt_name)
            gt = scio.loadmat(gt_path)['HR']
            gt = np.array(gt.astype('float32')).reshape(-1)
            gt = np.nanmean(gt[Step_Index:Step_Index + self.frames_num])
            gt = gt.astype('float32')'''

            gt, _, _ = utils.hr_fft(bvp, fs=30, harmonics_removal=True)
            gt = np.array(gt)
            gt = gt.astype('float32')

            rf_name = 'Label/RF_Filt.mat'
            rf_path = os.path.join(nowPath, rf_name)
            rf = scio.loadmat(rf_path)['RF']
            rf = np.array(rf.astype('float32')).reshape(-1)
            rf = rf[Step_Index:Step_Index + self.frames_num]
            rf = calculate_respiration_rate(rf)
            # rf = rf['resp_rate']
            '''
            rf = rf[Step_Index:Step_Index + self.frames_num]
            rf = (rf - np.min(rf)) / (np.max(rf) - np.min(rf))
            rf = rf.astype('float32')
            rf = calculate_respiration_rate(rf)
            rf = 6 + (rf * 18) / 70.3125'''
            rf = np.array(rf)
            rf = rf.astype('float32')

            return gt, bvp, rf

        elif self.dataName == 'PURE':
            bvp_name = 'Label/BVP.mat'
            bvp_path = os.path.join(nowPath, bvp_name)
            bvp = scio.loadmat(bvp_path)['BVP']
            bvp = np.array(bvp.astype('float32')).reshape(-1)
            bvp = bvp[Step_Index:Step_Index + self.frames_num]
            bvp = (bvp - np.min(bvp)) / (np.max(bvp) - np.min(bvp))
            bvp = bvp.astype('float32')

            gt_name = 'Label/HR.mat'
            gt_path = os.path.join(nowPath, gt_name)
            gt = scio.loadmat(gt_path)['HR']
            gt = np.array(gt.astype('float32')).reshape(-1)
            gt = np.nanmean(gt[Step_Index:Step_Index + self.frames_num])
            gt = gt.astype('float32')

            sp_name = 'Label/SPO2.mat'
            sp_path = os.path.join(nowPath, sp_name)
            sp = scio.loadmat(sp_path)['SPO2']
            sp = np.array(sp.astype('float32')).reshape(-1)
            sp = np.nanmean(sp[Step_Index:Step_Index + self.frames_num])
            sp = sp.astype('float32')
            return gt, bvp, sp

        elif self.dataName == 'UBFC':
            bvp_name = 'Label/BVP.mat'
            bvp_path = os.path.join(nowPath, bvp_name)
            bvp = scio.loadmat(bvp_path)['BVP']
            bvp = np.array(bvp.astype('float32')).reshape(-1)
            bvp = bvp[Step_Index:Step_Index + self.frames_num]
            bvp = (bvp - np.min(bvp)) / (np.max(bvp) - np.min(bvp))
            bvp = bvp.astype('float32')

            gt_name = 'Label/HR.mat'
            gt_path = os.path.join(nowPath, gt_name)
            gt = scio.loadmat(gt_path)['HR']
            gt = np.array(gt.astype('float32')).reshape(-1)
            gt = np.nanmean(gt[Step_Index:Step_Index + self.frames_num])
            gt = gt.astype('float32')

        return gt, bvp

    def __getitem__(self, idx):
        idx = idx + self.jump
        img_name = 'STMap'
        STMap_name = self.STMap_Name
        nowPath = os.path.join(self.root_dir, self.datalist[idx])
        temp = scio.loadmat(nowPath)
        nowPath = str(temp['Path'][0])
        # nowPath = nowPath.replace('/home/haolu', '/remote-home/hao.lu')
        Step_Index = int(temp['Step_Index'])
        # get HR value and bvp signal
        if self.dataName in ['PURE', 'VIPL']:
            gt, bvp, sp = self.getLabel(nowPath, Step_Index)
        elif self.dataName in ['HCW', 'V4V']:
            gt, bvp, rf = self.getLabel(nowPath, Step_Index)
        else:
            gt, bvp = self.getLabel(nowPath, Step_Index)
        # get STMap
        STMap_Path = os.path.join(nowPath, img_name)
        feature_map = cv2.imread(os.path.join(STMap_Path, STMap_name))
        With, Max_frame, _ = feature_map.shape
        # get original map
        map_ori = feature_map[:, Step_Index:Step_Index + self.frames_num, :]
        # get augmented map
        Spatial_aug_flag = 0
        Temporal_aug_flag = 0
        Step_Index_aug = Step_Index
        if self.args.spatial_aug_rate > 0:
            if (random.uniform(0, 100) / 100.0) < self.args.spatial_aug_rate:
                temp_ratio = (1.0 * random.uniform(0, 100) / 100.0)
                Index = np.arange(With)
                if temp_ratio < 0.3:
                    Index[random.randint(0, With - 1)] = random.randint(0, With - 1)
                    Index[random.randint(0, With - 1)] = random.randint(0, With - 1)
                    map_aug = map_ori[Index]
                elif temp_ratio < 0.6:
                    Index[random.randint(0, With - 1)] = random.randint(0, With - 1)
                    Index[random.randint(0, With - 1)] = random.randint(0, With - 1)
                    Index[random.randint(0, With - 1)] = random.randint(0, With - 1)
                    Index[random.randint(0, With - 1)] = random.randint(0, With - 1)
                    map_aug = map_ori[Index]
                elif temp_ratio < 0.9:
                    np.random.shuffle(Index[random.randint(0, With - 1):random.randint(0, With - 1)])
                    map_aug = map_ori[Index]
                else:
                    np.random.shuffle(Index)
                    map_aug = map_ori[Index]
                Spatial_aug_flag = 1
            else:
                map_aug = map_ori

        if ((Spatial_aug_flag == 0) and (self.args.temporal_aug_rate > 0)):
            if Step_Index + self.frames_num + 30 < Max_frame:
                if (random.uniform(0, 100) / 100.0) < self.args.temporal_aug_rate:
                    Step_Index_aug = int(random.uniform(0, 29) + Step_Index)
                    map_aug = feature_map[:, Step_Index_aug:Step_Index_aug + self.frames_num, :]
                    Temporal_aug_flag = 1
                else:
                    map_aug = map_ori
            else:
                map_aug = map_ori

        if ((Spatial_aug_flag == 0) and (Temporal_aug_flag == 0)):
            map_aug = map_ori

        if self.dataName in ['PURE', 'VIPL']:
            gt_aug, bvp_aug, sp_aug = self.getLabel(nowPath, Step_Index_aug)
        elif self.dataName in ['HCW', 'V4V']:
            gt_aug, bvp_aug, rf_aug = self.getLabel(nowPath, Step_Index_aug)
        else:
            gt_aug, bvp_aug = self.getLabel(nowPath, Step_Index_aug)
        # gt_aug, bvp_aug = self.getLabel(nowPath, Step_Index_aug)

        for c in range(map_ori.shape[2]):
            for r in range(map_ori.shape[0]):
                map_ori[r, :, c] = 255 * ((map_ori[r, :, c] - np.min(map_ori[r, :, c])) / \
                                          (0.00001 + np.max(map_ori[r, :, c]) - np.min(map_ori[r, :, c])))

        for c in range(map_aug.shape[2]):
            for r in range(map_aug.shape[0]):
                map_aug[r, :, c] = 255 * ((map_aug[r, :, c] - np.min(map_aug[r, :, c])) / \
                                          (0.00001 + np.max(map_aug[r, :, c]) - np.min(map_aug[r, :, c])))
        # if self.domain_label is not None:
        # domain_label = np.full((map_ori.shape[0], 1), np.inf)

        map_ori = Image.fromarray(np.uint8(map_ori))
        map_aug = Image.fromarray(np.uint8(map_aug))

        map_ori = self.transform(map_ori)
        map_aug = self.transform_aug(map_aug)

        if self.dataName in ['PURE', 'VIPL']:
            return (map_ori, bvp, gt, sp, gt, map_aug, bvp_aug, gt_aug, sp_aug, gt_aug, self.domain_label)
        elif self.dataName in ['HCW', 'V4V']:
            return (map_ori, bvp, gt, 0, rf, map_aug, bvp_aug, gt_aug, 0, rf_aug, self.domain_label)
        else:
            return (map_ori, bvp, gt, 0, gt, map_aug, bvp_aug, gt_aug, 0, gt_aug, self.domain_label)


class Data_DG_Intra(Dataset):
    def __init__(self, root_dir, dataName, STMap, frames_num, args, transform=None, domain_label=None, datalist=None):
        self.root_dir = root_dir
        self.dataName = dataName
        self.STMap_Name = STMap
        self.frames_num = int(frames_num)
        self.datalist = datalist
        self.datalist = sorted(self.datalist)
        self.num = len(self.datalist)
        self.domain_label = domain_label
        self.transform = transform
        self.args = args

        self.transform = transforms.Compose([transforms.Resize(size=(64, 256)),
                                             transforms.ToTensor(),
                                             transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                  std=[0.229, 0.224, 0.225])
                                             ])
        self.transform_aug = transforms.Compose([
            transforms.Resize(size=(64, 256)),  # transforms.Resize(size=(64, 256)),
            # transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.1),
            # transforms.RandomGrayscale(p=0.1),
            # transforms.RandomApply([transforms.GaussianBlur(kernel_size=7, sigma=(0.1, 2.0))], p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])


    def __len__(self):
        return self.num

    def getLabel(self, nowPath, Step_Index):
        # 读取每个数据集的心率值和BVP信号
        # nowPath=
        if self.dataName == 'COH':
            bvp_name = 'Label/BVP.mat'
            bvp_path = os.path.join(nowPath, bvp_name)
            bvp = scio.loadmat(bvp_path)['BVP']
            bvp = np.array(bvp.astype('float32')).reshape(-1)
            bvp = bvp[Step_Index:Step_Index + self.frames_num]
            bvp = (bvp - np.min(bvp)) / (np.max(bvp) - np.min(bvp))
            bvp = bvp.astype('float32')
            gt = np.array(0.0)
            gt = gt.astype('float32')

        elif self.dataName == 'BUAA':
            bvp_name = 'Label/BVP.mat'
            bvp_path = os.path.join(nowPath, bvp_name)
            bvp = scio.loadmat(bvp_path)['BVP']
            bvp = np.array(bvp.astype('float32')).reshape(-1)
            bvp = bvp[Step_Index:Step_Index + self.frames_num]
            bvp = (bvp - np.min(bvp)) / (np.max(bvp) - np.min(bvp))
            bvp = bvp.astype('float32')

            gt_name = 'Label/HR_256.mat'
            gt_path = os.path.join(nowPath, gt_name)
            gt = scio.loadmat(gt_path)['HR']
            gt = np.array(gt.astype('float32')).reshape(-1)
            gt = gt[int(Step_Index / 10)]
            gt = gt.astype('float32')

        elif self.dataName == 'VIPL':
            bvp_name = 'Label_CSI/BVP_Filt.mat'
            bvp_path = os.path.join(nowPath, bvp_name)
            bvp = scio.loadmat(bvp_path)['BVP']
            bvp = np.array(bvp.astype('float32')).reshape(-1)
            bvp = bvp[Step_Index:Step_Index + self.frames_num]
            bvp = (bvp - np.min(bvp)) / (np.max(bvp) - np.min(bvp))
            bvp = bvp.astype('float32')

            gt_name = 'Label_CSI/HR.mat'
            gt_path = os.path.join(nowPath, gt_name)
            gt = scio.loadmat(gt_path)['HR']
            gt = np.array(gt.astype('float32')).reshape(-1)
            gt = np.nanmean(gt[Step_Index:Step_Index + self.frames_num])
            gt = gt.astype('float32')

            sp_name = 'Label_CSI/SPO2.mat'
            sp_path = os.path.join(nowPath, sp_name)
            sp = scio.loadmat(sp_path)['SPO2']
            sp = np.array(sp.astype('float32')).reshape(-1)
            sp = np.nanmean(sp[Step_Index:Step_Index + self.frames_num])
            sp = sp.astype('float32')

            return gt, bvp, sp

        elif self.dataName == 'V4V':
            gt_name = 'Label/HR.mat'
            gt_path = os.path.join(nowPath, gt_name)
            gt = scio.loadmat(gt_path)['HR']
            gt = np.array(gt.astype('float32')).reshape(-1)
            gt = np.nanmean(gt[Step_Index:Step_Index + self.frames_num])
            gt = gt.astype('float32')
            bvp = np.array(0.0)
            bvp = bvp.astype('float32')

            rf_name = 'Label/RF.mat'
            rf_path = os.path.join(nowPath, rf_name)
            rf = scio.loadmat(rf_path)['RF']
            rf = np.array(rf.astype('float32')).reshape(-1)
            rf = np.nanmean(rf[Step_Index:Step_Index + self.frames_num])
            rf = rf.astype('float32')

            return gt, bvp, rf

        elif self.dataName == 'HCW':
            bvp_name = 'Label/BVP_Filt.mat'
            bvp_path = os.path.join(nowPath, bvp_name)
            bvp = scio.loadmat(bvp_path)['BVP']
            bvp = np.array(bvp.astype('float32')).reshape(-1)
            bvp = bvp[Step_Index:Step_Index + self.frames_num]
            bvp = (bvp - np.min(bvp)) / (np.max(bvp) - np.min(bvp))
            bvp = bvp.astype('float32')

            '''
            gt_name = 'Label/HR.mat'
            gt_path = os.path.join(nowPath, gt_name)
            gt = scio.loadmat(gt_path)['HR']
            gt = np.array(gt.astype('float32')).reshape(-1)
            gt = np.nanmean(gt[Step_Index:Step_Index + self.frames_num])
            gt = gt.astype('float32')'''


            gt, _, _ = utils.hr_fft(bvp, fs=30, harmonics_removal=True)
            gt = np.array(gt)
            gt = gt.astype('float32')

            #rf = rr_cal(bvp)

            rf_name = 'Label/RF_Filt.mat'
            rf_path = os.path.join(nowPath, rf_name)
            rf = scio.loadmat(rf_path)['RF']
            rf = np.array(rf.astype('float32')).reshape(-1)
            rf = rf[Step_Index:Step_Index + self.frames_num]
            rf = calculate_respiration_rate(rf)
            #rf = rf['resp_rate']
            '''
            rf = rf[Step_Index:Step_Index + self.frames_num]
            rf = (rf - np.min(rf)) / (np.max(rf) - np.min(rf))
            rf = rf.astype('float32')
            rf = calculate_respiration_rate(rf)
            rf = 6 + (rf * 18) / 70.3125'''
            rf = np.array(rf)
            rf = rf.astype('float32')

            return gt, bvp, rf

        elif self.dataName == 'PURE':
            bvp_name = 'Label/BVP.mat'
            bvp_path = os.path.join(nowPath, bvp_name)
            bvp = scio.loadmat(bvp_path)['BVP']
            bvp = np.array(bvp.astype('float32')).reshape(-1)
            bvp = bvp[Step_Index:Step_Index + self.frames_num]
            bvp = (bvp - np.min(bvp)) / (np.max(bvp) - np.min(bvp))
            bvp = bvp.astype('float32')

            gt_name = 'Label/HR.mat'
            gt_path = os.path.join(nowPath, gt_name)
            gt = scio.loadmat(gt_path)['HR']
            gt = np.array(gt.astype('float32')).reshape(-1)
            gt = np.nanmean(gt[Step_Index:Step_Index + self.frames_num])
            gt = gt.astype('float32')

            sp_name = 'Label/SPO2.mat'
            sp_path = os.path.join(nowPath, sp_name)
            sp = scio.loadmat(sp_path)['SPO2']
            sp = np.array(sp.astype('float32')).reshape(-1)
            sp = np.nanmean(sp[Step_Index:Step_Index + self.frames_num])
            sp = sp.astype('float32')
            return gt, bvp, sp

        elif self.dataName == 'UBFC':
            bvp_name = 'Label/BVP.mat'
            bvp_path = os.path.join(nowPath, bvp_name)
            bvp = scio.loadmat(bvp_path)['BVP']
            bvp = np.array(bvp.astype('float32')).reshape(-1)
            bvp = bvp[Step_Index:Step_Index + self.frames_num]
            bvp = (bvp - np.min(bvp)) / (np.max(bvp) - np.min(bvp))
            bvp = bvp.astype('float32')

            gt_name = 'Label/HR.mat'
            gt_path = os.path.join(nowPath, gt_name)
            gt = scio.loadmat(gt_path)['HR']
            gt = np.array(gt.astype('float32')).reshape(-1)
            gt = np.nanmean(gt[Step_Index:Step_Index + self.frames_num])
            gt = gt.astype('float32')

        return gt, bvp

    def __getitem__(self, idx):
        img_name = 'STMap'
        STMap_name = self.STMap_Name
        nowPath = os.path.join(self.root_dir, self.datalist[idx])
        temp = scio.loadmat(nowPath)
        nowPath = str(temp['Path'][0])
        # nowPath = nowPath.replace('/home/haolu', '/remote-home/hao.lu')
        Step_Index = int(temp['Step_Index'])
        # get HR value and bvp signal
        if self.dataName in ['PURE', 'VIPL']:
            gt, bvp, sp = self.getLabel(nowPath, Step_Index)
        elif self.dataName in ['HCW', 'V4V']:
            gt, bvp, rf = self.getLabel(nowPath, Step_Index)
        else:
            gt, bvp = self.getLabel(nowPath, Step_Index)
        # get STMap
        STMap_Path = os.path.join(nowPath, img_name)
        feature_map = cv2.imread(os.path.join(STMap_Path, STMap_name))
        With, Max_frame, _ = feature_map.shape
        # get original map
        map_ori = feature_map[:, Step_Index:Step_Index + self.frames_num, :]
        # get augmented map
        Spatial_aug_flag = 0
        Temporal_aug_flag = 0
        Step_Index_aug = Step_Index
        if self.args.spatial_aug_rate > 0:
            if (random.uniform(0, 100) / 100.0) < self.args.spatial_aug_rate:
                temp_ratio = (1.0 * random.uniform(0, 100) / 100.0)
                Index = np.arange(With)
                if temp_ratio < 0.3:
                    Index[random.randint(0, With - 1)] = random.randint(0, With - 1)
                    Index[random.randint(0, With - 1)] = random.randint(0, With - 1)
                    map_aug = map_ori[Index]
                elif temp_ratio < 0.6:
                    Index[random.randint(0, With - 1)] = random.randint(0, With - 1)
                    Index[random.randint(0, With - 1)] = random.randint(0, With - 1)
                    Index[random.randint(0, With - 1)] = random.randint(0, With - 1)
                    Index[random.randint(0, With - 1)] = random.randint(0, With - 1)
                    map_aug = map_ori[Index]
                elif temp_ratio < 0.9:
                    np.random.shuffle(Index[random.randint(0, With - 1):random.randint(0, With - 1)])
                    map_aug = map_ori[Index]
                else:
                    np.random.shuffle(Index)
                    map_aug = map_ori[Index]
                Spatial_aug_flag = 1
            else:
                map_aug = map_ori

        if ((Spatial_aug_flag == 0) and (self.args.temporal_aug_rate > 0)):
            if Step_Index + self.frames_num + 30 < Max_frame:
                if (random.uniform(0, 100) / 100.0) < self.args.temporal_aug_rate:
                    Step_Index_aug = int(random.uniform(0, 29) + Step_Index)
                    map_aug = feature_map[:, Step_Index_aug:Step_Index_aug + self.frames_num, :]
                    Temporal_aug_flag = 1
                else:
                    map_aug = map_ori
            else:
                map_aug = map_ori

        if ((Spatial_aug_flag == 0) and (Temporal_aug_flag == 0)):
            map_aug = map_ori

        if self.dataName in ['PURE', 'VIPL']:
            gt_aug, bvp_aug, sp_aug = self.getLabel(nowPath, Step_Index_aug)
        elif self.dataName in ['HCW', 'V4V']:
            gt_aug, bvp_aug, rf_aug = self.getLabel(nowPath, Step_Index_aug)
        else:
            gt_aug, bvp_aug = self.getLabel(nowPath, Step_Index_aug)
        # gt_aug, bvp_aug = self.getLabel(nowPath, Step_Index_aug)

        for c in range(map_ori.shape[2]):
            for r in range(map_ori.shape[0]):
                map_ori[r, :, c] = 255 * ((map_ori[r, :, c] - np.min(map_ori[r, :, c])) / \
                                          (0.00001 + np.max(map_ori[r, :, c]) - np.min(map_ori[r, :, c])))

        for c in range(map_aug.shape[2]):
            for r in range(map_aug.shape[0]):
                map_aug[r, :, c] = 255 * ((map_aug[r, :, c] - np.min(map_aug[r, :, c])) / \
                                          (0.00001 + np.max(map_aug[r, :, c]) - np.min(map_aug[r, :, c])))
        # if self.domain_label is not None:
        # domain_label = np.full((map_ori.shape[0], 1), np.inf)

        map_ori = Image.fromarray(np.uint8(map_ori))
        map_aug = Image.fromarray(np.uint8(map_aug))

        map_ori = self.transform(map_ori)
        map_aug = self.transform_aug(map_aug)

        if self.dataName in ['PURE', 'VIPL']:
            return (map_ori, bvp, gt, sp, gt, map_aug, bvp_aug, gt_aug, sp_aug, gt_aug, self.domain_label)
        elif self.dataName in ['HCW', 'V4V']:
            return (map_ori, bvp, gt, 0, rf, map_aug, bvp_aug, gt_aug, 0, rf_aug, self.domain_label)
        else:
            return (map_ori, bvp, gt, 0, gt, map_aug, bvp_aug, gt_aug, 0, gt_aug, self.domain_label)


def CrossValidation(root_dir, fold_num=5, fold_index=0):
    datalist = os.listdir(root_dir)
    # datalist.sort(key=lambda x: int(x))
    num = len(datalist)
    test_num = round(((num / fold_num) - 2))
    train_num = num - test_num
    test_index = datalist[fold_index * test_num:fold_index * test_num + test_num - 1]
    train_index = datalist[0:fold_index * test_num] + datalist[fold_index * test_num + test_num:]
    return test_index, train_index


def getIndex(root_path, filesList, save_path, Pic_path, Step, frames_num):
    Index_path = []
    print('Now processing' + root_path)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    for sub_file in filesList:
        now = os.path.join(root_path, sub_file)
        img_path = os.path.join(now, os.path.join('STMap', Pic_path))
        temp = cv2.imread(img_path)
        Num = temp.shape[1]
        Res = Num - frames_num - 1  # 可能是Diff数据
        Step_num = int(Res / Step)
        for i in range(Step_num):
            Step_Index = i * Step
            temp_path = sub_file + '_' + str(1000 + i) + '_.mat'
            scio.savemat(os.path.join(save_path, temp_path), {'Path': now, 'Step_Index': Step_Index})
            Index_path.append(temp_path)
    return Index_path


def calculate_respiration_rate(breathing_signal, sampling_rate=30):
    """
    Calculate the respiration rate from a breathing signal.

    :param breathing_signal: A 1-D numpy array of breathing signal data.
    :param sampling_rate: Sampling rate of the signal in Hz, default is 30Hz.
    :return: Respiration rate in breaths per minute.
    """
    peaks, _ = find_peaks(breathing_signal)
    num_of_breaths = len(peaks)
    duration_in_seconds = len(breathing_signal) / sampling_rate
    duration_in_minutes = duration_in_seconds / 60

    respiration_rate = num_of_breaths / duration_in_minutes
    return respiration_rate

# Example usage
# Replace 'your_breathing_signal_array' with your actual breathing signal data
# breathing_signal = np.array(your_breathing_signal_array)
# respiration_rate = calculate_respiration_rate(breathing_signal)
# print(f"Respiration Rate: {respiration_rate} breaths/minute")
