# -*- coding: UTF-8 -*-
import scipy.io as io
import torch
import torch.nn as nn
import numpy as np
import MyDataset
import MyLoss
import model
from torch.utils.data import DataLoader
from torch.autograd import Variable
# from thop import profile
# from basic_module import *
import utils
from datetime import datetime
import os
from utils import Logger, time_to_str
from timeit import default_timer as timer
import random
from tqdm import tqdm

import warnings

warnings.simplefilter('ignore')

TARGET_DOMAIN = {'VIPL': ['PURE', 'V4V', 'BUAA', 'UBFC', 'HCW'], \
                 'V4V': ['VIPL', 'PURE', 'BUAA', 'UBFC', 'HCW'], \
                 'PURE': ['VIPL', 'V4V', 'BUAA', 'UBFC', 'HCW'], \
                 'BUAA': ['VIPL', 'V4V', 'PURE', 'UBFC', 'HCW'], \
                 'UBFC': ['VIPL', 'V4V', 'PURE', 'BUAA', 'HCW'], \
                 'HCW': ['VIPL', 'V4V', 'PURE', 'BUAA', 'UBFC']}

FILEA_NAME = {'VIPL': ['VIPL', 'VIPL', 'STMap_RGB_Align_CSI'], \
              'V4V': ['V4V', 'V4V', 'STMap_RGB'], \
              'PURE': ['PURE', 'PURE', 'STMap'], \
              'BUAA': ['BUAA', 'BUAA', 'STMap_RGB'], \
              'UBFC': ['UBFC', 'UBFC', 'STMap'], \
              'HCW': ['HCW', 'HCW', 'STMap_RGB']}

if __name__ == '__main__':
    args = utils.get_args()
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    Source_domain_Names = TARGET_DOMAIN[args.tgt]
    root_file = r'/remote-home/hao.lu/Data/STMap/'
    # 参数
    File_Name_0 = FILEA_NAME[Source_domain_Names[0]]
    source_name_0 = Source_domain_Names[0]
    source_fileRoot_0 = root_file + File_Name_0[0]
    source_saveRoot_0 = root_file + 'STMap_Index/' + File_Name_0[1]
    source_map_0 = File_Name_0[2] + '.png'

    File_Name_1 = FILEA_NAME[Source_domain_Names[1]]
    source_name_1 = Source_domain_Names[1]
    source_fileRoot_1 = root_file + File_Name_1[0]
    source_saveRoot_1 = root_file + 'STMap_Index/' + File_Name_1[1]
    source_map_1 = File_Name_1[2] + '.png'

    File_Name_2 = FILEA_NAME[Source_domain_Names[2]]
    source_name_2 = Source_domain_Names[2]
    source_fileRoot_2 = root_file + File_Name_2[0]
    source_saveRoot_2 = root_file + 'STMap_Index/' + File_Name_2[1]
    source_map_2 = File_Name_2[2] + '.png'

    File_Name_3 = FILEA_NAME[Source_domain_Names[3]]
    source_name_3 = Source_domain_Names[3]
    source_fileRoot_3 = root_file + File_Name_3[0]
    source_saveRoot_3 = root_file + 'STMap_Index/' + File_Name_3[1]
    source_map_3 = File_Name_3[2] + '.png'

    File_Name_4 = FILEA_NAME[Source_domain_Names[4]]
    source_name_4 = Source_domain_Names[4]
    source_fileRoot_4 = root_file + File_Name_4[0]
    source_fileRoot_4 = source_fileRoot_4.replace('STMap/HCW', 'GPT-Chat/ChatGPT')
    source_saveRoot_4 = root_file + 'STMap_Index/' + File_Name_4[1]
    source_saveRoot_4 = source_saveRoot_4.replace('STMap/STMap_Index/HCW', 'GPT-Chat/STMap_Index/HCW')
    source_map_4 = File_Name_4[2] + '.png'

    FILE_Name = FILEA_NAME[args.tgt]
    Target_name = args.tgt
    Target_fileRoot = root_file + FILE_Name[0]
    Target_fileRoot = Target_fileRoot.replace('STMap/HCW', 'GPT-Chat/ChatGPT')
    Target_saveRoot = root_file + 'STMap_Index/' + FILE_Name[1]
    Target_saveRoot = Target_saveRoot.replace('STMap/STMap_Index/HCW', 'GPT-Chat/STMap_Index/HCW')
    Target_map = FILE_Name[2] + '.png'

    # 训练参数
    batch_size_num = args.batchsize
    epoch_num = args.epochs
    learning_rate = args.lr

    test_batch_size = args.batchsize
    num_workers = args.num_workers
    GPU = args.GPU

    # 图片参数
    input_form = args.form
    reTrain = args.reTrain
    frames_num = args.frames_num
    fold_num = args.fold_num
    fold_index = args.fold_index

    best_mae = 99

    print('batch num:', batch_size_num, ' epoch_num:', epoch_num, ' GPU Inedex:', GPU)
    print(' frames num:', frames_num, ' learning rate:', learning_rate, )
    print('fold num:', frames_num, ' fold index:', fold_index)

    if not os.path.exists('./Result_log'):
        os.makedirs('./Result_log')
    rPPGNet_name = 'rPPGNet_' + Target_name + 'Spatial' + str(args.spatial_aug_rate) + 'Temporal' + str(
        args.temporal_aug_rate)
    log = Logger()
    log.open('./Result_log/' + rPPGNet_name + '_log.txt', mode='a')
    log.write("\n----------------------------------------------- [START %s] %s\n\n" % (
        datetime.now().strftime('%Y-%m-%d %H:%M:%S'), '-' * 51))

    # 运行媒介
    if torch.cuda.is_available():
        device = torch.device('cuda:' + GPU if torch.cuda.is_available() else 'cpu')  #
        print('on GPU')
    else:
        print('on CPU')

    # 数据集
    if args.reData == 1:
        source_index_0 = os.listdir(source_fileRoot_0)
        source_index_1 = os.listdir(source_fileRoot_1)
        source_index_2 = os.listdir(source_fileRoot_2)
        source_index_3 = os.listdir(source_fileRoot_3)
        source_index_4 = os.listdir(source_fileRoot_4)
        Target_index = os.listdir(Target_fileRoot)

        source_Indexa_0 = MyDataset.getIndex(source_fileRoot_0, source_index_0, \
                                             source_saveRoot_0, source_map_0, 10, frames_num)
        source_Indexa_1 = MyDataset.getIndex(source_fileRoot_1, source_index_1, \
                                             source_saveRoot_1, source_map_1, 10, frames_num)
        source_Indexa_2 = MyDataset.getIndex(source_fileRoot_2, source_index_2, \
                                             source_saveRoot_2, source_map_2, 10, frames_num)
        source_Indexa_3 = MyDataset.getIndex(source_fileRoot_3, source_index_3, \
                                             source_saveRoot_3, source_map_3, 10, frames_num)
        source_Indexa_4 = MyDataset.getIndex(source_fileRoot_4, source_index_4, \
                                             source_saveRoot_4, source_map_4, 10, frames_num)
        Target_Indexa = MyDataset.getIndex(Target_fileRoot, Target_index, \
                                           Target_saveRoot, Target_map, 10, frames_num)

    source_db_0 = MyDataset.Data_DG(root_dir=source_saveRoot_0, dataName=source_name_0, \
                                    STMap=source_map_0, frames_num=frames_num, args=args, domain_label=0)
    source_db_1 = MyDataset.Data_DG(root_dir=source_saveRoot_1, dataName=source_name_1, \
                                    STMap=source_map_1, frames_num=frames_num, args=args, domain_label=1)
    source_db_2 = MyDataset.Data_DG(root_dir=source_saveRoot_2, dataName=source_name_2, \
                                    STMap=source_map_2, frames_num=frames_num, args=args, domain_label=2)
    source_db_3 = MyDataset.Data_DG(root_dir=source_saveRoot_3, dataName=source_name_3, \
                                    STMap=source_map_3, frames_num=frames_num, args=args, domain_label=3)
    source_db_4 = MyDataset.Data_DG(root_dir=source_saveRoot_4, dataName=source_name_4, \
                                    STMap=source_map_4, frames_num=frames_num, args=args, domain_label=4)
    Target_db = MyDataset.Data_DG(root_dir=Target_saveRoot, dataName=Target_name, \
                                  STMap=Target_map, frames_num=frames_num, args=args, domain_label=5)

    src_loader_0 = DataLoader(source_db_0, batch_size=batch_size_num, shuffle=True, num_workers=num_workers)
    src_loader_1 = DataLoader(source_db_1, batch_size=batch_size_num, shuffle=True, num_workers=num_workers)
    src_loader_2 = DataLoader(source_db_2, batch_size=batch_size_num, shuffle=True, num_workers=num_workers)
    src_loader_3 = DataLoader(source_db_3, batch_size=batch_size_num, shuffle=True, num_workers=num_workers)
    src_loader_4 = DataLoader(source_db_4, batch_size=batch_size_num, shuffle=True, num_workers=num_workers)
    tgt_loader = DataLoader(Target_db, batch_size=batch_size_num, shuffle=False, num_workers=num_workers)

    if 'resnet' in args.pt:
        my_model = model.BaseNet_CNN(pretrain=args.pt, gamma=args.r, lora_alpha=args.alpha)
    elif 'vit' in args.pt:
        my_model = model.BaseNet_ViT(gamma=args.r, lora_alpha=args.alpha)
    my_model.calculate_training_parameter_ratio()

    if reTrain == 1:
        my_model = torch.load('./Result_Model/rPPGNet_UBFCSpatial0.5Temporal0.1_12.608557239271615_3200_0.1_5.0',
                              map_location=device)
        print('load ' + rPPGNet_name + ' right')
    my_model.to(device=device)

    optimizer_rPPG = torch.optim.Adam(my_model.parameters(), lr=learning_rate)
    # optimizer_rPPG = torch.optim.Adam(rppg_params + other_params, lr=learning_rate)
    # optimizer_spo = torch.optim.Adam(spo_params + other_params, lr=learning_rate)

    loss_func_NP = MyLoss.P_loss3().to(device)
    loss_func_SPO = nn.SmoothL1Loss().to(device)
    loss_func_RF = nn.SmoothL1Loss().to(device)
    loss_func_L1 = nn.SmoothL1Loss().to(device)
    loss_merge = MyLoss.MergeLoss().to(device)
    loss_temporal = MyLoss.temporal_loss().to(device)
    loss_spatial = MyLoss.SpatialConsistencyLoss().to(device)
    loss_HB = MyLoss.bvp_hr_loss().to(device)
    loss_BR = MyLoss.bvp_rr_loss().to(device)
    loss_asp = MyLoss.Asp_loss().to(device)
    loss_func_SP = MyLoss.SP_loss(device, clip_length=frames_num).to(device)

    # loss_func_SP = MyLoss.SP_loss(device, clip_length=frames_num).to(device)
    src_iter_0 = src_loader_0.__iter__()
    src_iter_per_epoch_0 = len(src_iter_0)

    src_iter_1 = src_loader_1.__iter__()
    src_iter_per_epoch_1 = len(src_iter_1)

    src_iter_2 = src_loader_2.__iter__()
    src_iter_per_epoch_2 = len(src_iter_2)

    src_iter_3 = src_loader_3.__iter__()
    src_iter_per_epoch_3 = len(src_iter_3)

    src_iter_4 = src_loader_4.__iter__()
    src_iter_per_epoch_4 = len(src_iter_4)

    tgt_iter = iter(tgt_loader)
    tgt_iter_per_epoch = len(tgt_iter)

    max_iter = args.max_iter
    start = timer()
    loss_res = []
    with tqdm(range(max_iter + 1)) as it:
        for iter_num in it:
            my_model.train()
            if (iter_num % src_iter_per_epoch_0 == 0):
                src_iter_0 = src_loader_0.__iter__()
            if (iter_num % src_iter_per_epoch_1 == 0):
                src_iter_1 = src_loader_1.__iter__()
            if (iter_num % src_iter_per_epoch_2 == 0):
                src_iter_2 = src_loader_2.__iter__()
            if (iter_num % src_iter_per_epoch_3 == 0):
                src_iter_3 = src_loader_3.__iter__()
            if (iter_num % src_iter_per_epoch_4 == 0):
                src_iter_4 = src_loader_4.__iter__()

            ######### data prepare #########
            data0, bvp0, HR_rel0, spo_rel0, rf_rel0, data_aug0, bvp_aug0, HR_rel_aug0, spo_rel_aug0, rf_rel_aug0, domain_label0 = src_iter_0.__next__()
            data1, bvp1, HR_rel1, spo_rel1, rf_rel1, data_aug1, bvp_aug1, HR_rel_aug1, spo_rel_aug1, rf_rel_aug1, domain_label1 = src_iter_1.__next__()
            data2, bvp2, HR_rel2, spo_rel2, rf_rel2, data_aug2, bvp_aug2, HR_rel_aug2, spo_rel_aug2, rf_rel_aug2, domain_label2 = src_iter_2.__next__()
            data3, bvp3, HR_rel3, spo_rel3, rf_rel3, data_aug3, bvp_aug3, HR_rel_aug3, spo_rel_aug3, rf_rel_aug3, domain_label3 = src_iter_3.__next__()
            data4, bvp4, HR_rel4, spo_rel4, rf_rel4, data_aug4, bvp_aug4, HR_rel_aug4, spo_rel_aug4, rf_rel_aug4, domain_label4 = src_iter_4.__next__()

            data0 = Variable(data0).float().to(device=device)
            bvp0 = Variable(bvp0).float().to(device=device).unsqueeze(dim=1)
            HR_rel0 = Variable(torch.Tensor(HR_rel0)).float().to(device=device)
            spo_rel0 = Variable(torch.Tensor(spo_rel0)).float().to(device=device)
            rf_rel0 = Variable(torch.Tensor(rf_rel0)).float().to(device=device)

            data_aug0 = Variable(data_aug0).float().to(device=device)
            bvp_aug0 = Variable(bvp_aug0).float().to(device=device).unsqueeze(dim=1)
            HR_rel_aug0 = Variable(torch.Tensor(HR_rel_aug0)).float().to(device=device)
            spo_rel_aug0 = Variable(torch.Tensor(spo_rel_aug0)).float().to(device=device)
            rf_rel_aug0 = Variable(torch.Tensor(rf_rel_aug0)).float().to(device=device)
            # domain_label0 = domain_label0.long().to(device)

            data1 = Variable(data1).float().to(device=device)
            bvp1 = Variable((bvp1)).float().to(device=device).unsqueeze(dim=1)
            HR_rel1 = Variable(torch.Tensor(HR_rel1)).float().to(device=device)
            spo_rel1 = Variable(torch.Tensor(spo_rel1)).float().to(device=device)
            rf_rel1 = Variable(torch.Tensor(rf_rel1)).float().to(device=device)

            data_aug1 = Variable(data_aug1).float().to(device=device)
            bvp_aug1 = Variable((bvp_aug1)).float().to(device=device).unsqueeze(dim=1)
            HR_rel_aug1 = Variable(torch.Tensor(HR_rel_aug1)).float().to(device=device)
            spo_rel_aug1 = Variable(torch.Tensor(spo_rel_aug1)).float().to(device=device)
            rf_rel_aug1 = Variable(torch.Tensor(rf_rel_aug1)).float().to(device=device)
            # domain_label1 = domain_label1.long().to(device)

            data2 = Variable(data2).float().to(device=device)
            bvp2 = Variable((bvp2)).float().to(device=device).unsqueeze(dim=1)
            HR_rel2 = Variable(torch.Tensor(HR_rel2)).float().to(device=device)
            spo_rel2 = Variable(torch.Tensor(spo_rel2)).float().to(device=device)
            rf_rel2 = Variable(torch.Tensor(rf_rel2)).float().to(device=device)

            data_aug2 = Variable(data_aug2).float().to(device=device)
            bvp_aug2 = Variable((bvp_aug2)).float().to(device=device).unsqueeze(dim=1)
            HR_rel_aug2 = Variable(torch.Tensor(HR_rel_aug2)).float().to(device=device)
            spo_rel_aug2 = Variable(torch.Tensor(spo_rel_aug2)).float().to(device=device)
            rf_rel_aug2 = Variable(torch.Tensor(rf_rel_aug2)).float().to(device=device)
            # domain_label2 = domain_label2.long().to(device)

            data3 = Variable(data3).float().to(device=device)
            bvp3 = Variable((bvp3)).float().to(device=device).unsqueeze(dim=1)
            HR_rel3 = Variable(torch.Tensor(HR_rel3)).float().to(device=device)
            spo_rel3 = Variable(torch.Tensor(spo_rel3)).float().to(device=device)
            rf_rel3 = Variable(torch.Tensor(rf_rel3)).float().to(device=device)

            data_aug3 = Variable(data_aug3).float().to(device=device)
            bvp_aug3 = Variable((bvp_aug3)).float().to(device=device).unsqueeze(dim=1)
            HR_rel_aug3 = Variable(torch.Tensor(HR_rel_aug3)).float().to(device=device)
            spo_rel_aug3 = Variable(torch.Tensor(spo_rel_aug3)).float().to(device=device)
            rf_rel_aug3 = Variable(torch.Tensor(rf_rel_aug3)).float().to(device=device)
            # domain_label3 = domain_label3.long().to(device)

            data4 = Variable(data4).float().to(device=device)
            bvp4 = Variable((bvp4)).float().to(device=device).unsqueeze(dim=1)
            HR_rel4 = Variable(torch.Tensor(HR_rel4)).float().to(device=device)
            spo_rel4 = Variable(torch.Tensor(spo_rel4)).float().to(device=device)
            rf_rel4 = Variable(torch.Tensor(rf_rel4)).float().to(device=device)

            data_aug4 = Variable(data_aug4).float().to(device=device)
            bvp_aug4 = Variable((bvp_aug4)).float().to(device=device).unsqueeze(dim=1)
            HR_rel_aug4 = Variable(torch.Tensor(HR_rel_aug4)).float().to(device=device)
            spo_rel_aug4 = Variable(torch.Tensor(spo_rel_aug4)).float().to(device=device)
            rf_rel_aug4 = Variable(torch.Tensor(rf_rel_aug4)).float().to(device=device)
            # domain_label4 = domain_label4.long().to(device)

            optimizer_rPPG.zero_grad()
            # optimizer_spo.zero_grad()
            d_b0, d_b1, d_b2, d_b3, d_b4 = data0.shape[0], data1.shape[0], data2.shape[0], data3.shape[0], data4.shape[
                0]

            input = torch.cat([data0, data1, data2, data3, data4], dim=0)
            input_aug = torch.cat([data_aug0, data_aug1, data_aug2, data_aug3, data_aug4], dim=0)

            bvp_pre_zip, HR_pr_zip, spo_pre_zip, rf_pre_zip, feat, feat_spo = my_model(input)

            spatial_loss = 0
            for i in feat:
                spatial_loss += loss_spatial(i)

            bvp_pre_aug_zip, HR_pr_aug_zip, spo_pre_aug_zip, rf_pre_aug_zip, feat_aug, feat_spo_aug = my_model(input_aug)
            for i in feat_aug:
                spatial_loss += loss_spatial(i)

            bvp_pre0_zip, bvp_pre1_zip, bvp_pre2_zip, bvp_pre3_zip, bvp_pre4_zip = bvp_pre_zip[0:d_b0], bvp_pre_zip[
                                                                                                        d_b0:d_b0 + d_b1], \
                bvp_pre_zip[d_b0 + d_b1:d_b0 + d_b1 + d_b2], bvp_pre_zip[
                                                             d_b0 + d_b1 + d_b2:d_b0 + d_b1 + d_b2 + d_b3], bvp_pre_zip[
                                                                                                            d_b0 + d_b1 + d_b2 + d_b3:]
            HR_pr0_zip, HR_pr1_zip, HR_pr2_zip, HR_pr3_zip, HR_pr4_zip = HR_pr_zip[0:d_b0], HR_pr_zip[d_b0:d_b0 + d_b1], \
                HR_pr_zip[d_b0 + d_b1:d_b0 + d_b1 + d_b2], HR_pr_zip[
                                                           d_b0 + d_b1 + d_b2:d_b0 + d_b1 + d_b2 + d_b3], HR_pr_zip[
                                                                                                          d_b0 + d_b1 + d_b2 + d_b3:]
            spo_pre0_zip, spo_pre1_zip, spo_pre2_zip, spo_pre3_zip, spo_pre4_zip = spo_pre_zip[0:d_b0], spo_pre_zip[
                                                                                                        d_b0:d_b0 + d_b1], \
                spo_pre_zip[d_b0 + d_b1:d_b0 + d_b1 + d_b2], spo_pre_zip[
                                                             d_b0 + d_b1 + d_b2:d_b0 + d_b1 + d_b2 + d_b3], spo_pre_zip[
                                                                                                            d_b0 + d_b1 + d_b2 + d_b3:]
            rf_pre0_zip, rf_pre1_zip, rf_pre2_zip, rf_pre3_zip, rf_pre4_zip = rf_pre_zip[0:d_b0], rf_pre_zip[
                                                                                                  d_b0:d_b0 + d_b1], \
                rf_pre_zip[d_b0 + d_b1:d_b0 + d_b1 + d_b2], rf_pre_zip[
                                                            d_b0 + d_b1 + d_b2:d_b0 + d_b1 + d_b2 + d_b3], rf_pre_zip[
                                                                                                           d_b0 + d_b1 + d_b2 + d_b3:]

            bvp_pre_aug0_zip, bvp_pre_aug1_zip, bvp_pre_aug2_zip, bvp_pre_aug3_zip, bvp_pre_aug4_zip = bvp_pre_aug_zip[
                                                                                                       0:d_b0], bvp_pre_aug_zip[
                                                                                                                d_b0:d_b0 + d_b1], \
                bvp_pre_aug_zip[d_b0 + d_b1:d_b0 + d_b1 + d_b2], bvp_pre_aug_zip[
                                                                 d_b0 + d_b1 + d_b2:d_b0 + d_b1 + d_b2 + d_b3], bvp_pre_aug_zip[
                                                                                                                d_b0 + d_b1 + d_b2 + d_b3:]
            HR_pr_aug0_zip, HR_pr_aug1_zip, HR_pr_aug2_zip, HR_pr_aug3_zip, HR_pr_aug4_zip = HR_pr_aug_zip[
                                                                                             0:d_b0], HR_pr_aug_zip[
                                                                                                      d_b0:d_b0 + d_b1], \
                HR_pr_aug_zip[d_b0 + d_b1:d_b0 + d_b1 + d_b2], HR_pr_aug_zip[
                                                               d_b0 + d_b1 + d_b2:d_b0 + d_b1 + d_b2 + d_b3], HR_pr_aug_zip[
                                                                                                              d_b0 + d_b1 + d_b2 + d_b3:]
            spo_pre_aug0_zip, spo_pre_aug1_zip, spo_pre_aug2_zip, spo_pre_aug3_zip, spo_pre_aug4_zip = spo_pre_aug_zip[
                                                                                                       0:d_b0], spo_pre_aug_zip[
                                                                                                                d_b0:d_b0 + d_b1], \
                spo_pre_aug_zip[d_b0 + d_b1:d_b0 + d_b1 + d_b2], spo_pre_aug_zip[
                                                                 d_b0 + d_b1 + d_b2:d_b0 + d_b1 + d_b2 + d_b3], spo_pre_aug_zip[
                                                                                                                d_b0 + d_b1 + d_b2 + d_b3:]
            rf_pre_aug0_zip, rf_pre_aug1_zip, rf_pre_aug2_zip, rf_pre_aug3_zip, rf_pre_aug4_zip = rf_pre_aug_zip[
                                                                                                  0:d_b0], rf_pre_aug_zip[
                                                                                                           d_b0:d_b0 + d_b1], \
                rf_pre_aug_zip[d_b0 + d_b1:d_b0 + d_b1 + d_b2], rf_pre_aug_zip[
                                                                d_b0 + d_b1 + d_b2:d_b0 + d_b1 + d_b2 + d_b3], rf_pre_aug_zip[
                                                                                                               d_b0 + d_b1 + d_b2 + d_b3:]

            rppg_loss_0 = MyLoss.get_loss(bvp_pre0_zip, HR_pr0_zip, bvp0, HR_rel0, source_name_0, \
                                           loss_func_NP, loss_func_L1, args, iter_num, spo_pre=spo_pre0_zip,
                                           spo_gt=spo_rel0,
                                           loss_spo=loss_func_SPO, rf_pre=rf_pre0_zip,
                                           rf_gt=rf_rel0,
                                           loss_rf=loss_func_RF)
            rppg_loss_1 = MyLoss.get_loss(bvp_pre1_zip, HR_pr1_zip, bvp1, HR_rel1, source_name_1, \
                                           loss_func_NP, loss_func_L1, args, iter_num, spo_pre=spo_pre1_zip,
                                           spo_gt=spo_rel1,
                                           loss_spo=loss_func_SPO, rf_pre=rf_pre1_zip,
                                           rf_gt=rf_rel1,
                                           loss_rf=loss_func_RF)
            rppg_loss_2 = MyLoss.get_loss(bvp_pre2_zip, HR_pr2_zip, bvp2, HR_rel2, source_name_2, \
                                           loss_func_NP, loss_func_L1, args, iter_num, spo_pre=spo_pre2_zip,
                                           spo_gt=spo_rel2,
                                           loss_spo=loss_func_SPO, rf_pre=rf_pre2_zip,
                                           rf_gt=rf_rel2,
                                           loss_rf=loss_func_RF)
            rppg_loss_3 = MyLoss.get_loss(bvp_pre3_zip, HR_pr3_zip, bvp3, HR_rel3, source_name_3, \
                                           loss_func_NP, loss_func_L1, args, iter_num, spo_pre=spo_pre3_zip,
                                           spo_gt=spo_rel3,
                                           loss_spo=loss_func_SPO, rf_pre=rf_pre3_zip,
                                           rf_gt=rf_rel3,
                                           loss_rf=loss_func_RF)
            rppg_loss_4 = MyLoss.get_loss(bvp_pre4_zip, HR_pr4_zip, bvp4, HR_rel4, source_name_4, \
                                           loss_func_NP, loss_func_L1, args, iter_num, spo_pre=spo_pre4_zip,
                                           spo_gt=spo_rel4,
                                           loss_spo=loss_func_SPO, rf_pre=rf_pre4_zip,
                                           rf_gt=rf_rel4,
                                           loss_rf=loss_func_RF)

            rppg_loss_aug_0 = MyLoss.get_loss(bvp_pre_aug0_zip, HR_pr_aug0_zip, bvp_aug0, HR_rel_aug0,
                                               source_name_0, \
                                               loss_func_NP, loss_func_L1, args, iter_num,
                                               spo_pre=spo_pre_aug0_zip,
                                               spo_gt=spo_rel_aug0, loss_spo=loss_func_SPO, rf_pre=rf_pre_aug0_zip,
                                               rf_gt=rf_rel_aug0,
                                               loss_rf=loss_func_RF)
            rppg_loss_aug_1 = MyLoss.get_loss(bvp_pre_aug1_zip, HR_pr_aug1_zip, bvp_aug1, HR_rel_aug1,
                                               source_name_1, \
                                               loss_func_NP, loss_func_L1, args, iter_num,
                                               spo_pre=spo_pre_aug1_zip,
                                               spo_gt=spo_rel_aug1, loss_spo=loss_func_SPO, rf_pre=rf_pre_aug1_zip,
                                               rf_gt=rf_rel_aug1,
                                               loss_rf=loss_func_RF)
            rppg_loss_aug_2 = MyLoss.get_loss(bvp_pre_aug2_zip, HR_pr_aug2_zip, bvp_aug2, HR_rel_aug2,
                                               source_name_2, \
                                               loss_func_NP, loss_func_L1, args, iter_num,
                                               spo_pre=spo_pre_aug2_zip,
                                               spo_gt=spo_rel_aug2, loss_spo=loss_func_SPO, rf_pre=rf_pre_aug2_zip,
                                               rf_gt=rf_rel_aug2,
                                               loss_rf=loss_func_RF)
            rppg_loss_aug_3 = MyLoss.get_loss(bvp_pre_aug3_zip, HR_pr_aug3_zip, bvp_aug3, HR_rel_aug3,
                                               source_name_3, \
                                               loss_func_NP, loss_func_L1, args, iter_num,
                                               spo_pre=spo_pre_aug3_zip,
                                               spo_gt=spo_rel_aug3, loss_spo=loss_func_SPO, rf_pre=rf_pre_aug3_zip,
                                               rf_gt=rf_rel_aug3,
                                               loss_rf=loss_func_RF)
            rppg_loss_aug_4 = MyLoss.get_loss(bvp_pre_aug4_zip, HR_pr_aug4_zip, bvp_aug4, HR_rel_aug4,
                                               source_name_4, \
                                               loss_func_NP, loss_func_L1, args, iter_num,
                                               spo_pre=spo_pre_aug4_zip,
                                               spo_gt=spo_rel_aug4, loss_spo=loss_func_SPO, rf_pre=rf_pre_aug4_zip,
                                               rf_gt=rf_rel_aug4,
                                               loss_rf=loss_func_RF)

            w = my_model.get_task_weights()
            loss_m = 0
            for i in range(len(w)):
                for j in range(len(w[0])):
                    for k in range(j, len(w[0])):
                        loss_m += loss_merge(w[i][j], w[i][k], device)

            temporal_loss = loss_temporal(HR_pr_zip, HR_pr_aug_zip) + loss_temporal(spo_pre_zip, spo_pre_aug_zip) \
                            + loss_temporal(rf_pre_zip, rf_pre_aug_zip)
            HB_loss_0 = loss_HB(bvp_pre0_zip, HR_rel0) + loss_HB(bvp_pre_aug0_zip, HR_rel_aug0)
            HB_loss_1 = loss_HB(bvp_pre1_zip, HR_rel1) + loss_HB(bvp_pre_aug1_zip, HR_rel_aug1)
            HB_loss_2 = loss_HB(bvp_pre2_zip, HR_rel2) + loss_HB(bvp_pre_aug2_zip, HR_rel_aug2)
            HB_loss_3 = loss_HB(bvp_pre3_zip, HR_rel3) + loss_HB(bvp_pre_aug3_zip, HR_rel_aug3)
            HB_loss_4 = loss_HB(bvp_pre4_zip, HR_rel4) + loss_HB(bvp_pre_aug4_zip, HR_rel_aug4)
            HB_loss = (HB_loss_0 + HB_loss_1 + HB_loss_2 + HB_loss_3 + HB_loss_4) / 5

            BR_loss_0 = loss_BR(bvp_pre0_zip, rf_rel0) + loss_BR(bvp_pre_aug0_zip, rf_rel_aug0)
            BR_loss_1 = loss_BR(bvp_pre1_zip, rf_rel1) + loss_BR(bvp_pre_aug1_zip, rf_rel_aug1)
            BR_loss_2 = loss_BR(bvp_pre2_zip, rf_rel2) + loss_BR(bvp_pre_aug2_zip, rf_rel_aug2)
            BR_loss_3 = loss_BR(bvp_pre3_zip, rf_rel3) + loss_BR(bvp_pre_aug3_zip, rf_rel_aug3)
            BR_loss_4 = loss_BR(bvp_pre4_zip, rf_rel4) + loss_BR(bvp_pre_aug4_zip, rf_rel_aug4)
            BR_loss = (BR_loss_0 + BR_loss_1 + BR_loss_2 + BR_loss_3 + BR_loss_4) / 5

            rppg_loss = (rppg_loss_0 + rppg_loss_1 + rppg_loss_2 + rppg_loss_3 + rppg_loss_4) \
                        + (rppg_loss_aug_0 + rppg_loss_aug_1 + rppg_loss_aug_2 + rppg_loss_aug_3 + rppg_loss_aug_4)

            if spo_rel0[0] != 0:
                feat_prio, spo_prio = loss_asp.generate_aug(feat_spo[0:d_b0], spo_rel0)
                feat_prio_aug, spo_prio_aug = loss_asp.generate_aug(feat_spo_aug[0:d_b0], spo_rel_aug0)
                asp_loss = loss_asp(my_model.predict_spo(feat_prio), spo_prio) + loss_asp(
                    my_model.predict_spo(feat_prio_aug), spo_prio_aug)
            else:
                asp_loss = torch.tensor(0).to(device)

            k = 2.0 / (1.0 + np.exp(-10.0 * iter_num / args.max_iter)) - 1.0
            loss_all = rppg_loss + args.w2 * k * spatial_loss + args.lemda / 2 * k * loss_m \
                       + 0.0001 * k * HB_loss + 0.0001 * k * BR_loss + 0.0001 * k * asp_loss #+ 0.01 * k * (loss_CD + loss_CD_aug) / 2

            if torch.sum(torch.isnan(rppg_loss)) > 0:
                print('Nan')
                continue
            else:
                rppg_loss.backward()
                optimizer_rPPG.step()

                it.set_postfix(
                    ordered_dict={
                        "Train Inter": iter_num,
                        "rppg loss": rppg_loss.data.cpu().numpy(),
                        # "spo loss": spo_loss.data.cpu().numpy()
                    },
                    refresh=False,
                )
            log.write(
                'Train Inter:' + str(iter_num) \
                + ' | Overall Loss:  ' + str(loss_all.data.cpu().numpy()) \
                + ' | rppg Loss:  ' + str(rppg_loss.data.cpu().numpy()) \
                + ' | Temporal Loss' + ' : ' + str((args.w1 * k * temporal_loss).data.cpu().numpy()) \
                + ' | Spatial Loss' + ' : ' + str((args.w2 * k * (spatial_loss)).data.cpu().numpy()) \
                + ' | Merge Loss' + ' : ' + str((args.lemda / 2 * k * loss_m).data.cpu().numpy()) \
                #+ ' | CD Loss' + ' : ' + str((0.01 * k * (loss_CD + loss_CD_aug) / 2).data.cpu().numpy()) \
                + ' | HB Loss' + ' : ' + str((0.0001 * k * HB_loss).data.cpu().numpy()) \
                + ' | BR Loss' + ' : ' + str((0.0001 * k * BR_loss).data.cpu().numpy()) \
                + ' | Asp Loss' + ' : ' + str((0.0001 * k * asp_loss).data.cpu().numpy()) \
                + ' |' + time_to_str(timer() - start, 'min'))
            log.write('\n')

            if (iter_num > 0) and (iter_num % 1000 == 0):
                # 测试
                print('Test:\n')
                my_model.eval()
                loss_mean = []
                Label_pr = []
                Label_gt = []
                HR_pr_temp = []
                HR_rel_temp = []
                HR_pr2_temp = []
                BVP_ALL = []
                BVP_PR_ALL = []
                Spo_pr_temp = []
                Spo_rel_temp = []
                RF_pr_temp = []
                RF_rel_temp = []
                for step, (data, bvp, HR_rel, spo, rf, _, _, _, _, _, _) in tqdm(enumerate(tgt_loader)):
                    data = Variable(data).float().to(device=device)
                    bvp = Variable(bvp).float().to(device=device)
                    HR_rel = Variable(HR_rel).float().to(device=device)
                    Spo_rel = Variable(spo).float().to(device=device)
                    RF_rel = Variable(rf).float().to(device=device)
                    bvp = bvp.unsqueeze(dim=1)
                    Wave = bvp
                    rand_idx = torch.randperm(data.shape[0])
                    Wave_pr, HR_pr, Spo_pr, RF_pr, _, _ = my_model(data)

                    if Target_name in ['VIPL', 'PURE']:
                        HR_pr_temp.extend(HR_pr.data.cpu().numpy())
                        HR_rel_temp.extend(HR_rel.data.cpu().numpy())
                        Spo_pr_temp.extend(Spo_pr.data.cpu().numpy())
                        Spo_rel_temp.extend(Spo_rel.data.cpu().numpy())
                        BVP_ALL.extend(Wave.data.cpu().numpy())
                        BVP_PR_ALL.extend(Wave_pr.data.cpu().numpy())
                    elif Target_name in ['V4V', 'HCW']:
                        HR_pr_temp.extend(HR_pr.data.cpu().numpy())
                        HR_rel_temp.extend(HR_rel.data.cpu().numpy())
                        RF_pr_temp.extend(RF_pr.data.cpu().numpy())
                        RF_rel_temp.extend(RF_rel.data.cpu().numpy())
                        BVP_ALL.extend(Wave.data.cpu().numpy())
                        BVP_PR_ALL.extend(Wave_pr.data.cpu().numpy())
                    else:
                        temp, HR_rel = loss_func_SP(Wave, HR_rel)
                        HR_rel_temp.extend(HR_rel.data.cpu().numpy())
                        temp, HR_pr = loss_func_SP(Wave_pr, HR_pr)
                        HR_pr_temp.extend(HR_pr.data.cpu().numpy())
                        BVP_ALL.extend(Wave.data.cpu().numpy())
                        BVP_PR_ALL.extend(Wave_pr.data.cpu().numpy())

                # print('HR:')
                ME, STD, MAE, RMSE, MER, P = utils.MyEval(HR_pr_temp, HR_rel_temp)
                log.write(
                    'DG' + args.tgt + ' Test Inter HR:' + str(iter_num) \
                    + ' | ME:  ' + str(ME) \
                    + ' | STD: ' + str(STD) \
                    + ' | MAE: ' + str(MAE) \
                    + ' | RMSE: ' + str(RMSE) \
                    + ' | MER: ' + str(MER) \
                    + ' | P ' + str(P))
                log.write('\n')
                ME, STD, MAE, RMSE, MER, P = utils.MyEval_bvp_hr(BVP_PR_ALL, BVP_ALL)
                log.write(
                        'Test Inter HR from BVP:' + str(iter_num) \
                        + ' | ME:  ' + str(ME) \
                        + ' | STD: ' + str(STD) \
                        + ' | MAE: ' + str(MAE) \
                        + ' | RMSE: ' + str(RMSE) \
                        + ' | MER: ' + str(MER) \
                        + ' | P ' + str(P))
                log.write('\n')
                ME, STD, MAE, RMSE, MER, P = utils.MyEval_bvp_rr(BVP_PR_ALL, BVP_ALL)
                log.write(
                        'Test Inter Resp from BVP:' + str(iter_num) \
                        + ' | ME:  ' + str(ME) \
                        + ' | STD: ' + str(STD) \
                        + ' | MAE: ' + str(MAE) \
                        + ' | RMSE: ' + str(RMSE) \
                        + ' | MER: ' + str(MER) \
                        + ' | P ' + str(P))
                log.write('\n')
                if Target_name in ['VIPL', 'PURE']:
                    ME, STD, MAE, RMSE, MER, P = utils.MyEval(Spo_pr_temp, Spo_rel_temp)
                    log.write(
                        'Test Inter SPO2:' + str(iter_num) \
                        + ' | ME:  ' + str(ME) \
                        + ' | STD: ' + str(STD) \
                        + ' | MAE: ' + str(MAE) \
                        + ' | RMSE: ' + str(RMSE) \
                        + ' | MER: ' + str(MER) \
                        + ' | P ' + str(P))
                    log.write('\n')
                if Target_name in ['V4V', 'HCW']:
                    ME, STD, MAE, RMSE, MER, P = utils.MyEval(RF_pr_temp, RF_rel_temp)
                    log.write(
                        'Test Inter Resp:' + str(iter_num) \
                        + ' | ME:  ' + str(ME) \
                        + ' | STD: ' + str(STD) \
                        + ' | MAE: ' + str(MAE) \
                        + ' | RMSE: ' + str(RMSE) \
                        + ' | MER: ' + str(MER) \
                        + ' | P ' + str(P))
                    log.write('\n')

                if not os.path.exists('./Result_Model'):
                    os.makedirs('./Result_Model')

                if not os.path.exists('./Result'):
                    os.makedirs('./Result')
                io.savemat('./Result/' + Target_name + '_' + str(iter_num) + '_' + str(
                    args.alpha) + '_' + str(args.r) + '_' + str(args.pt) + '_' + str(
                    args.lemda) + '_DG' + '_HR_pr.mat', {'HR_pr': HR_pr_temp})
                io.savemat('./Result/' + Target_name + '_' + str(iter_num) + '_' + str(
                    args.alpha) + '_' + str(args.r) + '_' + str(args.pt) + '_' + str(
                    args.lemda) + '_DG' + '_HR_rel.mat',
                           {'HR_rel': HR_rel_temp})
                if Target_name in ['VIPL', 'PURE']:
                    io.savemat('./Result/' + Target_name + '_' + str(iter_num) + '_' + str(
                        args.alpha) + '_' + str(args.r) + '_' + str(args.pt) + '_' + str(
                        args.lemda) + '_DG' + '_SPO_pr.mat',
                               {'SPO_pr': Spo_pr_temp})
                    io.savemat('./Result/' + Target_name + '_' + str(iter_num) + '_' + str(
                        args.alpha) + '_' + str(args.r) + '_' + str(args.pt) + '_' + str(
                        args.lemda) + '_DG' + '_SPO_rel.mat',
                               {'SPO_rel': Spo_rel_temp})
                elif Target_name in ['HCW', 'V4V']:
                    io.savemat('./Result/' + Target_name + '_' + str(iter_num) + '_' + str(
                        args.alpha) + '_' + str(args.r) + '_' + str(args.pt) + '_' + str(
                        args.lemda) + '_DG' + '_RF_pr.mat',
                               {'RF_pr': RF_pr_temp})
                    io.savemat('./Result/' + Target_name + '_' + str(iter_num) + '_' + str(
                        args.alpha) + '_' + str(args.r) + '_' + str(args.pt) + '_' + str(
                        args.lemda) + '_DG' + '_RF_rel.mat',
                               {'RF_rel': RF_rel_temp})
                io.savemat('./Result/' + Target_name + '_' + str(iter_num) + '_' + str(
                    args.alpha) + '_' + str(args.r) + '_' + str(args.pt) + '_' + str(
                    args.lemda) + '_DG' + '_WAVE_ALL.mat',
                           {'Wave': BVP_ALL})
                io.savemat('./Result/' + Target_name + '_' + str(iter_num) + '_' + str(
                    args.alpha) + '_' + str(args.r) + '_' + str(args.pt) + '_' + str(
                    args.lemda) + '_DG' + '_WAVE_PR_ALL.mat',
                           {'Wave': BVP_PR_ALL})
                torch.save(my_model,
                           './Result_Model/' + Target_name + '_' + str(iter_num) + '_' + str(
                               args.alpha) + '_' + str(args.r) + '_' + str(args.pt) + '_' + str(args.lemda) + '_DG')
                print('saveModel As ' + Target_name + '_' + str(iter_num) + '_' + str(
                    args.alpha) + '_' + str(args.r) + '_' + str(args.pt) + '_' + str(args.lemda) + '_DG')
