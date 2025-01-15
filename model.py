# -*- coding: UTF-8 -*-
import torch
import sys
from torchvision import models
import numpy as np
import utils
import models_vit
# from ZipLora import *
from MoE.Lora import *

np.set_printoptions(threshold=np.inf)
sys.path.append('..')
args = utils.get_args()


class BasicBlock(nn.Module):
    def __init__(self, inplanes, out_planes, stride=2, downsample=1, Res=0, islast=False):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(inplanes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_planes),
        )
        if downsample == 1:
            self.down = nn.Sequential(
                nn.Conv2d(inplanes, out_planes, kernel_size=1, stride=stride, padding=0, bias=False),
                nn.BatchNorm2d(out_planes)
            )
        self.downsample = downsample
        self.Res = Res
        self.islast = islast

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        if self.Res == 1:
            if self.downsample == 1:
                x = self.down(x)
            out += x
        if self.islast:
            return out
        else:
            return F.relu(out)


class BaseNet_CNN(nn.Module):
    def __init__(self, pretrain='resnet18', gamma=8, lora_alpha=16):
        super(BaseNet_CNN, self).__init__()
        if pretrain == 'resnet18':
            self.resnet = models.resnet18(pretrained=False)
            self.resnet.load_state_dict(torch.load('./pre_encoder/resnet18-5c106cde.pth'))

            self.add_adapter(MLoraConv2d, gamma=gamma, lora_alpha=lora_alpha)
            self.freeze_model(True)

            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.hr = nn.Linear(512, 1)
            self.spo = nn.Linear(512, 1)
            self.rf = nn.Linear(512, 1)

            self.up1_bvp = nn.Sequential(
                nn.ConvTranspose2d(512, 512, kernel_size=[1, 2], stride=[1, 2]),
                BasicBlock(512, 256, [2, 1], downsample=1),
            )
            self.up2_bvp = nn.Sequential(
                nn.ConvTranspose2d(256, 256, kernel_size=[1, 2], stride=[1, 2]),
                BasicBlock(256, 64, [1, 1], downsample=1),
            )
            self.up3_bvp = nn.Sequential(
                nn.ConvTranspose2d(64, 64, kernel_size=[1, 2], stride=[1, 2]),
                BasicBlock(64, 32, [2, 1], downsample=1),
            )
            self.up4_bvp = nn.Sequential(
                nn.ConvTranspose2d(32, 32, kernel_size=[1, 2], stride=[1, 2]),
                BasicBlock(32, 1, [1, 1], downsample=1, islast=True),
            )

            self.gate_hr = nn.Sequential(
                nn.Conv2d(512, 512, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
                nn.Conv2d(512, 512, kernel_size=1, stride=1, padding=0, bias=False),
                nn.Dropout2d(0.1),
                nn.BatchNorm2d(512),
                nn.Sigmoid()
            )

            self.gate_spo = nn.Sequential(
                nn.Conv2d(512, 512, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
                nn.Conv2d(512, 512, kernel_size=1, stride=1, padding=0, bias=False),
                nn.Dropout2d(0.1),
                nn.BatchNorm2d(512),
                nn.Sigmoid()
            )

            self.gate_rf = nn.Sequential(
                nn.Conv2d(512, 512, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
                nn.Conv2d(512, 512, kernel_size=1, stride=1, padding=0, bias=False),
                nn.Dropout2d(0.1),
                nn.BatchNorm2d(512),
                nn.Sigmoid()
            )


        elif pretrain == 'resnet50':
            self.resnet = models.resnet50(pretrained=False)
            self.resnet.load_state_dict(torch.load('./pre_encoder/resnet50-19c8e357.pth'))

            self.add_adapter(LoraConv2d, gamma=gamma, lora_alpha=lora_alpha)
            self.freeze_model(True)

            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.hr = nn.Linear(2048, 1)
            self.spo = nn.Linear(2048, 1)
            # For Sig
            # 模仿以下的结构，将[batch_size, 2048, 4, 16]的特征图转换为[batch_size, 1, 256, 1]的特征图
            self.up1_bvp = nn.Sequential(
                nn.ConvTranspose2d(2048, 2048, kernel_size=[1, 2], stride=[1, 2]),
                BasicBlock(2048, 512, [2, 1], downsample=1),
            )
            self.up2_bvp = nn.Sequential(
                nn.ConvTranspose2d(512, 512, kernel_size=[1, 2], stride=[1, 2]),
                BasicBlock(512, 128, [1, 1], downsample=1),
            )
            self.up3_bvp = nn.Sequential(
                nn.ConvTranspose2d(128, 128, kernel_size=[1, 2], stride=[1, 2]),
                BasicBlock(128, 32, [2, 1], downsample=1),
            )
            self.up4_bvp = nn.Sequential(
                nn.ConvTranspose2d(32, 32, kernel_size=[1, 2], stride=[1, 2]),
                BasicBlock(32, 1, [1, 1], downsample=1),
            )

    def add_adapter(self, adapter_class, gamma=8, lora_alpha=16):
        # Add adapter for resnet blocks
        target_layers = [
            self.resnet.layer1,
            self.resnet.layer2,
            self.resnet.layer3,
            self.resnet.layer4
        ]

        for layer in target_layers:
            for bottleneck_layer in layer:
                for cv in ["conv1", "conv2", "conv3"]:
                    if hasattr(bottleneck_layer, cv) and getattr(bottleneck_layer, cv) is not None:
                        target_conv = getattr(bottleneck_layer, cv)
                        adapter = adapter_class(
                            r=gamma,
                            lora_alpha=lora_alpha,
                            conv_layer=target_conv
                        )
                        setattr(bottleneck_layer, cv, adapter)

    def query(self, x):
        # Add adapter for resnet blocks
        target_layers = [
            self.resnet.layer1,
            self.resnet.layer2,
            self.resnet.layer3,
            self.resnet.layer4
        ]

        for layer in target_layers:
            for bottleneck_layer in layer:
                if getattr(bottleneck_layer, 'downsample') is not None:
                    x = bottleneck_layer.downsample(x)
                for cv in ["conv1", "conv2", "conv3"]:
                    if hasattr(bottleneck_layer, cv) and getattr(bottleneck_layer, cv) is not None:
                        target_conv = getattr(bottleneck_layer, cv)
                        target_conv.get_query(x)
        return x

    def freeze_model(self, freeze=True):
        """Freezes all weights of the model."""
        if freeze:
            # First freeze/ unfreeze all model weights
            for n, p in self.named_parameters():
                if 'lora_' not in n and 'merge_' not in n and 'downsample' not in n and 'experts' not in n and 'gate' not in n:
                    p.requires_grad = False
                else:
                    p.requires_grad = True

            for n, p in self.named_parameters():
                if 'bias' in n:
                    if "fc" not in n:
                        p.requires_grad = True
                elif "bn" in n:
                    p.requires_grad = True
        else:
            # Unfreeze
            for n, p in self.named_parameters():
                p.requires_grad = True
        self.resnet.conv1.requires_grad = True
        self.model_frozen = freeze

    def get_av(self, x):
        av = torch.mean(torch.mean(x, dim=-1), dim=-1)
        min, _ = torch.min(av, dim=1, keepdim=True)
        max, _ = torch.max(av, dim=1, keepdim=True)
        av = torch.mul((av - min), ((max - min).pow(-1)))
        return av

    def count_parameters(self, grad):
        return sum(p.numel() for p in self.parameters() if p.requires_grad == grad)

    def calculate_training_parameter_ratio(self):
        trainable_param_num = self.count_parameters(True)
        other_param_num = self.count_parameters(False)
        print("Non-trainable parameters (M):", other_param_num/ (1024 ** 2))
        print("Trainable parameters (M):", trainable_param_num / (1024 ** 2))

        ratio = trainable_param_num / (other_param_num + trainable_param_num)
        # final_ratio = (ratio / (1 - ratio))
        print("Ratio:", ratio)

        return ratio

    def get_task_weights(self):
        w = []

        target_layers = [
            self.resnet.layer1,
            self.resnet.layer2,
            self.resnet.layer3,
            self.resnet.layer4
        ]
        for layer in target_layers:
            for bottleneck_layer in layer:
                for cv in ["conv1", "conv2", "conv3"]:
                    if hasattr(bottleneck_layer, cv) and getattr(bottleneck_layer, cv) is not None:
                        target_conv = getattr(bottleneck_layer, cv)
                        w_now = target_conv.get_task_weights()
                        w.append(w_now)
        return w

    def predict_spo(self, feature):
        return self.spo(feature)


    def forward(self, input):
        # input, loss = self.LowRankDecomposition(input)
        x = self.resnet.conv1(input)
        #loss = self.LowRankDecomposition(x)
        feat = [x]
        query = self.query(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.layer1(x)
        feat.append(x)
        x = self.resnet.layer2(x)
        feat.append(x)
        x = self.resnet.layer3(x)
        feat.append(x)
        em = self.resnet.layer4(x)
        feat.append(em)

        query_hr = 2 * self.gate_hr(query)
        em_hr = torch.mul(query_hr, em)
        HR = self.hr(self.avgpool(em_hr).view(x.size(0), -1))

        query_spo = 2 * self.gate_spo(query)
        em_spo = torch.mul(query_spo, em)
        SPO = self.spo(self.avgpool(em_spo).view(x.size(0), -1))

        query_rf = 2 * self.gate_rf(query)
        em_rf = torch.mul(query_rf, em)
        RF = self.rf(self.avgpool(em_rf).view(x.size(0), -1))

        # For Sig
        x = self.up1_bvp(em_hr)
        x = self.up2_bvp(x)
        x = self.up3_bvp(x)
        Sig = self.up4_bvp(x).squeeze(dim=1)

        return Sig, HR, SPO, RF, feat, self.avgpool(em_spo).view(x.size(0), -1), [self.avgpool(em).view(x.size(0), -1), self.avgpool(em_hr).view(x.size(0), -1), self.avgpool(em_spo).view(x.size(0), -1), self.avgpool(em_rf).view(x.size(0), -1)]
        # return Sig, HR, SPO, SPO, em


class BaseNet_ViT(nn.Module):
    def __init__(self, pretrain='vit_base_patch16', gamma=8, lora_alpha=16):
        super(BaseNet_ViT, self).__init__()
        if pretrain == 'vit_base_patch16':
            self.vit = models_vit.__dict__['vit_base_patch16'](
                num_classes=1000,
                drop_path_rate=0.1,
                global_pool=True,
                in_chans=3
            )
            self.vit.load_state_dict(torch.load('./pre_encoder/jx_vit_base_p16_224-80ecf9dd.pth'))
            del self.vit.head

            self.add_adapter(MLoraLinear, gamma=gamma, lora_alpha=lora_alpha)
            #self.norm = nn.LayerNorm((768,), eps=1e-06, elementwise_affine=True)
            self.fc_norm = nn.LayerNorm((768,), eps=1e-06, elementwise_affine=True)
            self.freeze_model(True)

            self.hr = nn.Linear(768, 1)
            self.spo = nn.Linear(768, 1)
            self.rf = nn.Linear(768, 1)
            self.bvp = nn.Sequential(
                nn.Linear(768, 768),
                nn.ReLU(inplace=True),
                nn.Linear(768, 256))

            self.gate_hr = nn.Sequential(
                nn.Linear(768, 768, bias=True),
                nn.LayerNorm(768),
                nn.ReLU(inplace=True),
                nn.Linear(768, 768, bias=True),
                nn.LayerNorm(768),
                nn.Dropout(0.1),
                nn.Sigmoid()
            )

            self.gate_spo = nn.Sequential(
                nn.Linear(768, 768, bias=True),
                nn.LayerNorm(768),
                nn.ReLU(inplace=True),
                nn.Linear(768, 768, bias=True),
                nn.LayerNorm(768),
                nn.Dropout(0.1),
                nn.Sigmoid()
            )

            self.gate_rf = nn.Sequential(
                nn.Linear(768, 768, bias=True),
                nn.LayerNorm(768),
                nn.ReLU(inplace=True),
                nn.Linear(768, 768, bias=True),
                nn.LayerNorm(768),
                nn.Dropout(0.1),
                nn.Sigmoid()
            )


    def add_adapter(self, adapter_class, gamma=8, lora_alpha=16):
        target_layers = self.vit.blocks

        for layer in target_layers:
            bottleneck_layer = layer.attn
            for cv in ["qkv"]:
                if hasattr(bottleneck_layer, cv) and getattr(bottleneck_layer, cv) is not None:
                    target_linear = getattr(bottleneck_layer, cv)
                    adapter = adapter_class(
                        r=gamma,
                        lora_alpha=lora_alpha,
                        linear_layer=target_linear
                    )
                    setattr(bottleneck_layer, cv, adapter)
            bottleneck_layer = layer.mlp
            for cv in ['fc1', 'fc2']:
                if hasattr(bottleneck_layer, cv) and getattr(bottleneck_layer, cv) is not None:
                    target_linear = getattr(bottleneck_layer, cv)
                    adapter = adapter_class(
                        r=gamma,
                        lora_alpha=lora_alpha,
                        linear_layer=target_linear
                    )
                    setattr(bottleneck_layer, cv, adapter)


    def query(self, x):
        # Add adapter for resnet blocks
        target_layers = self.vit.blocks

        for layer in target_layers:
            bottleneck_layer = layer.attn
            for cv in ["qkv"]:
                if hasattr(bottleneck_layer, cv) and getattr(bottleneck_layer, cv) is not None:
                    target_linear = getattr(bottleneck_layer, cv)
                    target_linear.get_query(x)
            bottleneck_layer = layer.mlp
            for cv in ['fc1', 'fc2']:
                if hasattr(bottleneck_layer, cv) and getattr(bottleneck_layer, cv) is not None:
                    target_linear = getattr(bottleneck_layer, cv)
                    target_linear.get_query(x)
        return x

    def freeze_model(self, freeze=True):
        """Freezes all weights of the model."""
        if freeze:
            # First freeze/ unfreeze all model weights
            for n, p in self.named_parameters():
                if 'lora_' not in n and 'ls' not in n and 'drop' not in n and 'patch' not in n and 'experts' not in n and 'gate' not in n:
                    p.requires_grad = False
                else:
                    p.requires_grad = True
        else:
            # Unfreeze
            for n, p in self.named_parameters():
                p.requires_grad = True
        self.vit.patch_embed.requires_grad = True
        self.vit.pos_drop.requires_grad = True
        self.vit.patch_drop.requires_grad = True
        self.vit.norm_pre.requires_grad = True

        self.model_frozen = freeze


    def count_parameters(self, grad):
        return sum(p.numel() for p in self.parameters() if p.requires_grad == grad)

    def calculate_training_parameter_ratio(self):
        trainable_param_num = self.count_parameters(True)
        other_param_num = self.count_parameters(False)
        print("Non-trainable parameters (M):", other_param_num/ (1024 ** 2))
        print("Trainable parameters (M):", trainable_param_num / (1024 ** 2))

        ratio = trainable_param_num / (other_param_num + trainable_param_num)
        # final_ratio = (ratio / (1 - ratio))
        print("Ratio:", ratio)

        return ratio

    def get_task_weights(self):
        w = []

        target_layers = self.vit.blocks

        for layer in target_layers:
            bottleneck_layer = layer.attn
            for cv in ["qkv", "proj"]:
                if hasattr(bottleneck_layer, cv) and getattr(bottleneck_layer, cv) is not None:
                    target_linear = getattr(bottleneck_layer, cv)
                    w_now = target_linear.get_task_weights()
                    w.append(w_now)
            bottleneck_layer = layer.mlp
            for cv in ['fc1', 'fc2']:
                if hasattr(bottleneck_layer, cv) and getattr(bottleneck_layer, cv) is not None:
                    target_linear = getattr(bottleneck_layer, cv)
                    w_now = target_linear.get_task_weights()
                    w.append(w_now)
        return w

    def predict_spo(self, feature):
        return self.spo(feature)

    def forward(self, input):
        # input, loss = self.LowRankDecomposition(input)
        x = self.vit.forward_pos(input)
        query = self.query(x)
        #loss = self.LowRankDecomposition(x)
        feat = [x]
        for blk in self.vit.blocks:
            x = blk(x)
            feat.append(x)
        em = x[:, 1:, :].mean(dim=1)  # global pool without cls token
        em = self.fc_norm(em)
        query = query[:, 1:, :].mean(dim=1)
        query = self.fc_norm(query)

        query_hr = 2 * self.gate_hr(query)
        em_hr = torch.mul(query_hr, em)
        HR = self.hr(em_hr)
        query_spo = 2 * self.gate_spo(query)
        em_spo = torch.mul(query_spo, em)
        SPO = self.spo(em_spo)
        query_rf = 2 * self.gate_rf(query)
        em_rf = torch.mul(query_rf, em)
        RF = self.rf(em_rf)
        # For Sig
        Sig = self.bvp(em_hr)

        return Sig, HR, SPO, RF, feat, em_spo, [em, em_hr, em_spo, em_rf]#loss
        # return Sig, HR, SPO, SPO, em
