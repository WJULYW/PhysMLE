from functools import partial

import torch
import torch.nn as nn
import numpy as np

import timm.models.vision_transformer
from timm.models.vision_transformer import Block


class VisionTransformer(timm.models.vision_transformer.VisionTransformer):
    """ Vision Transformer with support for global average pooling
    """

    def __init__(self, global_pool=False, **kwargs):
        super(VisionTransformer, self).__init__(**kwargs)
        # self.cross_block = cross_Block(dim=kwargs['embed_dim'], num_heads=kwargs['num_heads'],
        #  mlp_ratio=kwargs['mlp_ratio'], qkv_bias=True, qk_scale=None, norm_layer=kwargs['norm_layer'])
        self.head = None


    def forward_pos(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        return x

    def forward_features(self, x):
        if self.global_pool:
            x = x[:, 1:, :].mean(dim=1)  # global pool without cls token
            outcome = self.fc_norm(x)
        else:
            x = self.norm(x)
            outcome = x[:, 0]

        return outcome

    def forward(self, x):
        x = self.forward_pos(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.forward_features(x)
        return x




def vit_base_patch16(**kwargs):
    #url = "https://storage.googleapis.com/vit_models/imagenet21k/ViT-B_16.npz"
    #load npz file
    #checkpoint = np.load('./pre_encoder/ViT-B_16.npz')
    #checkpoint = np.load(url=url, map_location="cpu", check_hash=True)

    model = VisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    #model.load_state_dict(checkpoint["model"])
    return model


def vit_large_patch16(**kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_huge_patch14(**kwargs):
    model = VisionTransformer(
        patch_size=14, embed_dim=1280, depth=32, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model