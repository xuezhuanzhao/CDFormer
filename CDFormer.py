
import torch
import torch.nn as nn
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import torch.nn.functional as F
import math
from functools import partial
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg
from einops import rearrange

from timm.models.layers import trunc_normal_, DropPath
from timm.models.registry import register_model
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.layers.helpers import to_2tuple

def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': 1.0, 'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD, 'classifier': 'head',
        **kwargs
    }


default_cfgs = {
    'cdformer_tiny': _cfg(
        url=''),
}

class LayerNorm2d(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.norm = nn.LayerNorm(dim, eps=1e-6)

    def forward(self, x):
        return self.norm(x.permute(0, 2, 3, 1).contiguous()).permute(0, 3, 1, 2).contiguous()



class ResDWC(nn.Module):
    def __init__(self, dim, kernel_size=3):
        super().__init__()

        self.dim = dim
        self.kernel_size = kernel_size
        self.conv = nn.Conv2d(dim, dim, kernel_size, 1, kernel_size // 2, groups=dim)

    def forward(self, x):
        return x + self.conv(x)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., conv_pos=True,
                 downsample=False, kernel_size=5):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features = out_features or in_features
        self.hidden_features = hidden_features = hidden_features or in_features

        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.act1 = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)

        self.conv = ResDWC(hidden_features, 3)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act1(x)
        x = self.drop(x)
        x = self.conv(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x





class LayerNormProxy(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        x = rearrange(x, 'b c h w -> b h w c')
        x = x.contiguous()
        x = self.norm(x)
        x = rearrange(x, 'b h w c -> b c h w')
        x = x.contiguous()
        return x





class CDAttention(nn.Module):
    def __init__(self, dim, head_dim=32, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., samp_ratio=2,dwconv_k=5,neibor=3):
        super().__init__()
        print("CDAttention dwconv_k samp_ratio neibor",dwconv_k,samp_ratio,neibor)
        self.dim = dim
        self.num_heads = dim // head_dim
        head_dim = head_dim
        self.distribution_scale = dim ** -0.5
        self.scale = qk_scale or head_dim ** -0.5
        self.lepe = nn.Conv2d(dim, dim, kernel_size=dwconv_k, stride=1, padding=dwconv_k // 2, groups=dim)
        self.kv = nn.Conv2d(dim, dim * 2, 1, bias=qkv_bias)
        self.attn_drop1 = nn.Dropout(attn_drop)
        self.attn_drop2 = nn.Dropout(attn_drop)
        self.proj = nn.Conv2d(dim, dim, 1)
        self.proj_drop = nn.Dropout(proj_drop)
        self.r=samp_ratio
        self.sr = nn.AvgPool2d((self.r, self.r))
        self.proj_q = nn.Conv2d(
            dim, dim,
            kernel_size=1, stride=1, bias=qkv_bias,padding=0
        )
        self.k=neibor
        self.n_k=neibor**2
        self.unfold=torch.nn.Unfold(self.k, padding=self.k//2, stride=1)

    def forward(self, x):
        B0, C0, H0, W0 = x.shape
        pad_l = pad_t = 0
        pad_right = (self.r-W0 % self.r) % self.r
        pad_below = (self.r-H0 % self.r) % self.r
        if pad_right > 0 or pad_below > 0:
            x = F.pad(x, (pad_l, pad_right, pad_t, pad_below))
        B, C, H, W = x.shape
        N = H * W
        k, v = self.kv(x).reshape(B, self.num_heads, C // self.num_heads * 2, N).chunk(2,dim=2)  # (B, num_heads, head_dim, N)
        x_samp=self.sr(x)
        bb,cc,hh,ww=x_samp.shape
        nn=hh*ww
        x_unfold=rearrange(x, 'b c (h r1) (w r2) -> b (h w) (r1 r2) c',r1=self.r,r2=self.r)
        q_samp=self.proj_q(x_samp).reshape(B, self.num_heads, C // self.num_heads, nn)
        lepe = self.lepe(v.reshape(B, C, H, W))

        attn_collection = (k.transpose(-1, -2) @ q_samp) * self.scale
        attn_collection = attn_collection.softmax(dim=-2)
        attn_collection = self.attn_drop1(attn_collection)
        distribution = (v @ attn_collection)

        samp_unfold=self.unfold(x_samp).reshape(B, C,self.n_k,nn)
        samp_unfold=samp_unfold.permute(0,3,1,2)
        distribution_matrix = (x_unfold @ samp_unfold) * self.distribution_scale
        distribution_matrix = distribution_matrix.softmax(dim=-1)
        feature_unfold=self.unfold(distribution.reshape(B, C, hh, ww)).reshape(B, C,self.n_k,nn)
        feature_unfold=feature_unfold.permute(0,3,1,2)
        feature = distribution_matrix @ feature_unfold.transpose(-1,-2)
        x=rearrange(feature, 'b (h w) (r1 r2) c -> b c (h r1) (w r2)',h=hh,r1=self.r)
        x = x + lepe
        if pad_right > 0 or pad_below > 0:
            x = x[:, :, :H0, :W0]

        x = self.proj(x)
        x = self.proj_drop(x)
        return x




class Attention(nn.Module):
    def __init__(self, dim, head_dim=32, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., dwconv_k=3):
        super().__init__()
        print("Attention dwconv_k",dwconv_k)
        self.dim = dim
        self.num_heads = dim // head_dim
        head_dim = head_dim
        self.lepe = nn.Conv2d(dim, dim, kernel_size=dwconv_k, stride=1, padding=dwconv_k // 2, groups=dim)
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Conv2d(dim, dim * 3, 1, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Conv2d(dim, dim, 1)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, C, H, W = x.shape
        N = H * W
        q, k, v = self.qkv(x).reshape(B, self.num_heads, C // self.num_heads * 3, N).chunk(3,
                                                                                           dim=2)  # (B, num_heads, head_dim, N)
        lepe = self.lepe(v.reshape(B, C, H, W))
        attn = (k.transpose(-1, -2) @ q) * self.scale

        attn = attn.softmax(dim=-2)  # (B, h, N, N)
        attn = self.attn_drop(attn)

        x = (v @ attn).reshape(B, C, H, W)
        x = x + lepe
        x = self.proj(x)
        x = self.proj_drop(x)
        return x



class Scale(nn.Module):
    """
    Scale vector by element multiplications.
    """

    def __init__(self, dim, init_value=1.0, trainable=True):
        super().__init__()
        self.scale = nn.Parameter(init_value * torch.ones(1, dim, 1, 1), requires_grad=trainable)

    def forward(self, x):
        return x * self.scale


class BasicLayer(nn.Module):
    def __init__(self, dim, mlp_ratio=4., drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, layerscale=False, resscale=False, init_values=1.0e-5,
                 mixers_type=1,samp_ratio=2,dwconv_k=3,neibor=3):
        super().__init__()
        self.resscale = resscale
        self.layerscale = layerscale
        if mixers_type == 1:
            self.token_mixers = CDAttention(dim=dim, samp_ratio=samp_ratio,dwconv_k=dwconv_k,neibor=neibor)
        elif mixers_type == 2:
            self.token_mixers = Attention(dim=dim, dwconv_k=dwconv_k)


        self.pos_embed = ResDWC(dim, 3)
        self.norm1 = LayerNorm2d(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = nn.BatchNorm2d(dim)
        self.mlp2 = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), out_features=dim, act_layer=act_layer,
                        drop=drop)

        if layerscale:
            self.gamma_1 = nn.Parameter(init_values * torch.ones(1, dim, 1, 1), requires_grad=True)
            self.gamma_2 = nn.Parameter(init_values * torch.ones(1, dim, 1, 1), requires_grad=True)
        if resscale:
            self.scale = Scale(dim=dim)

    def forward(self, x):
        x = self.pos_embed(x)
        if self.layerscale:
            x = x + self.drop_path(self.gamma_1 * self.token_mixers(self.norm1(x)))
            x = x + self.drop_path(self.gamma_2 * self.mlp2(self.norm2(x)))
        elif self.resscale:
            x = self.scale(x) + self.drop_path(self.token_mixers(self.norm1(x)))
            x = self.scale(x) + self.drop_path(self.mlp2(self.norm2(x)))
        else:
            x = x + self.drop_path(self.token_mixers(self.norm1(x)))
            x = x + self.drop_path(self.mlp2(self.norm2(x)))
        return x


class StageLayer(nn.Module):
    def __init__(self, num_layers, dim, mlp_ratio=4., drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, layerscale=False, resscale=False, init_values=1.0e-5, mixers_type=1,
                 samp_ratio=2,dwconv_k=3,neibor=3,downsample=False):
        super().__init__()
        self.blocks = nn.ModuleList([BasicLayer(
            dim=dim[0], mlp_ratio=mlp_ratio, drop=drop, attn_drop=attn_drop,
            drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
            act_layer=act_layer,
            layerscale=layerscale, resscale=resscale, init_values=init_values,
            mixers_type=mixers_type,samp_ratio=samp_ratio,dwconv_k=dwconv_k,neibor=neibor) for i in
            range(num_layers)])
        if downsample:
            self.downsample = PatchMerging(dim[0], dim[1])
        else:
            self.downsample = None

    def forward(self, x):
        for idx, blk in enumerate(self.blocks):
            x = blk(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x


class PatchEmbed(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 2, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(out_channels // 2),
            nn.GELU(),

            nn.Conv2d(out_channels // 2, out_channels // 2, 3, 1, 1),
            nn.BatchNorm2d(out_channels // 2),
            nn.GELU(),

            nn.Conv2d(out_channels // 2, out_channels, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),

            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),

        )

    def forward(self, x):
        x = self.proj(x)
        return x


class PatchMerging(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        x = self.proj(x)
        return x


class CDFormer(nn.Module):
    def __init__(self, in_chans=3, num_classes=1000,
                 embed_dim=[96, 192, 384, 768], depths=[2, 2, 8, 2],
                 mlp_ratio=4., drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 freeze_bn=False, layerscale=[False, False, False, False], resscale=[False, False, False, False],
                 init_values=1e-6, **kwargs):
        super().__init__()

        print("lr 2e-3 300 warmup-epochs 20 3 DSAtten 1attention adamw")
        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.num_features = embed_dim[-1]
        self.mlp_ratio = mlp_ratio
        self.mixers_type = [1, 1, 1, 2]
        self.samp_ratio=[8,4,2,None]
        self.dwconv_k=[5,5,5,5]
        self.neibor=[5,5,3,None]
        self.freeze_bn = freeze_bn
        self.patch_embed = PatchEmbed(in_chans, embed_dim[0])

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        print(dpr)

        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = StageLayer(num_layers=depths[i_layer],
                               dim=[embed_dim[i_layer],
                                    embed_dim[i_layer + 1] if i_layer < self.num_layers - 1 else None],
                               mlp_ratio=self.mlp_ratio,
                               drop=drop_rate, attn_drop=attn_drop_rate,
                               drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                               downsample=i_layer < self.num_layers - 1,
                               layerscale=layerscale[i_layer],
                               resscale=resscale[i_layer],
                               init_values=init_values, mixers_type=self.mixers_type[i_layer],
                               samp_ratio=self.samp_ratio[i_layer],dwconv_k=self.dwconv_k[i_layer],neibor=self.neibor[i_layer])
            self.layers.append(layer)

        self.norm = nn.BatchNorm2d(self.num_features)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(self.num_features, num_classes)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def forward_features(self, x):
        x = self.patch_embed(x)

        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        x = self.avgpool(x).flatten(1)  # B C 1
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x

@register_model
def cdformer_b0(pretrained=False, **kwargs):
    model = CDFormer(embed_dim=[32, 64, 192, 256],
                     depths=[2, 2, 8, 2],
                     mlp_ratio=3,
                     layerscale=[False, False, False, False],
                     resscale=[False, False, False, False],
                     init_values=1e-5, **kwargs)
    model.default_cfg = default_cfgs['cdformer_tiny']
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(
            url= model.default_cfg['url'], map_location="cpu", check_hash=True)
        model.load_state_dict(state_dict)
    return model
@register_model
def cdformer_b1(pretrained=False, **kwargs):
    model = CDFormer(embed_dim=[64, 128, 256, 512],
                     depths=[2, 2, 8, 2],
                     mlp_ratio=3,
                     layerscale=[bFalse, False, False, False],
                     resscale=[False, False, False, False],
                     init_values=1e-5, **kwargs)
    model.default_cfg = default_cfgs['cdformer_tiny']
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(
            url= model.default_cfg['url'], map_location="cpu", check_hash=True)
        model.load_state_dict(state_dict)
    return model

@register_model
def cdformer_b2(pretrained=False, **kwargs):
    model = CDFormer(embed_dim=[64, 128, 320, 512],
                     depths=[4, 4, 12, 4],
                     mlp_ratio=3,
                     layerscale=[False, False, False, False],
                     resscale=[False, False, False, False],
                     init_values=1e-5, **kwargs)
    model.default_cfg = default_cfgs['cdformer_tiny']
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(
            url= model.default_cfg['url'], map_location="cpu", check_hash=True)
        model.load_state_dict(state_dict)
    return model


