import torch
from torch import nn, einsum
import torch.nn as nn
import torch.nn.functional
import torch.nn.functional as F
from functools import partial
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange, Reduce
import timm
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import types
import math

class LayerNorm(nn.Module):
    r""" From ConvNeXt (https://arxiv.org/pdf/2201.03545.pdf)
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_first"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)

        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

class ScratchFormer(nn.Module):

    def __init__(self, input_nc=3, output_nc=2, decoder_softmax=False, embed_dim=256):
        super(ScratchFormer, self).__init__()
        #Transformer Encoder
        self.embed_dims = [64, 128, 320, 512]
        self.depths     = [3, 3, 9, 3]
        self.embedding_dim = embed_dim
        self.attn_drop = 0.1
        self.drop_path_rate = 0.1 

        self.Tenc_x2    = EncoderTransformer(patch_size = 7, in_chans=input_nc, num_classes=output_nc, embed_dims=self.embed_dims,
                                             attn_drop_rate = self.attn_drop, drop_path_rate=self.drop_path_rate,
                                             norm_layer=partial(LayerNorm, eps=1e-6), depths=self.depths)
        
        #Transformer Decoder
        self.TDec_x2   = DecoderTransformer(align_corners=False, in_channels = self.embed_dims, embedding_dim= self.embedding_dim,
                                            output_nc=output_nc, decoder_softmax = decoder_softmax)

    def forward(self, x1, x2):

        [fx1, fx2] = [self.Tenc_x2(x1), self.Tenc_x2(x2)]
        cp = self.TDec_x2(fx1, fx2)

        return cp[4]

class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ConvLayer, self).__init__()
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)

    def forward(self, x):
        out = self.conv2d(x)
        return out


class UpsampleConvLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(UpsampleConvLayer, self).__init__()
        self.conv2d = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=1)

    def forward(self, x):
        out = self.conv2d(x)
        return out

class ResidualBlock(torch.nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = ConvLayer(channels, channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = ConvLayer(channels, channels, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        out = self.relu(self.conv1(x))
        out = self.conv2(out) * 0.1
        out = torch.add(out, residual)
        return out

class OverlapPatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()

        patch_size = to_2tuple(patch_size)

        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride, padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = LayerNorm(embed_dim, eps=1e-6, data_format="channels_first")

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)

            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))

            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.proj(x)

        _, _, H, W = x.shape
        #x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)

        return x, H, W

def resize(input, size=None, scale_factor=None, mode='nearest', align_corners=None, warning=True):
    if warning:
        if size is not None and align_corners:
            input_h, input_w = tuple(int(x) for x in input.shape[2:])
            output_h, output_w = tuple(int(x) for x in size)

            if output_h > input_h or output_w > output_h:

                if ((output_h > 1 and output_w > 1 and input_h > 1
                     and input_w > 1) and (output_h - 1) % (input_h - 1)
                        and (output_w - 1) % (input_w - 1)):

                    warnings.warn(
                        f'When align_corners={align_corners}, '
                        'the output would more aligned if '
                        f'input size {(input_h, input_w)} is `x+1` and '
                        f'out size {(output_h, output_w)} is `nx+1`')
    return F.interpolate(input, size, scale_factor, mode, align_corners)


# Transformer Decoder MLP
class MLP(nn.Module):
    """
    Linear Embedding
    """
    def __init__(self, input_dim=2048, embed_dim=768):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x


#Difference module
def conv_diff(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.BatchNorm2d(out_channels),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU()
    )

#Intermediate prediction module
def make_prediction(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.BatchNorm2d(out_channels),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
    )


#Transormer Ecoder with x2, x4, x8, x16 scales
class EncoderTransformer(nn.Module):
    def __init__(self, img_size=256, patch_size=3, in_chans=3, num_classes=2, embed_dims=[32, 64, 128, 256],
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=LayerNorm, depths=[3, 3, 6, 18]):
        super().__init__()
        self.num_classes    = num_classes
        self.depths         = depths
        self.embed_dims     = embed_dims

        # patch embedding definitions
        self.patch_embed1 = OverlapPatchEmbed(patch_size=7, stride=4, in_chans=in_chans, embed_dim=embed_dims[0])
        self.patch_embed2 = OverlapPatchEmbed(patch_size=patch_size, stride=2, in_chans=embed_dims[0], embed_dim=embed_dims[1])
        self.patch_embed3 = OverlapPatchEmbed(patch_size=patch_size, stride=2, in_chans=embed_dims[1], embed_dim=embed_dims[2])
        self.patch_embed4 = OverlapPatchEmbed(patch_size=patch_size, stride=2, in_chans=embed_dims[2], embed_dim=embed_dims[3])

        # Stage-1 (x1/4 scale)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0 

        self.block1 = nn.ModuleList([EncoderBlock(dim=embed_dims[0], dim_head=4)
                                     for i in range(depths[0])])
        self.norm1 = norm_layer(embed_dims[0])
        
        # Stage-2 (x1/8 scale)
        cur += depths[0]

        self.block2 = nn.ModuleList([EncoderBlock(dim=embed_dims[1], dim_head=4)
                                     for i in range(depths[1])])
        self.norm2 = norm_layer(embed_dims[1])
       
       # Stage-3 (x1/16 scale)
        cur += depths[1]
        
        self.block3 = nn.ModuleList([EncoderBlock(dim=embed_dims[2], dim_head=8)
                                     for i in range(depths[2])])
        self.norm3 = norm_layer(embed_dims[2])
        
        # Stage-4 (x1/32 scale)
        cur += depths[2]

        self.block4 = nn.ModuleList([EncoderBlock(dim=embed_dims[3], dim_head=8)
                                     for i in range(depths[3])])
        self.norm4 = norm_layer(embed_dims[3])

        self.apply(self._init_weights)


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)

            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))

            if m.bias is not None:
                m.bias.data.zero_()
    
    def forward_features(self, feats):
        B = feats.shape[0]
        outs = []
    
        # stage 1
        feats, H1, W1 = self.patch_embed1(feats)
        for i, blk in enumerate(self.block1):
            feats = blk(feats, H1, W1)
        feats = self.norm1(feats)
        #feats = feats.reshape(B, H1, W1, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(feats)

        feats, H1, W1 = self.patch_embed2(feats)
        for i, blk in enumerate(self.block2):
            feats = blk(feats, H1, W1)
        feats = self.norm2(feats)
        #feats = feats.reshape(B, H1, W1, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(feats)

        # stage 3
        feats, H1, W1 = self.patch_embed3(feats)
        for i, blk in enumerate(self.block3):
            feats = blk(feats, H1, W1)
        feats = self.norm3(feats)
        #feats = feats.reshape(B, H1, W1, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(feats)

        # stage 4
        feats, H1, W1 = self.patch_embed4(feats)
        for i, blk in enumerate(self.block4):
            feats = blk(feats, H1, W1)
        feats = self.norm4(feats)
        #feats = feats.reshape(B, H1, W1, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(feats)
        
        return outs

    def forward(self, x):
        x = self.forward_features(x)
        return x

class DecoderTransformer(nn.Module):
    """
    Transformer Decoder
    """
    def __init__(self, align_corners=True, in_channels=[64, 128, 320, 512], embedding_dim=256, output_nc=2, decoder_softmax=False):
        super(DecoderTransformer, self).__init__()
        
        #settings
        self.align_corners   = align_corners
        self.in_channels     = in_channels
        self.embedding_dim   = embedding_dim
        self.output_nc       = output_nc
        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = self.in_channels

        #MLP decoder heads
        self.linear_c4 = MLP(input_dim=c4_in_channels, embed_dim=self.embedding_dim)
        self.linear_c3 = MLP(input_dim=c3_in_channels, embed_dim=self.embedding_dim)
        self.linear_c2 = MLP(input_dim=c2_in_channels, embed_dim=self.embedding_dim)
        self.linear_c1 = MLP(input_dim=c1_in_channels, embed_dim=self.embedding_dim)

        #taking outputs from middle of the encoder
        self.make_pred_c4 = make_prediction(in_channels=self.embedding_dim, out_channels=self.output_nc)
        self.make_pred_c3 = make_prediction(in_channels=self.embedding_dim, out_channels=self.output_nc)
        self.make_pred_c2 = make_prediction(in_channels=self.embedding_dim, out_channels=self.output_nc)
        self.make_pred_c1 = make_prediction(in_channels=self.embedding_dim, out_channels=self.output_nc)

        #Final linear fusion layer
        self.linear_fuse = nn.Sequential(
           nn.Conv2d(in_channels=self.embedding_dim*len(in_channels), out_channels=self.embedding_dim, kernel_size=1),
            nn.BatchNorm2d(self.embedding_dim)
        )
      
        self.ceff1 = CEFF(in_channels=self.embedding_dim, height=2)
        self.ceff2 = CEFF(in_channels=self.embedding_dim, height=2)
        self.ceff3 = CEFF(in_channels=self.embedding_dim, height=2)
        self.ceff4 = CEFF(in_channels=self.embedding_dim, height=2)

        #Final predction head
        self.convd2x    = UpsampleConvLayer(self.embedding_dim, self.embedding_dim, kernel_size=4, stride=2)
        self.dense_2x   = nn.Sequential( ResidualBlock(self.embedding_dim))
        self.convd1x    = UpsampleConvLayer(self.embedding_dim, self.embedding_dim, kernel_size=4, stride=2)
        self.dense_1x   = nn.Sequential( ResidualBlock(self.embedding_dim))
        self.change_probability = ConvLayer(self.embedding_dim, self.output_nc, kernel_size=3, stride=1, padding=1)
        
        #Final activation
        self.output_softmax     = decoder_softmax
        self.active             = nn.Sigmoid()

    def forward(self, x_1, x_2):

        #img1 and img2 features
        c1_1, c2_1, c3_1, c4_1 = x_1
        c1_2, c2_2, c3_2, c4_2 = x_2

        ############## MLP decoder on C1-C4 ###########
        n, _, h, w = c4_1.shape

        outputs = []
        # Stage 4: x1/32 scale
        _c4_1 = self.linear_c4(c4_1).permute(0,2,1).reshape(n, -1, c4_1.shape[2], c4_1.shape[3])
        _c4_2 = self.linear_c4(c4_2).permute(0,2,1).reshape(n, -1, c4_2.shape[2], c4_2.shape[3])
        _c4   = self.ceff1([_c4_1, _c4_2])
        p_c4  = self.make_pred_c4(_c4)
        outputs.append(p_c4)
        _c4_up= resize(_c4, size=c1_2.size()[2:], mode='bilinear', align_corners=False)

        # Stage 3: x1/16 scale
        _c3_1 = self.linear_c3(c3_1).permute(0,2,1).reshape(n, -1, c3_1.shape[2], c3_1.shape[3])
        _c3_2 = self.linear_c3(c3_2).permute(0,2,1).reshape(n, -1, c3_2.shape[2], c3_2.shape[3])
        _c3   = self.ceff2([_c3_1, _c3_2])
        p_c3  = self.make_pred_c3(_c3)
        outputs.append(p_c3)
        _c3_up= resize(_c3, size=c1_2.size()[2:], mode='bilinear', align_corners=False)

        # Stage 2: x1/8 scale
        _c2_1 = self.linear_c2(c2_1).permute(0,2,1).reshape(n, -1, c2_1.shape[2], c2_1.shape[3])
        _c2_2 = self.linear_c2(c2_2).permute(0,2,1).reshape(n, -1, c2_2.shape[2], c2_2.shape[3])
        _c2   = self.ceff3([_c2_1, _c2_2])
        p_c2  = self.make_pred_c2(_c2)
        outputs.append(p_c2)
        _c2_up= resize(_c2, size=c1_2.size()[2:], mode='bilinear', align_corners=False)

        # Stage 1: x1/4 scale
        _c1_1 = self.linear_c1(c1_1).permute(0,2,1).reshape(n, -1, c1_1.shape[2], c1_1.shape[3])
        _c1_2 = self.linear_c1(c1_2).permute(0,2,1).reshape(n, -1, c1_2.shape[2], c1_2.shape[3])
        _c1   = self.ceff4([_c1_1, _c1_2])
        p_c1  = self.make_pred_c1(_c1)
        outputs.append(p_c1)

        #Linear Fusion of difference image from all scales
        _c = self.linear_fuse(torch.cat([_c4_up, _c3_up, _c2_up, _c1],dim=1))

        #Upsampling x2 (x1/2 scale)
        x = self.convd2x(_c)
        #Residual block
        x = self.dense_2x(x)
        #Upsampling x2 (x1 scale)
        x = self.convd1x(x)
        #Residual block
        x = self.dense_1x(x)

        #Final prediction
        cp = self.change_probability(x)
        
        outputs.append(cp)

        if self.output_softmax:
            temp = outputs
            outputs = []
            for pred in temp:
                outputs.append(self.active(pred))

        return outputs

class Attention(nn.Module):
    def __init__(self, dim, dim_head = 32, dropout = 0., window_size = 7):
        super().__init__()
        assert (dim % dim_head) == 0, 'dimension should be divisible by dimension per head'

        self.heads = dim // dim_head
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, dim * 3, bias = False)

        self.attend = nn.Sequential(
            nn.Softmax(dim = -1),
            nn.Dropout(dropout)
        )

        self.to_out = nn.Sequential(
            nn.Linear(dim, dim, bias = False),
            nn.Dropout(dropout)
        )

        # relative positional bias

        self.rel_pos_bias = nn.Embedding((2 * window_size - 1) ** 2, self.heads)

        pos = torch.arange(window_size)
        grid = torch.stack(torch.meshgrid(pos, pos, indexing = 'ij'))
        grid = rearrange(grid, 'c i j -> (i j) c')
        rel_pos = rearrange(grid, 'i ... -> i 1 ...') - rearrange(grid, 'j ... -> 1 j ...')
        rel_pos += window_size - 1
        rel_pos_indices = (rel_pos * torch.tensor([2 * window_size - 1, 1])).sum(dim = -1)

        self.register_buffer('rel_pos_indices', rel_pos_indices, persistent = False)

    def forward(self, x):
        batch, height, width, window_height, window_width, _, device, h = *x.shape, x.device, self.heads

        # flatten
        x = rearrange(x, 'b x y w1 w2 d -> (b x y) (w1 w2) d')

        # project for queries, keys, values
        q, k, v = self.to_qkv(x).chunk(3, dim = -1)

        # split heads
        q, k, v = map(lambda t: rearrange(t, 'b n (h d ) -> b h n d', h = h), (q, k, v))

        # scale
        q = q * self.scale

        # sim
        sim = einsum('b h i d, b h j d -> b h i j', q, k)

        # add positional bias
        bias = self.rel_pos_bias(self.rel_pos_indices)
        sim = sim + rearrange(bias, 'i j h -> h i j')

        # attention
        attn = self.attend(sim)

        # aggregate
        out = einsum('b h i j, b h j d -> b h i d', attn, v)

        # merge heads
        out = rearrange(out, 'b h (w1 w2) d -> b w1 w2 (h d)', w1 = window_height, w2 = window_width)

        # combine heads out
        out = self.to_out(out)

        return rearrange(out, '(b x y) ... -> b x y ...', x = height, y = width)


class Conv_FeedForward(nn.Module):
    def __init__(self, dim, mult=4):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(dim, dim * mult, 1, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(dim * mult, dim, 1, 1, bias=False),
        )

    def forward(self, x):       
        return self.net(x)


class EncoderBlock(nn.Module):
    def __init__(self, dim, dim_head=8, grid_dropout=0., window_size=4, drop_path=0.1):
        super().__init__()
        self.window_size = window_size
        layer_scale_init_value = 1e-6

        self.pos = nn.Conv2d(dim, dim, kernel_size=3, padding=1, stride=1, padding_mode='replicate', bias=False, groups=dim)

        self.layer_norm1 = LayerNorm(dim, eps=1e-6, data_format="channels_first")
        self.layer_norm2 = LayerNorm(dim, eps=1e-6, data_format="channels_first")
        self.layer_norm0 = LayerNorm(dim, eps=1e-6, data_format="channels_first")

        self.mlp = Conv_FeedForward(dim=dim)        

        # sparse attention
        self.deform_grid = DeformableGrid(dim)
        self.attn = nn.Sequential(
                    Rearrange('b d (w1 x) (w2 y) -> b x y w1 w2 d', w1 = self.window_size, w2 = self.window_size),
                    Attention(dim = dim, dim_head = dim_head, dropout = grid_dropout, window_size = self.window_size),
                    Rearrange('b x y w1 w2 d -> b d (w1 x) (w2 y)'),
                )
        self.act = nn.GELU()

        self.layer_scale_1 = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
        self.drop_path_1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.layer_scale_2 = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
        self.drop_path_2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        
    def forward(self, x, H, W):
        B, C, H, W = x.shape

        skip = x
        x = self.layer_norm0(skip)
        x = skip + self.act(self.pos(x))
        
        skip = x       
        x = self.layer_norm1(skip)
        x = self.deform_grid(x)
        x = self.attn(x)
        x = self.drop_path_1(self.layer_scale_1.unsqueeze(-1).unsqueeze(-1) * x)
        out = x  + skip

        x = self.layer_norm2(out)
        x = self.mlp(x)
        x = self.drop_path_2(self.layer_scale_2.unsqueeze(-1).unsqueeze(-1) * x)
        out = out + x

        return out


class CEFF(nn.Module):
    def __init__(self, in_channels, height=2, reduction=8, bias=False):
        super(CEFF, self).__init__()
        
        self.height = height
        d = max(int(in_channels/reduction),4)
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(nn.Conv2d(in_channels, d, 1, padding=0, bias=bias), nn.LeakyReLU(0.2))

        self.fcs = nn.ModuleList([])
        for i in range(self.height):
            self.fcs.append(nn.Conv2d(d, in_channels, kernel_size=1, stride=1,bias=bias))
        
        self.softmax = nn.Softmax(dim=1)

    def forward(self, inp_feats):
        batch_size = inp_feats[0].shape[0]
        n_feats =  inp_feats[0].shape[1]

        inp_feats = torch.cat(inp_feats, dim=1)
        inp_feats = inp_feats.view(batch_size, self.height, n_feats, inp_feats.shape[2], inp_feats.shape[3])
        
        feats_U = torch.sum(inp_feats, dim=1)
        feats_S = self.avg_pool(feats_U)
        feats_Z = self.conv_du(feats_S)

        attention_vectors = [fc(feats_Z) for fc in self.fcs]
        attention_vectors = torch.cat(attention_vectors, dim=1)
        attention_vectors = attention_vectors.view(batch_size, self.height, n_feats, 1, 1)
        
        attention_vectors = self.softmax(attention_vectors)
        
        feats_V = torch.sum(inp_feats*attention_vectors, dim=1)
        
        return feats_V        

class DeformableGrid(nn.Module):
    def __init__(self, in_channels, kernel_size=3, stride=1, padding=1, bias=True):

        super(DeformableGrid, self).__init__()

        self.offset_conv = nn.Conv2d(in_channels, 2, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)

        nn.init.kaiming_uniform_(self.offset_conv.weight, a=math.sqrt(5))

        if self.offset_conv.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.offset_conv.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.offset_conv.bias, -bound, bound)

    def forward(self, x):
        B, C, H, W = x.shape
        
        max_offset = max(H, W)/4.
        
        offset = self.offset_conv(x).clamp(-max_offset, max_offset)

        x_offset = offset[0,0,:,:]
        y_offset = offset[0,1,:,:]
        
        xgrid, ygrid = torch.meshgrid(torch.arange(H), torch.arange(W))
        xgrid = xgrid.to(x.device)
        ygrid = ygrid.to(x.device)
        
        xgrid = xgrid + x_offset
        ygrid = ygrid + y_offset
        
        xgrid = xgrid.to(torch.long)
        xgrid[xgrid >= H] = H-1

        ygrid = ygrid.to(torch.long)
        ygrid[ygrid >= W] = W-1
        
        out = x.clone()
        out = x[:,:,xgrid,ygrid]
        
        return out