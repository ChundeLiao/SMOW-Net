import torch
import math
import torch.nn as nn
from einops import rearrange
import torch.nn.functional as F
from torch.hub import load_state_dict_from_url


class SMOW_Net_LW(nn.Module):
    def __init__(self):
        super(SMOW_Net_LW, self).__init__()
        self.backbone = mobilenet_v2(pretrained=True)

        self.OFW = OFW(16)

        self.MaxPool = max_pooling_3d()

        self.C3DT1 = conv_trans_block_3d(320, 320)
        self.C3D1 = conv_block_2_3d(640, 160)
        self.C3DT2 = conv_trans_block_3d(160, 160)
        self.C3D2 = conv_block_2_3d(256, 64)
        self.C3DT3 = conv_trans_block_3d(64, 64)
        self.C3D3 = conv_block_2_3d(96, 32)
        self.C3DT4 = conv_trans_block_3d(32, 32)
        self.C3D4 = conv_block_2_3d(56, 28)
        self.C3DT5 = conv_trans_block_3d(28, 28)
        self.C3D5 = conv_block_2_3d(44, 16)

        self.Transformer_Encoder = Transformer_Encoder(in_chan=16)
        self.Transformer_Decoder = Transformer_Decoder(in_chan=64)
        self.decoder = Classifier(in_chan=64, n_class=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x1, x2):
        x1_1, x1_2, x1_3, x1_4, x1_5 = self.backbone(x1)
        x2_1, x2_2, x2_3, x2_4, x2_5 = self.backbone(x2)

        x11 = x1_1.unsqueeze(2)
        x22 = x2_1.unsqueeze(2)
        x0 = torch.cat([x11, x22], 2)

        x11 = x1_2.unsqueeze(2)
        x22 = x2_2.unsqueeze(2)
        x1 = torch.cat([x11, x22], 2)

        x11 = x1_3.unsqueeze(2)
        x22 = x2_3.unsqueeze(2)
        x2 = torch.cat([x11, x22], 2)

        x11 = x1_4.unsqueeze(2)
        x22 = x2_4.unsqueeze(2)
        x3 = torch.cat([x11, x22], 2)

        x11 = x1_5.unsqueeze(2)
        x22 = x2_5.unsqueeze(2)
        x4 = torch.cat([x11, x22], 2)

        x8 = self.OFW(x0)
        
        x8 = self.Transformer_Encoder(x8)

        b, c, t, h, w = x0.shape
        x0 = F.interpolate(x0, size=(4, h, w), mode='trilinear', align_corners=True)
        b, c, t, h, w = x1.shape
        x1 = F.interpolate(x1, size=(4, h, w), mode='trilinear', align_corners=True)
        b, c, t, h, w = x2.shape
        x2 = F.interpolate(x2, size=(4, h, w), mode='trilinear', align_corners=True)
        b, c, t, h, w = x3.shape
        x3 = F.interpolate(x3, size=(4, h, w), mode='trilinear', align_corners=True)
        b, c, t, h, w = x4.shape
        x4 = F.interpolate(x4, size=(4, h, w), mode='trilinear', align_corners=True)
        
        maxpool = self.MaxPool(x4)

        c3dt1 = self.C3DT1(maxpool)
        concat1 = torch.cat([c3dt1, x4], dim=1)
        c3d1 = self.C3D1(concat1)
        
        c3dt2 = self.C3DT2(c3d1)
        concat2 = torch.cat([c3dt2, x3], dim=1)
        c3d2 = self.C3D2(concat2)
        
        c3dt3 = self.C3DT3(c3d2)
        concat3 = torch.cat([c3dt3, x2], dim=1)
        c3d3 = self.C3D3(concat3)
        
        c3dt4 = self.C3DT4(c3d3)
        concat4 = torch.cat([c3dt4, x1], dim=1)
        c3d4 = self.C3D4(concat4)

        c3dt5 = self.C3DT5(c3d4)
        concat5 = torch.cat([c3dt5, x0], dim=1)
        c3d5 = self.C3D5(concat5)

        x16 = self.Transformer_Decoder(c3d5, x8)
        x16 = self.decoder(x16)
        x16 = self.sigmoid(x16)

        return x16


class conv_trans_block_3d(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(conv_trans_block_3d, self).__init__()
        self.conv3d_spatial = nn.ConvTranspose3d(in_dim, out_dim, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1), output_padding=(0, 1, 1))
        self.conv3d_time_1 = nn.ConvTranspose3d(out_dim, out_dim, kernel_size=(1, 1, 1), stride=(1, 1, 1), bias=False)
        self.conv3d_time_2 = nn.ConvTranspose3d(out_dim, out_dim, kernel_size=(1, 1, 1), stride=(1, 1, 1), bias=False)
        self.conv3d_time_3 = nn.ConvTranspose3d(out_dim, out_dim, kernel_size=(1, 1, 1), stride=(1, 1, 1), bias=False)
        self.conv3d_time_4 = nn.ConvTranspose3d(out_dim, out_dim, kernel_size=(1, 1, 1), stride=(1, 1, 1), bias=False)
        self.conv3d_time_5 = nn.ConvTranspose3d(out_dim, out_dim, kernel_size=(1, 1, 1), stride=(1, 1, 1), bias=False)
        torch.nn.init.constant_(self.conv3d_time_1.weight, 0.0)
        torch.nn.init.constant_(self.conv3d_time_2.weight, 0.0)
        torch.nn.init.constant_(self.conv3d_time_3.weight, 0.0)
        torch.nn.init.constant_(self.conv3d_time_4.weight, 0.0)
        torch.nn.init.eye_(self.conv3d_time_5.weight[:, :, 0, 0, 0])
        self.batch = nn.BatchNorm3d(out_dim)
        self.leaky =  nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        x_spatial = self.conv3d_spatial(x)
        T1 = x_spatial[:, :, 0:1, :, :]
        T2 = x_spatial[:, :, 1:2, :, :]
        T3 = x_spatial[:, :, 2:3, :, :]
        T4 = x_spatial[:, :, 3:4, :, :]
        T1_F1 = self.conv3d_time_5(T1)
        T2_F1 = self.conv3d_time_5(T2)
        T3_F1 = self.conv3d_time_5(T3)
        T4_F1 = self.conv3d_time_5(T4)
        T1_F2 = self.conv3d_time_1(T1)
        T2_F2 = self.conv3d_time_2(T2)
        T3_F2 = self.conv3d_time_3(T3)
        T4_F2 = self.conv3d_time_4(T4)
        x = torch.cat([T1_F1 + T2_F2, T2_F1 + T3_F2,  T3_F1 + T4_F2,  T4_F1 + T1_F2], dim=2)
        x = self.batch(x)
        x = self.leaky(x)

        return x


class conv_block_2_3d(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(conv_block_2_3d, self).__init__()
        self.conv3d_s = nn.Conv3d(in_dim, out_dim, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1))
        self.conv3d_t1 = nn.Conv3d(out_dim, out_dim, kernel_size=(1, 1, 1), stride=(1, 1, 1), bias=False)
        self.conv3d_t2 = nn.Conv3d(out_dim, out_dim, kernel_size=(1, 1, 1), stride=(1, 1, 1), bias=False)
        self.conv3d_t3 = nn.Conv3d(out_dim, out_dim, kernel_size=(1, 1, 1), stride=(1, 1, 1), bias=False)
        self.conv3d_t4 = nn.Conv3d(out_dim, out_dim, kernel_size=(1, 1, 1), stride=(1, 1, 1), bias=False)
        self.conv3d_t5 = nn.Conv3d(out_dim, out_dim, kernel_size=(1, 1, 1), stride=(1, 1, 1), bias=False)
        torch.nn.init.constant_(self.conv3d_t1.weight, 0.0)
        torch.nn.init.constant_(self.conv3d_t2.weight, 0.0)
        torch.nn.init.constant_(self.conv3d_t3.weight, 0.0)
        torch.nn.init.constant_(self.conv3d_t4.weight, 0.0)
        torch.nn.init.eye_(self.conv3d_t5.weight[:, :, 0, 0, 0])
        self.b = nn.BatchNorm3d(out_dim)
        self.l =  nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        x_s = self.conv3d_s(x)
        T1 = x_s[:, :, 0:1, :, :]
        T2 = x_s[:, :, 1:2, :, :]
        T3 = x_s[:, :, 2:3, :, :]
        T4 = x_s[:, :, 3:4, :, :]
        T1_F1 = self.conv3d_t5(T1)
        T2_F1 = self.conv3d_t5(T2)
        T3_F1 = self.conv3d_t5(T3)
        T4_F1 = self.conv3d_t5(T4)
        T1_F2 = self.conv3d_t1(T1)
        T2_F2 = self.conv3d_t2(T2)
        T3_F2 = self.conv3d_t3(T3)
        T4_F2 = self.conv3d_t4(T4)
        x = torch.cat([T1_F1 + T2_F2, T2_F1 + T3_F2,  T3_F1 + T4_F2,  T4_F1 + T1_F2], dim=2)
        x = self.b(x)
        x = self.l(x)

        return x

def max_pooling_3d():
    return nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2), padding=0)

class Transformer_Encoder(nn.Module):
    def __init__(self, in_chan=16, token_len=8, heads=8):
        super(Transformer_Encoder, self).__init__()
        self.token_len = token_len
        # This conv layer will now operate on individual time steps
        self.conv_a = nn.Conv2d(in_chan, token_len, kernel_size=1, padding=0)
        # Positional embedding for each time step
        self.pos_embedding = nn.Parameter(torch.randn(4, token_len, in_chan))  # 4 for the time steps
        self.transformer = Transformer(dim=in_chan*4, depth=1, heads=heads, dim_head=in_chan*4, mlp_dim=in_chan*4, dropout=0)

    def forward(self, x):
        b, c, t, h, w = x.shape
        assert t == 4, "The time dimension (t) must be 4."
        
        # Process each time step independently
        tokens_all_time_steps = []
        for time_step in range(t):
            x_t = x[:, :, time_step, :, :]
            spatial_attention = self.conv_a(x_t)
            spatial_attention = spatial_attention.view(b, self.token_len, -1).contiguous()
            spatial_attention = torch.softmax(spatial_attention, dim=-1)
            x_t = x_t.view(b, c, -1).contiguous()
            tokens = torch.einsum('bln, bcn -> blc', spatial_attention, x_t)
            tokens += self.pos_embedding[time_step]
            tokens_all_time_steps.append(tokens)
        
        tokens_concat = torch.cat(tokens_all_time_steps, dim=2)  
        tokens_transformed = self.transformer(tokens_concat)

        return tokens_transformed


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x

        return x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):

        return self.fn(self.norm(x), **kwargs)


class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x)

        qkv = qkv.chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        out = torch.matmul(attn, v)

        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)

        return out


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):

        return self.net(x)


class Transformer_Decoder(nn.Module):
    def __init__(self, in_chan = 64, heads = 8):
        super(Transformer_Decoder, self).__init__()
        self.transformer_decoder = TransformerDecoder(dim=in_chan, depth=1, heads=heads, dim_head=True, mlp_dim=in_chan*2, dropout=0,softmax=in_chan)

    def forward(self, x, m):
        b, c, t, h, w = x.shape
        x = x.view(b, c * t, h, w)
        x = rearrange(x, 'b c h w -> b (h w) c')
        x = self.transformer_decoder(x, m)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)

        return x


class TransformerDecoder(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout, softmax=True):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual2(PreNorm2(dim, Cross_Attention(dim, heads = heads,
                                                        dim_head = dim_head, dropout = dropout,
                                                        softmax=softmax))),
                Residual(PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout)))
            ]))

    def forward(self, x, m, mask = None):
        """target(query), memory"""
        for attn, ff in self.layers:
            x = attn(x, m, mask = mask)
            x = ff(x)

        return x


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):

        return self.fn(x, **kwargs) + x


class Residual2(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, x2, **kwargs):

        return self.fn(x, x2, **kwargs) + x


class PreNorm2(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, x2, **kwargs):

        return self.fn(self.norm(x), self.norm(x2), **kwargs)


class Cross_Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0., softmax=True):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim ** -0.5

        self.softmax = softmax
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_k = nn.Linear(dim, inner_dim, bias=False)
        self.to_v = nn.Linear(dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, m, mask = None):

        b, n, _, h = *x.shape, self.heads
        q = self.to_q(x)
        k = self.to_k(m)
        v = self.to_v(m)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), [q,k,v])

        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        mask_value = -torch.finfo(dots.dtype).max

        if mask is not None:
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = mask[:, None, :] * mask[:, :, None]
            dots.masked_fill_(~mask, mask_value)
            del mask

        if self.softmax:
            attn = dots.softmax(dim=-1)
        else:
            attn = dots

        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)

        return out


class Classifier(nn.Module):
    def __init__(self, in_chan, n_class, scale=2, pad=0):
        super(Classifier, self).__init__()
        self.conv1 = nn.Conv2d(in_chan, n_class * scale * scale, kernel_size=1, padding=pad, bias=False)
        self.scale = scale

    def forward(self, x):
        x = self.conv1(x)
        N, C, H, W = x.size()

        # N, H, W, C
        x_permuted = x.permute(0, 2, 3, 1)

        # N, H, W*scale, C/scale
        x_permuted = x_permuted.contiguous().view((N, H, W * self.scale, int(C / (self.scale))))

        # N, W*scale,H, C/scale
        x_permuted = x_permuted.permute(0, 2, 1, 3)
        # N, W*scale,H*scale, C/(scale**2)
        x_permuted = x_permuted.contiguous().view(
            (N, W * self.scale, H * self.scale, int(C / (self.scale * self.scale))))

        x = x_permuted.permute(0, 3, 2, 1)

        return x

class OFW(nn.Module):
    def __init__(self, inplane):
        super(OFW, self).__init__()

        self.down = nn.Sequential(
            nn.Conv3d(inplane, inplane, kernel_size=(3,3,3), stride=(1,2,2), padding=(1,1,1), groups=inplane),
            nn.BatchNorm3d(inplane),
            nn.ReLU(inplace=True),
            nn.Conv3d(inplane, inplane, kernel_size=(3,3,3), stride=(1,2,2), padding=(1,1,1), groups=inplane),
            nn.BatchNorm3d(inplane),
            nn.ReLU(inplace=True),
            nn.Conv3d(inplane, inplane, kernel_size=(3,3,3), stride=(1,2,2), padding=(1,1,1), groups=inplane),
            nn.BatchNorm3d(inplane),
            nn.ReLU(inplace=True)
        )
        self.flow_make = nn.Conv3d(inplane * 2, 2, kernel_size=(3,3,3), padding=(1,1,1), bias=False)

    def forward(self, x):
        size = x.size()[3:]
        seg_down = self.down(x)
        seg_down = F.interpolate(seg_down, size=(2, 128, 128), mode='trilinear', align_corners=True)
        flow = self.flow_make(torch.cat([x, seg_down], dim=1))
        seg_flow_warp = self.flow_warp(x, flow, size)
        return seg_flow_warp

    def flow_warp(self, input, flow, size):
        B, C, T, H, W = input.size()
        _, _, _, flow_H, flow_W = flow.size()
        out_h, out_w = size
    
        h_grid = torch.linspace(-1.0, 1.0, out_h).view(-1, 1).repeat(1, out_w)
        w_grid = torch.linspace(-1.0, 1.0, out_w).repeat(out_h, 1)
        grid = torch.cat((w_grid.unsqueeze(2), h_grid.unsqueeze(2)), 2)

        grid = grid.repeat(B, 1, 1, 1).type_as(input).to(input.device)
        output_list = []
        for t in range(T):
            current_frame_input = input[:, :, t, :, :]
            current_frame_flow = flow[:, :, t, :, :]

            norm = torch.tensor([[[[flow_W, flow_H]]]]).type_as(input).to(input.device)

            flow_field = current_frame_flow.permute(0, 2, 3, 1) / norm

            warped_frame = F.grid_sample(current_frame_input, (grid + flow_field).clamp(-1, 1), mode='bilinear', padding_mode='border', align_corners=True)
            output_list.append(warped_frame.unsqueeze(2))

        x1 = input[:, :, 0:1, :, :]
        x2 = input[:, :, 1:2, :, :]
        output = torch.cat([x1] + output_list + [x2], dim=2)
        
        return output

model_urls = {
    'mobilenet_v2': 'https://download.pytorch.org/models/mobilenet_v2-b0353104.pth',
}

class ConvBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1, dilation=1):
        padding = (kernel_size - 1) // 2
        if dilation != 1:
            padding = dilation
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, dilation=dilation,
                      bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU6(inplace=True)
        )

class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio, dilation=1):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            # pw
            layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1))
        layers.extend([
            # dw
            ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim, dilation=dilation),
            # pw-linear
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

class MobileNetV2(nn.Module):
    def __init__(self, pretrained=None, num_classes=1000, width_mult=1.0):
        super(MobileNetV2, self).__init__()
        block = InvertedResidual
        input_channel = 32
        last_channel = 1280
        inverted_residual_setting = [
            # t, c, n, s, d
            [1, 16, 1, 1, 1],
            [6, 24, 2, 2, 1],
            [6, 32, 3, 2, 1],
            [6, 64, 4, 2, 1],
            [6, 96, 3, 1, 1],
            [6, 160, 3, 2, 1],
            [6, 320, 1, 1, 1],
        ]

        # building first layer
        input_channel = int(input_channel * width_mult)
        self.last_channel = int(last_channel * max(1.0, width_mult))
        features = [ConvBNReLU(3, input_channel, stride=2)]
        # building inverted residual blocks
        for t, c, n, s, d in inverted_residual_setting:
            output_channel = int(c * width_mult)
            for i in range(n):
                stride = s if i == 0 else 1
                dilation = d if i == 0 else 1
                features.append(block(input_channel, output_channel, stride, expand_ratio=t, dilation=d))
                input_channel = output_channel
        # building last several layers
        features.append(ConvBNReLU(input_channel, self.last_channel, kernel_size=1))
        # make it nn.Sequential
        self.features = nn.Sequential(*features)

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        res = []
        for idx, m in enumerate(self.features):
            x = m(x)
            if idx in [1, 3, 6, 13, 17]:
                res.append(x)
        return res

def mobilenet_v2(pretrained=True, progress=True, **kwargs):
    model = MobileNetV2(**kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls['mobilenet_v2'],
                                              progress=progress)
        print("loading imagenet pretrained mobilenetv2")
        model.load_state_dict(state_dict, strict=False)
        print("loaded imagenet pretrained mobilenetv2")
    return model