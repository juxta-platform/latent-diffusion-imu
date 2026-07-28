import torch
import torch.nn as nn
import torch.nn.functional as F


def nonlinearity(x):
    return F.silu(x)


def Normalize(in_channels, num_groups=32):
    return nn.GroupNorm(num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True)


class ResnetBlock1D(nn.Module):
    def __init__(self, *, in_channels, out_channels=None, temb_channels=0, dropout=0.0):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels

        self.norm1 = Normalize(in_channels)
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        if temb_channels > 0:
            self.temb_proj = nn.Linear(temb_channels, out_channels)
        self.norm2 = Normalize(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        if self.in_channels != self.out_channels:
            self.nin_shortcut = nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x, temb=None):
        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)

        if temb is not None:
            h = h + self.temb_proj(nonlinearity(temb))[:, :, None]

        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            x = self.nin_shortcut(x)

        return x + h


class AttnBlock1D(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels
        self.norm = Normalize(in_channels)
        self.q = nn.Conv1d(in_channels, in_channels, kernel_size=1)
        self.k = nn.Conv1d(in_channels, in_channels, kernel_size=1)
        self.v = nn.Conv1d(in_channels, in_channels, kernel_size=1)
        self.proj_out = nn.Conv1d(in_channels, in_channels, kernel_size=1)

    def forward(self, x):
        h_ = self.norm(x)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        b, c, t = q.shape
        q = q.permute(0, 2, 1)       # b, t, c
        w_ = torch.bmm(q, k)         # b, t, t
        w_ = w_ * (int(c) ** (-0.5))
        w_ = F.softmax(w_, dim=2)

        w_ = w_.permute(0, 2, 1)     # b, t, t
        h_ = torch.bmm(v, w_)        # b, c, t

        h_ = self.proj_out(h_)
        return x + h_


class Downsample1D(nn.Module):
    def __init__(self, in_channels, stride=2):
        super().__init__()
        self.stride = stride
        if stride == 2:
            self.conv = nn.Conv1d(in_channels, in_channels, kernel_size=3, stride=2, padding=0)
        else:
            self.conv = nn.Conv1d(in_channels, in_channels, kernel_size=stride, stride=stride, padding=0)

    def forward(self, x):
        if self.stride == 2:
            x = F.pad(x, (0, 1), mode="constant", value=0)
        x = self.conv(x)
        return x


class Upsample1D(nn.Module):
    def __init__(self, in_channels, factor=2):
        super().__init__()
        self.factor = factor
        self.conv = nn.Conv1d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=float(self.factor), mode="nearest")
        x = self.conv(x)
        return x


class Encoder1D(nn.Module):
    def __init__(self, *, ch, in_channels, z_channels, ch_mult=(1, 2, 4),
                 num_res_blocks=2, dropout=0.0, downsample_factors=(2, 2, 5),
                 double_z=True, **ignore_kwargs):
        super().__init__()
        self.ch = ch
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks

        self.conv_in = nn.Conv1d(in_channels, ch, kernel_size=3, padding=1)

        block_in = ch
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            block_out = ch * ch_mult[i_level]
            for i_block in range(num_res_blocks):
                block.append(ResnetBlock1D(in_channels=block_in, out_channels=block_out, dropout=dropout))
                block_in = block_out
            down = nn.Module()
            down.block = block
            down.downsample = Downsample1D(block_in, stride=downsample_factors[i_level])
            self.down.append(down)

        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock1D(in_channels=block_in, out_channels=block_in, dropout=dropout)
        self.mid.attn_1 = AttnBlock1D(block_in)
        self.mid.block_2 = ResnetBlock1D(in_channels=block_in, out_channels=block_in, dropout=dropout)

        self.norm_out = Normalize(block_in)
        self.conv_out = nn.Conv1d(block_in, 2 * z_channels if double_z else z_channels, kernel_size=3, padding=1)

    def forward(self, x):
        h = self.conv_in(x)

        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](h)
            h = self.down[i_level].downsample(h)

        h = self.mid.block_1(h)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h)

        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h


class Decoder1D(nn.Module):
    def __init__(self, *, ch, out_channels, z_channels, ch_mult=(1, 2, 4),
                 num_res_blocks=2, dropout=0.0, upsample_factors=(5, 2, 2),
                 **ignore_kwargs):
        super().__init__()
        self.ch = ch
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks

        block_in = ch * ch_mult[-1]
        self.conv_in = nn.Conv1d(z_channels, block_in, kernel_size=3, padding=1)

        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock1D(in_channels=block_in, out_channels=block_in, dropout=dropout)
        self.mid.attn_1 = AttnBlock1D(block_in)
        self.mid.block_2 = ResnetBlock1D(in_channels=block_in, out_channels=block_in, dropout=dropout)

        self.up = nn.ModuleList()
        upsample_idx = 0
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            block_out = ch * ch_mult[i_level]
            for i_block in range(num_res_blocks + 1):
                block.append(ResnetBlock1D(in_channels=block_in, out_channels=block_out, dropout=dropout))
                block_in = block_out
            up = nn.Module()
            up.block = block
            up.upsample = Upsample1D(block_in, factor=upsample_factors[upsample_idx])
            upsample_idx += 1
            self.up.insert(0, up)

        self.norm_out = Normalize(block_in)
        self.conv_out = nn.Conv1d(block_in, out_channels, kernel_size=3, padding=1)

    def forward(self, z):
        h = self.conv_in(z)

        h = self.mid.block_1(h)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h)

        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](h)
            h = self.up[i_level].upsample(h)

        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h
