import torch
import torch.nn as nn
import torch.nn.functional as F

from ldm.modules.diffusionmodules.util import timestep_embedding


class ResBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, emb_channels, dropout=0.0):
        super().__init__()
        groups1 = min(32, in_channels)
        groups2 = min(32, out_channels)

        self.norm1 = nn.GroupNorm(groups1, in_channels)
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.emb_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_channels, out_channels),
        )
        self.norm2 = nn.GroupNorm(groups2, out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)

        if in_channels != out_channels:
            self.skip_proj = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        else:
            self.skip_proj = nn.Identity()

    def forward(self, x, emb):
        h = self.norm1(x)
        h = F.silu(h)
        h = self.conv1(h)
        h = h + self.emb_proj(emb).unsqueeze(-1)
        h = self.norm2(h)
        h = F.silu(h)
        h = self.dropout(h)
        h = self.conv2(h)
        return h + self.skip_proj(x)


class AttentionBlock1D(nn.Module):
    def __init__(self, channels, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.norm = nn.GroupNorm(min(32, channels), channels)
        self.qkv = nn.Linear(channels, channels * 3)
        self.out_proj = nn.Linear(channels, channels)

    def forward(self, x):
        B, C, T = x.shape
        h = self.norm(x)
        h = h.permute(0, 2, 1)  # [B, T, C]
        qkv = self.qkv(h).reshape(B, T, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, heads, T, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]
        scale = self.head_dim ** -0.5
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)  # [B, heads, T, head_dim]
        out = out.permute(0, 2, 1, 3).reshape(B, T, C)
        out = self.out_proj(out)
        out = out.permute(0, 2, 1)  # [B, C, T]
        return x + out


class Downsample1D(nn.Module):
    def __init__(self, channels, stride=2):
        super().__init__()
        self.conv = nn.Conv1d(channels, channels, kernel_size=3, stride=stride, padding=1)

    def forward(self, x):
        return self.conv(x)


class Upsample1D(nn.Module):
    def __init__(self, channels, factor=2):
        super().__init__()
        self.factor = factor
        self.conv = nn.Conv1d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.factor, mode='nearest')
        return self.conv(x)


class UNetModel1D(nn.Module):
    def __init__(
        self,
        in_channels=11,
        out_channels=8,
        model_channels=128,
        channel_mult=(1, 2, 4),
        num_res_blocks=2,
        attention_levels=(1, 2),
        dropout=0.0,
        num_heads=8,
    ):
        super().__init__()
        self.model_channels = model_channels
        emb_channels = model_channels * 4

        self.input_proj = nn.Conv1d(in_channels, model_channels, kernel_size=3, padding=1)

        self.time_embed = nn.Sequential(
            nn.Linear(model_channels, emb_channels),
            nn.SiLU(),
            nn.Linear(emb_channels, emb_channels),
        )

        # Encoder
        self.down_blocks = nn.ModuleList()
        self.down_samples = nn.ModuleList()
        channels_list = [model_channels]
        ch = model_channels
        self.num_levels = len(channel_mult)
        for level, mult in enumerate(channel_mult):
            out_ch = model_channels * mult
            level_blocks = nn.ModuleList()
            for i in range(num_res_blocks):
                block_in = ch if i == 0 else out_ch
                level_blocks.append(ResBlock1D(block_in, out_ch, emb_channels, dropout))
                if level in attention_levels:
                    level_blocks.append(AttentionBlock1D(out_ch, num_heads))
                channels_list.append(out_ch)
            ch = out_ch
            self.down_blocks.append(level_blocks)
            if level < len(channel_mult) - 1:
                self.down_samples.append(Downsample1D(ch))
                channels_list.append(ch)

        # Middle
        self.mid_block = nn.ModuleList([
            ResBlock1D(ch, ch, emb_channels, dropout),
            AttentionBlock1D(ch, num_heads),
            ResBlock1D(ch, ch, emb_channels, dropout),
        ])

        # Decoder
        self.up_blocks = nn.ModuleList()
        self.up_samples = nn.ModuleList()
        for level in reversed(range(len(channel_mult))):
            out_ch = model_channels * channel_mult[level]
            level_blocks = nn.ModuleList()
            for i in range(num_res_blocks + 1):
                skip_ch = channels_list.pop()
                block_in = ch + skip_ch if i == 0 else out_ch + skip_ch
                level_blocks.append(ResBlock1D(block_in, out_ch, emb_channels, dropout))
                if level in attention_levels:
                    level_blocks.append(AttentionBlock1D(out_ch, num_heads))
            ch = out_ch
            self.up_blocks.append(level_blocks)
            if level > 0:
                self.up_samples.append(Upsample1D(ch))

        # Output
        self.out_norm = nn.GroupNorm(min(32, ch), ch)
        self.out_conv = nn.Conv1d(ch, out_channels, kernel_size=3, padding=1)
        nn.init.zeros_(self.out_conv.weight)
        nn.init.zeros_(self.out_conv.bias)

    def forward(self, x, timesteps):
        emb = timestep_embedding(timesteps, self.model_channels)
        emb = self.time_embed(emb)

        h = self.input_proj(x)
        skips = [h]

        # Encoder
        for level, blocks in enumerate(self.down_blocks):
            idx = 0
            while idx < len(blocks):
                h = blocks[idx](h, emb)
                idx += 1
                if idx < len(blocks) and isinstance(blocks[idx], AttentionBlock1D):
                    h = blocks[idx](h)
                    idx += 1
                skips.append(h)
            if level < self.num_levels - 1:
                h = self.down_samples[level](h)
                skips.append(h)

        # Middle
        h = self.mid_block[0](h, emb)
        h = self.mid_block[1](h)
        h = self.mid_block[2](h, emb)

        # Decoder
        for level, blocks in enumerate(self.up_blocks):
            idx = 0
            while idx < len(blocks):
                block = blocks[idx]
                if isinstance(block, ResBlock1D):
                    h = torch.cat([h, skips.pop()], dim=1)
                    h = block(h, emb)
                else:
                    h = block(h)
                idx += 1
            if level < len(self.up_samples):
                h = self.up_samples[level](h)

        h = self.out_norm(h)
        h = F.silu(h)
        h = self.out_conv(h)
        return h
