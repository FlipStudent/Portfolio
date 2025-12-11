# Source: positional encoding adapted from
# (https://medium.com/@hunter-j-phillips/positional-encoding-7a93db4109e6)
# Source: U-Net adapted from pytorch-diffusion by awjuliani
# (https://github.com/awjuliani/pytorch-diffusion/tree/master)

import src.modules as md

import torch
import torch.nn as nn
import torch.nn.functional as F


# Source: U-Net adapted from pytorch-diffusion by awjuliani
# (https://github.com/awjuliani/pytorch-diffusion/tree/master)
class DiffusionUNet(nn.Module):
    def __init__(self, in_size, t_range, img_depth, device):
        super().__init__()
        self.forward_process = ForwardProcess(device=device)
        self.in_size = in_size
        self.device = device

        bilinear = True
        self.inc = md.DoubleConv(img_depth, 32)
        self.down1 = md.Down(32, 64)
        self.down2 = md.Down(64, 128)
        factor = 2 if bilinear else 1
        self.down3 = md.Down(128, 256 // factor)
        self.sa = md.SAWrapper(128, 12)
        self.up1 = md.Up(256, 128 // factor, bilinear)
        self.up2 = md.Up(128, 64 // factor, bilinear)
        self.up3 = md.Up(64, 32, bilinear)
        self.outc = md.OutConv(32, img_depth)

    def pos_encoding(self, t, channels, embed_size):
        # device = time.device
        inv_freq = 1.0 / (
            10000
            ** (torch.arange(0, channels, 2, device=self.device).float() / channels)
        )
        batch_size = t.shape[0]
        t = t.reshape((batch_size, 1))
        inv_freq = inv_freq.unsqueeze(0)
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc.view(-1, channels, 1, 1).repeat(1, 1, embed_size, embed_size)

    def forward(self, x, t):
        x1 = self.inc(x)
        x2 = self.down1(x1) + self.pos_encoding(t, 64, self.in_size // 2)
        x3 = self.down2(x2) + self.pos_encoding(t, 128, self.in_size // 4)
        x4 = self.down3(x3) + self.pos_encoding(t, 128, self.in_size // 8)
        x4 = self.sa(x4)
        x = self.up1(x4, x3) + self.pos_encoding(t, 64, self.in_size // 4)
        x = self.up2(x, x2) + self.pos_encoding(t, 32, self.in_size // 2)
        x = self.up3(x, x1) + self.pos_encoding(t, 32, self.in_size)
        x = self.outc(x)
        return x

    def get_loss(self, noise_pred, noise, t):
        temp_loss = F.mse_loss(noise_pred, noise, reduction='none')
        temp_loss = torch.mean(temp_loss, dim=(1, 2, 3))
        scale = 2 * self.forward_process.beta[t]
        return torch.mean(temp_loss / scale)


# Source: positional encoding adapted from
# (https://medium.com/@hunter-j-phillips/positional-encoding-7a93db4109e6)
class PositionalEncoding():
    def __init__(self, d_model, max_len=1000, n=100):
        self.d_model = d_model
        self.max_len = max_len
        self.n = n
        self.pe = self.get_positional_encoding()

    def get_positional_encoding(self):
        pe = torch.zeros(self.max_len, self.d_model)
        for k in range(0, self.max_len):
            for i in range(0, self.d_model, 2):
                theta = k / (self.n ** ((2*i)/self.d_model))
                pe[k, 2*i] = torch.sin(theta)
                pe[k, 2*i+1] = torch.cos(theta)
        return pe


class ForwardProcess():
    def __init__(self, device, T=1000):
        self.device = device
        self.T = T
        self.beta = torch.arange(start=0.001, end=0.02, step=(0.02-0.001)/self.T)
        self.alpha = 1 - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)
        self.beta = self.beta.to(self.device)
        self.alpha_bar = self.alpha_bar.to(self.device)

    def __call__(self, x0: torch.Tensor, t):
        batch_size = x0.shape[0]
        random = torch.randn(size=x0.shape).to(self.device)
        noise = torch.sqrt(1-self.alpha_bar[t]).reshape(batch_size, 1, 1, 1)*random
        x_noised = torch.sqrt(self.alpha_bar[t]).reshape(batch_size, 1, 1, 1)*x0 + noise.to(self.device)
        return x_noised, noise
