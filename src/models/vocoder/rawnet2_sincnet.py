import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class SincConv1d(nn.Module):
    def __init__(self, num_filters, kernel_size, sample_rate=16000,
                 min_low_hz=50.0, min_band_hz=50.0, stride=1, padding=0):
        super().__init__()
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.sample_rate = sample_rate
        self.min_low_hz = min_low_hz
        self.min_band_hz = min_band_hz
        self.stride = stride
        self.padding = padding

        hz_to_mel = lambda hz: 2595 * np.log10(1 + hz / 700)
        mel_to_hz = lambda m: 700 * (10**(m / 2595) - 1)
        
        high_hz = sample_rate / 2 - (min_low_hz + min_band_hz)
        mel_points = np.linspace(hz_to_mel(min_low_hz), hz_to_mel(high_hz), num_filters + 1)
        hz_points = mel_to_hz(mel_points)
        
        self.low_hz_ = nn.Parameter(torch.Tensor(hz_points[:-1]).view(-1, 1))
        self.band_hz_ = nn.Parameter(torch.Tensor(np.diff(hz_points)).view(-1, 1))

        n_lin = torch.linspace(0, (kernel_size / 2) - 1, steps=int(kernel_size / 2))
        self.register_buffer('window_', 0.54 - 0.46 * torch.cos(2 * np.pi * n_lin / kernel_size))
        self.register_buffer('n_', 2 * np.pi * n_lin / sample_rate)

    def forward(self, x):
        low = self.min_low_hz + torch.abs(self.low_hz_)
        high = torch.clamp(low + self.min_band_hz + torch.abs(self.band_hz_), self.min_low_hz, self.sample_rate / 2)
        band = (high - low)[:, 0]
        
        n_view = self.n_.view(1, -1)
        window_view = self.window_.view(1, -1)
        
        f_times_t_low = torch.matmul(low, n_view)
        f_times_t_high = torch.matmul(high, n_view)

        band_pass_left = ((torch.sin(f_times_t_high) - torch.sin(f_times_t_low)) / (n_view / 2)) * window_view
        band_pass_center = 2 * band.view(-1, 1)
        band_pass_right = torch.flip(band_pass_left, dims=[1])

        band_pass = torch.cat([band_pass_left, band_pass_center, band_pass_right], dim=1)
        band_pass = band_pass / (2 * band[:, None])
        self.filters = band_pass.view(self.num_filters, 1, self.kernel_size)
        
        return F.conv1d(x, self.filters, stride=self.stride, padding=self.padding)

class FMS(nn.Module):
    def __init__(self, num_channels):
        super().__init__()
        self.fc = nn.Linear(num_channels, num_channels)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        y = x.mean(dim=-1)
        y = self.fc(y)
        y = self.sig(y).unsqueeze(-1)
        return x * y + x

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, first=False):
        super().__init__()
        self.first = first
        if not first:
            self.bn1 = nn.BatchNorm1d(in_channels)
        self.lrelu = nn.LeakyReLU(0.3)
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        
        if in_channels != out_channels or first:
            self.downsample = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm1d(out_channels)
            )
        else:
            self.downsample = None
            
        self.mp = nn.MaxPool1d(3)
        self.fms = FMS(out_channels)

    def forward(self, x):
        identity = x
        
        if not self.first:
            out = self.bn1(x)
            out = self.lrelu(out)
        else:
            out = x
            
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.lrelu(out)
        out = self.conv2(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
            
        out = out + identity
        out = self.fms(out)
        out = self.mp(out)
        return out

class RawNet2WithSincNet(nn.Module):
    def __init__(self, sinc_num_filters, sinc_kernel_size, sinc_min_low_hz,
                 sinc_min_band_hz, filts, gru_node, nb_gru_layer,
                 nb_fc_node, nb_classes, sample_rate):
        super().__init__()
        self.sinc_conv = SincConv1d(num_filters=sinc_num_filters, kernel_size=sinc_kernel_size,
                               sample_rate=sample_rate, min_low_hz=sinc_min_low_hz,
                               min_band_hz=sinc_min_band_hz)
        
        self.bn_sinc = nn.BatchNorm1d(sinc_num_filters)
        self.sinc_lrelu = nn.LeakyReLU(0.3)
        self.sinc_mp = nn.MaxPool1d(3)

        in_channels = sinc_num_filters
        
        if len(filts) == 2:
            out_channels_list = [filts[0], filts[0], filts[1], filts[1], filts[1], filts[1]]
        else:
            out_channels_list = filts

        self.num_blocks = len(out_channels_list)
        for i, out_channels in enumerate(out_channels_list):
            self.add_module(f"block{i}", ResBlock(in_channels, out_channels, first=(i==0)))
            in_channels = out_channels

        self.bn_before_gru = nn.BatchNorm1d(out_channels_list[-1])
        
        self.gru = nn.GRU(input_size=out_channels_list[-1], hidden_size=gru_node,
                          num_layers=nb_gru_layer, batch_first=True)
                          
        self.fc = nn.Linear(gru_node, nb_fc_node)
        self.out = nn.Linear(nb_fc_node, nb_classes)

    def forward(self, x):
        x = self.sinc_conv(x)
        x = self.bn_sinc(x)
        x = self.sinc_lrelu(x)
        x = self.sinc_mp(x)

        for i in range(self.num_blocks):
            block = getattr(self, f"block{i}")
            x = block(x)

        x = self.bn_before_gru(x)
        x = F.leaky_relu(x, 0.3)
        x = x.transpose(1, 2)
        
        x, _ = self.gru(x)
        x = x[:, -1, :]
        x = self.fc(x)
        x = F.leaky_relu(x, 0.3)
        x = self.out(x)
        return x
