import torch
import torch.nn as nn
import torch.nn.functional as F

class FCA(nn.Module):
    def __init__(self, inp, oup, mode="tf"):
        super(FCA, self).__init__()

        layers = [
            nn.AvgPool2d(kernel_size=2, stride=2, padding=0),
            nn.Conv2d(inp, oup, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(oup),
        ]

        mode_weights = {
            "temp": [
                nn.Conv2d(oup, oup, kernel_size=(1, 5), padding=(0, 2), groups=oup, bias=False),
                nn.BatchNorm2d(oup),
                nn.Conv2d(oup, oup, kernel_size=(1, 5), padding=(0, 2), groups=oup, bias=False),
                nn.BatchNorm2d(oup),
            ],
            "freq": [
                nn.Conv2d(oup, oup, kernel_size=(5, 1), padding=(2, 0), groups=oup, bias=False),
                nn.BatchNorm2d(oup),
                nn.Conv2d(oup, oup, kernel_size=(5, 1), padding=(2, 0), groups=oup, bias=False),
                nn.BatchNorm2d(oup),
            ],
            "tf": [
                nn.Conv2d(oup, oup, kernel_size=(1, 5), padding=(0, 2), groups=oup, bias=False),
                nn.BatchNorm2d(oup),
                nn.Conv2d(oup, oup, kernel_size=(5, 1), padding=(2, 0), groups=oup, bias=False),
                nn.BatchNorm2d(oup),
            ],
        }

        layers.extend(mode_weights.get(mode, []))

        layers.append(nn.Sigmoid())

        self.attn = nn.Sequential(*layers)

    def forward(self, x):
        # x: (B, C, F, T)
        attn_map = F.interpolate(self.attn(x), size=(x.shape[-2], x.shape[-1]), mode='nearest')
        return attn_map

class Sandglass(nn.Module):
    def __init__(self, inp, oup, mid, ksize=3, stride=1):
        super(Sandglass, self).__init__()
        
        # First depthwise convolution
        self.dw1 = nn.Sequential(
            nn.Conv2d(inp, inp, kernel_size=ksize, stride=stride, padding=ksize // 2, bias=False, groups=inp),
            nn.BatchNorm2d(inp),
            nn.ReLU6(inplace=True),
        )
        
        # Pointwise convolution (reduce)
        self.pw_reduce = nn.Sequential(
            nn.Conv2d(inp, mid, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(mid),
        )
        
        # Pointwise convolution (expand)
        self.pw_expand = nn.Sequential(
            nn.Conv2d(mid, oup, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(oup),
            nn.ReLU6(inplace=True),
        )
        
        # Second depthwise convolution
        self.dw2 = nn.Sequential(
            nn.Conv2d(oup, oup, kernel_size=ksize, stride=stride, padding=ksize // 2, bias=False, groups=oup),
            nn.BatchNorm2d(oup),
            nn.ReLU6(inplace=True),
        )

        self.use_residual = (inp == oup)
        
    def forward(self, x):
        out = self.dw1(x)
        out = self.pw_reduce(out)
        out = self.pw_expand(out)
        out = self.dw2(out)
        if self.use_residual:
            out += x
        return out
        
class FCABlock(nn.Module):
    def __init__(self, inp, oup, mid_channels=None, ksize=3, stride=1, at_mode="tf"):
        super(FCABlock, self).__init__()
        assert stride in [1, 2], "Stride must be 1 or 2"
        mid = mid_channels if mid_channels is not None else oup // 2

        # Attention module
        self.attn = FCA(inp, oup, mode=at_mode)

        self.res1 = nn.Sequential(
            nn.Conv2d(inp, mid, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(mid),
            nn.ReLU6(inplace=True),
        )
        self.res2 = nn.Sequential(
            nn.Conv2d(mid, oup - mid, kernel_size=ksize, stride=1, padding=ksize // 2, groups=mid, bias=False),
            nn.BatchNorm2d(oup - mid),
            nn.ReLU6(inplace=True),
        )

        self.FF = nn.Sequential(
            nn.Conv2d(oup, oup, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(oup),
            nn.ReLU6(inplace=True),
        )

        # Shortcut connection
        if stride == 1 and inp == oup:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inp, inp, kernel_size=ksize, stride=stride, padding=ksize // 2, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                nn.Conv2d(inp, oup, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        r1 = self.res1(x)
        r2 = self.res2(r1)
        res = torch.cat((r1, r2), dim=1)
        attn_map = self.attn(x)
        out = self.FF(attn_map * res) + self.shortcut(x)
        return out



class lmfcaNet(nn.Module):
    def __init__(self, in_ch=6, out_ch=2):
        super(lmfcaNet, self).__init__()

        channels = [48, 96, 224, 480]

        # First Block
        self.firstblock = nn.Sequential(
            FCABlock(inp=in_ch, oup=channels[0]),
            Sandglass(channels[0], channels[0], channels[0] // 2, ksize=3, stride=1),
        )

        # Encoder (Downsampling) Layers
        at_modes = ["freq", "temp", "freq"]
        self.down_blocks = nn.ModuleList()
        for idx in range(3):
            self.down_blocks.append(nn.Sequential(
                FCABlock(channels[idx], channels[idx + 1], ksize=3, stride=1, at_mode=at_modes[idx]),
                Sandglass(channels[idx + 1], channels[idx + 1], channels[idx], ksize=3, stride=1),
                nn.MaxPool2d(2, 2),
            ))

        # Decoder (Upsampling) Layers
        self.up_blocks = nn.ModuleList()
        
        # Up4
        self.up_blocks.append(nn.Sequential(
            Sandglass(channels[3], channels[3], channels[2], ksize=3, stride=1),
            Sandglass(channels[3], channels[3], channels[2], ksize=3, stride=1),
        ))
        # Up3 to Up1
        for idx in range(2, -1, -1):
            self.up_blocks.append(nn.Sequential(
                FCABlock(channels[idx + 1], channels[idx]),
                Sandglass(channels[idx], channels[idx], channels[max(idx - 1, 0)], ksize=3, stride=1),
                nn.ConvTranspose2d(channels[idx], channels[idx], kernel_size=2, stride=2, padding=0),
            ))

        self.lastConv = nn.Sequential(
            Sandglass(channels[0], channels[0], channels[0] // 2, ksize=3, stride=1),
            Sandglass(channels[0], channels[0], channels[0] // 2, ksize=3, stride=1),
            nn.Conv2d(channels[0], out_ch, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(out_ch),
            nn.Tanh(),
        )

        self.initialize_weights()

    def initialize_weights(self):
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, x):
        # Encoder
        e0 = self.firstblock(x)
        e1 = self.down_blocks[0](e0)
        e2 = self.down_blocks[1](e1)
        e3 = self.down_blocks[2](e2)

        # Decoder
        d4 = self.up_blocks[0](e3)

        d3_input = d4 + e3
        d3 = self.up_blocks[1](d3_input)

        d2_input = d3 + e2
        d2 = self.up_blocks[2](d2_input)

        d1_input = d2 + e1
        d1 = self.up_blocks[3](d1_input)

        d1 += e0
        out = self.lastConv(d1)

        return out


 # 音频输入: [B, 2, L]
def wav2spec(wav, n_fft=510, hop=128, win_len=510):
    stft = lambda x: torch.stft(
        x, n_fft=n_fft, hop_length=hop, win_length=win_len,
        return_complex=True, center=True, normalized=False
    )
    specs = [stft(wav[:, i]) for i in range(2)]
    specs = torch.stack(specs, dim=1)  # [B, 2, F, T]
    real_imag = torch.view_as_real(specs)  # [B, 2, F, T, 2]
    real_imag = real_imag.permute(0, 1, 4, 2, 3)  # [B, 2, 2, F, T]
    real_imag = real_imag.reshape(wav.shape[0], 2 * 2, specs[0].shape[-2], specs[0].shape[-1])  # [B, 4, F, T]
    return real_imag


# 模型输出: [B, 2, F, T] -> complex tensor
def spec2wav(output, n_fft=510, hop=128, win_len=510, length=None):
    real, imag = output[:, 0], output[:, 1]  # [B, F, T]
    spec = torch.complex(real, imag)  # [B, F, T]
    wav = torch.istft(
        spec,
        n_fft=n_fft,
        hop_length=hop,
        win_length=win_len,
        center=True,
        normalized=False,
        length=length
    )
    wav = wav.unsqueeze(1) # [B T] ==> [B C=1 T]
    return wav


def test_lmfcaNet():
    # B C=2 T
    noisy_2ch_wav = torch.randn(16, 2, 16256)
    print(f"noisy_2ch_wav.shape: {noisy_2ch_wav.shape}") # torch.Size([1, 2, 16256])
    noisy_2ch_spec = wav2spec(noisy_2ch_wav)
    print(f"noisy_2ch_spec.shape: {noisy_2ch_spec.shape}") # torch.Size([1, 4, 256, 128])

    # Initialize the model
    model = lmfcaNet(in_ch=4, out_ch=2)

    # Forward pass
    est_spec = model(noisy_2ch_spec)

    est_1ch_wav = spec2wav(est_spec)
    print(f"est_1ch_wav.shape: {est_1ch_wav.shape}") # [B C=1 T] torch.Size([16, 1, 16256])


def test_model_complexity_info():
    from ptflops import get_model_complexity_info
    nnet = lmfcaNet(in_ch=4, out_ch=2)
    flops, params = get_model_complexity_info(nnet,
                                              (4, 256, 128),
                                              as_strings=True,
                                              print_per_layer_stat=False)
    print(f'flops:{flops}, params:{params}')
    # flops:5.72 GMac, params:2.13 M

if __name__ == "__main__":    
    test_lmfcaNet()
    test_model_complexity_info()