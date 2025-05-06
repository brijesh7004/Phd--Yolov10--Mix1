import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict


def dimOfVariant(variant='n'):
    config = {
        'n': {'depth': 0.33, 'width': 0.25, 'mc': 1024, 'backbone_ch': [16,32,64,128,256], 'maxCh':384, 'c2fCh':192, 'detect_conv_channel': [64,80] },
        's': {'depth': 0.33, 'width': 0.50, 'mc': 1024, 'backbone_ch': [32,64,128,256,512], 'maxCh':768, 'c2fCh':384, 'detect_conv_channel': [64,128]},
        'm': {'depth': 0.67, 'width': 0.75, 'mc': 768, 'backbone_ch': [48,96,192,384,576], 'maxCh':960, 'c2fCh':576, 'detect_conv_channel': [64,192]},
        'b': {'depth': 0.67, 'width': 1.00, 'mc': 512, 'backbone_ch': [48,96,192,384,768], 'maxCh':960, 'c2fCh':768, 'detect_conv_channel': [64,192]},
        'l': {'depth': 1.00, 'width': 1.00, 'mc': 512, 'backbone_ch': [64,128,256,512,512], 'maxCh':1024, 'c2fCh':768, 'detect_conv_channel': [64,256]},
        'x': {'depth': 1.00, 'width': 1.25, 'mc': 512, 'backbone_ch': [80,160,320,640,640], 'maxCh':1280, 'c2fCh':960, 'detect_conv_channel': [80,320]},
    }[variant]
    return config

class Conv(nn.Module):
    """Standard convolution with batch norm and activation."""
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=None, groups=1, bias=False, act=True):
        super().__init__()
        # print(f"Input shape: {in_channels.shape}")
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride,
                              padding=kernel_size//2 if padding is None else padding,
                              groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.03)
        self.act = nn.SiLU(inplace=True) if act else nn.Identity()

    def forward(self, x):
        # print(f"Conv input: {x.shape}, weight: {self.conv.weight.shape}")
        return self.act(self.bn(self.conv(x)))
    
class RepVGGDW(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = Conv(channels, channels, kernel_size=7, stride=1, padding=3, groups=channels, act=False)
        self.conv1 = Conv(channels, channels, kernel_size=3, stride=1, padding=1, groups=channels, act=False)
        self.act = nn.SiLU(inplace=True) #if act else nn.Identity()

    def forward(self, x):
        return self.act(self.conv1(self.conv(x)))  #self.conv(x) + self.conv1(x)

class Bottleneck(nn.Module):
    """Standard bottleneck layer."""
    def __init__(self, c, shortcut=True):
        super().__init__()
        self.cv1 = Conv(c, c, 3)
        self.cv2 = Conv(c, c, 3)
        self.add = shortcut

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))

class C2f(nn.Module):
    """C2f module with multiple bottleneck layers."""
    def __init__(self, in_channels, out_channels, n=1, shortcut=True):
        super().__init__()
        self.c = out_channels // 2
        self.cv1 = Conv(in_channels, 2 * self.c, 1)
        self.cv2 = Conv((2 + n) * self.c, out_channels, 1)
        self.m = nn.ModuleList(Bottleneck(self.c, shortcut) for _ in range(n))
        
    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

class CIB(nn.Module):
    """CIB block with depthwise convolutions."""
    def __init__(self, c, isRepVGGDW=False):
        super().__init__()
        self.cv1 = nn.Sequential(
            Conv(c, c, 3, groups=c),
            Conv(c, 2 * c, 1),
            Conv(2 * c, 2 * c, 3, groups=2 * c) if not isRepVGGDW else RepVGGDW(2 * c),
            Conv(2 * c, c, 1),
            Conv(c, c, 3, groups=c)
        )

    def forward(self, x):
        return x + self.cv1(x)

class C2fCIB(nn.Module):
    """C2f module with CIB blocks."""
    def __init__(self, in_channels, out_channels, n=1, isRepVGGDW=False):
        super().__init__()
        self.c = out_channels // 2
        self.cv1 = Conv(in_channels, 2 * self.c, 1)
        self.cv2 = Conv((2 + n) * self.c, out_channels, 1)
        self.m = nn.ModuleList(CIB(self.c, isRepVGGDW=isRepVGGDW) for _ in range(n))
        
    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

class SCDown(nn.Module):
    """Spatial-channel downsampling block."""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.cv1 = Conv(in_channels, out_channels, 1)
        self.cv2 = Conv(out_channels, out_channels, 3, stride=2, groups=out_channels, act=False)

    def forward(self, x):
        return self.cv2(self.cv1(x))

class SPPF(nn.Module):
    """SPPF layer with max pooling."""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.c = in_channels // 2
        self.cv1 = Conv(in_channels, self.c, 1)
        self.cv2 = Conv(self.c * 4, out_channels, 1)
        self.m = nn.MaxPool2d(5, 1, 2)

    def forward(self, x):
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        return self.cv2(torch.cat([x, y1, y2, self.m(y2)], 1))

class PSA(nn.Module):
    """PSA attention module."""
    def __init__(self, channels):
        super().__init__()
        self.c = channels // 2
        self.cv1 = Conv(channels, channels, 1)
        self.cv2 = Conv(channels, channels, 1)

        self.attn = nn.Sequential(
                OrderedDict([
                  ('qkv', Conv(self.c, channels, 1, act=False)),
                  ('proj', Conv(channels, self.c, 1, act=False)),
                  ('pe', Conv(self.c, self.c, 3, groups=self.c, act=False)),
                ])
        )

        # # Attention components
        # self.qkv = Conv(self.c, channels, 1, act=False)  # Query/Key/Value projection
        # self.proj = Conv(channels, self.c, 1, act=False)  # Projection back
        # self.pe = Conv(self.c, self.c, 3, groups=self.c, act=False)  # Position encoding
        
        # self.attn = nn.Sequential(
        #     Conv(self.c, channels, 1, act=False),
        #     Conv(channels, self.c, 1, act=False),
        #     Conv(self.c, self.c, 3, groups=self.c, act=False)
        # )
        
        self.ffn = nn.Sequential(
            Conv(self.c, channels, 1),
            Conv(channels, self.c, 1, act=False)
        )

    def forward(self, x):
        x = self.cv1(x)
        a, b = x.chunk(2, 1)

        # Process attention branch
        attn_out = self.attn(b)
        # qkv, proj, pe = attn_out[0], attn_out[1], attn_out[2]
        b_attn = attn_out + b  # Residual connections

        #  # Attention processing
        # qkv = self.qkv(b)
        # proj = self.proj(qkv)
        # pe = self.pe(proj)
        # b_attn = qkv + proj + pe + b  # Residual connections

        b_ffn = self.ffn(b_attn) + b_attn
        return self.cv2(torch.cat([a, b_ffn], 1))

class DFL(nn.Module):
    """Distribution Focal Loss module."""
    def __init__(self, ch=16):
        super().__init__()
        self.ch = ch
        self.conv = nn.Conv2d(ch, 1, 1, bias=False)
        nn.init.constant_(self.conv.weight, 0.0)
        
    def forward(self, x):
        b, c, a = x.shape
        x = x.view(b, 4, self.ch, a).transpose(2, 1)
        return self.conv(x.softmax(1)).view(b, 4, a)

class v10Detect(nn.Module):
    """YOLOv10 detection head."""
    def __init__(self, nc=80, ch=(192, 384, 576), ch2=(64,192)):
        super().__init__()
        self.nc = nc  # number of classes
        self.nl = len(ch)  # number of detection layers
        self.reg_max = 16  # DFL channels

        # Classification and regression branches
        self.cv2 = nn.ModuleList(nn.Sequential(
            Conv(x, ch2[0], 3),
            Conv(ch2[0], ch2[0], 3),
            nn.Conv2d(ch2[0], 64, 1)
        ) for x in ch)
        
        self.cv3 = nn.ModuleList(nn.Sequential(
            nn.Sequential(
                Conv(x, x, 3, groups=x),
                Conv(x, ch2[1], 1)
            ),
            nn.Sequential(
                Conv(ch2[1], ch2[1], 3, groups=ch2[1]),
                Conv(ch2[1], ch2[1], 1)
            ),
            nn.Conv2d(ch2[1], self.nc, 1)
        ) for x in ch)

        self.dfl = DFL(self.reg_max)
        
        # One-to-one branches
        self.one2one_cv2 = nn.ModuleList(nn.Sequential(
            Conv(x, ch2[0], 3),
            Conv(ch2[0], ch2[0], 3),
            nn.Conv2d(ch2[0], 64, 1)
        ) for x in ch)
        
        self.one2one_cv3 = nn.ModuleList(nn.Sequential(
            nn.Sequential(
                Conv(x, x, 3, groups=x),
                Conv(x, ch2[1], 1)
            ),
            nn.Sequential(
                Conv(ch2[1], ch2[1], 3, groups=ch2[1]),
                Conv(ch2[1], ch2[1], 1)
            ),
            nn.Conv2d(ch2[1], self.nc, 1)
        ) for x in ch)
        
        
        
    def forward(self, x):
        """Process features from all levels."""
        outputs = []
        for i in range(self.nl):
            cls_out = self.cv3[i](x[i])
            reg_out = self.cv2[i](x[i])
            outputs.append(torch.cat([reg_out, cls_out], dim=1))
        return outputs

class Concat(nn.Module):
    """Simple concatenation layer."""
    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim
        
    def forward(self, x):
        # Ensure x is a list/tuple of tensors
        if isinstance(x, torch.Tensor):
            return x
        return torch.cat(list(x), dim=self.dim)  # Explicit conversion to list

class DetectionModel(nn.Module):
    """Complete YOLOv10 model architecture."""
    def __init__(self, variant='m', num_classes=80):
        super().__init__()

        config = dimOfVariant(variant=variant)
        d, bb_ch, maxCh, c2fCh, det_conv_ch = config['depth'], config['backbone_ch'], config['maxCh'], config['c2fCh'], config['detect_conv_channel']
        def n_layers(n): return max(round(n * d), 1)
        # print(d, n_layers(3), n_layers(6))
        
        # --- Backbone ---
        self.backbone = nn.ModuleList([
            Conv(3, bb_ch[0], 3, 2),                        # 0
            Conv(bb_ch[0], bb_ch[1], 3, 2),                 # 1
            C2f(bb_ch[1], bb_ch[1], n=n_layers(3)),         # 2  # P3 output
            Conv(bb_ch[1], bb_ch[2], 3, 2),                 # 3
            C2f(bb_ch[2], bb_ch[2], n=n_layers(6)),         # 4  # P4 output
            SCDown(bb_ch[2], bb_ch[3]),                     # 5
            C2fCIB(bb_ch[3], bb_ch[3], n=n_layers(6)) if variant=='x' else C2f(bb_ch[3], bb_ch[3], n=n_layers(6)), # 6
            SCDown(bb_ch[3], bb_ch[4]),                     # 7
            C2f(bb_ch[4], bb_ch[4], n=n_layers(3)) if variant=='n' else C2fCIB(bb_ch[4], bb_ch[4], n=n_layers(3), isRepVGGDW=variant=='s'), # 8 # P5 input to neck
            SPPF(bb_ch[4], bb_ch[4]),                       # 9 # P5 input to neck after SPPF
            PSA(bb_ch[4]),                                  # 10 # P5 input to neck after PSA (final P5 feature)
        ])
        # Note: The exact indices for P3, P4, P5 outputs from the backbone might vary slightly
        # based on the official YOLOv10 implementation details for each variant.
        # I'm assuming P3 is after layer 2, P4 after layer 4, and P5 after layer 10.

        # --- Neck ---
        self.neck = nn.ModuleList([
            nn.Upsample(scale_factor=2, mode='nearest'),    # Upsample P5 -> P4 scale
            C2fCIB(bb_ch[4] + bb_ch[3], bb_ch[3], n=n_layers(3)) if (variant=='l' or variant=='x') else C2f(bb_ch[4] + bb_ch[3], bb_ch[3], n=n_layers(3)), # Fuse P5_up + P4 -> P4 output
            nn.Upsample(scale_factor=2, mode='nearest'),    # Upsample P4 -> P3 scale
            C2f(bb_ch[3] + bb_ch[2], bb_ch[2], n=n_layers(3)), # Fuse P4_up + P3 -> P3 output

            # Downsampling path
            Conv(bb_ch[2], bb_ch[2], 3, 2),                 # P3 -> P4 scale
            C2f(bb_ch[2] + bb_ch[3], bb_ch[3], n=n_layers(3)) if (variant=='n' or variant=='s') else C2fCIB(bb_ch[2] + bb_ch[3], bb_ch[3], n=n_layers(3)), # Fuse P3_down + P4 -> P4 output (for P4->P5 downsampling)
            SCDown(bb_ch[3], bb_ch[3]),                     # P4 -> P5 scale
            C2fCIB(bb_ch[3] + bb_ch[4], bb_ch[4], n=n_layers(3), isRepVGGDW=(variant=='n' or variant=='s')), # Fuse P4_down + P5 -> P5 output
        ])

        # --- Detection Head ---
        # The detection head takes a list of neck outputs (P3, P4, P5)
        self.detection_head = v10Detect(nc=num_classes, ch=bb_ch[2:], ch2=det_conv_ch)

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # print(f"Input shape: {x.shape}") # Optional print for debugging

        # --- Backbone ---
        # Process through backbone layers and store outputs at specific stages
        x0 = self.backbone[0](x)
        x1 = self.backbone[1](x0)
        x2 = self.backbone[2](x1)
        p3 = self.backbone[3](x2) # P3 output (1/8)

        x4 = self.backbone[4](p3)
        x5 = self.backbone[5](x4)
        p4 = self.backbone[6](x5) # P4 output (1/16)

        x7 = self.backbone[7](p4)
        x8 = self.backbone[8](x7)
        x9 = self.backbone[9](x8)
        p5 = self.backbone[10](x9) # P5 output (1/32)

        # backbone_features now contains [P3, P4, P5] from the backbone
        # print(f"Backbone P3 shape: {p3.shape}")
        # print(f"Backbone P4 shape: {p4.shape}")
        # print(f"Backbone P5 shape: {p5.shape}")

        # --- Neck ---
        # Top-down path (P5 -> P4 -> P3)
        p5_up = self.neck[0](p5) # Upsample P5 (1/32 -> 1/16)
        # print(f"p5_up shape: {p5_up.shape}") # Should match p4 shape
        p4_fused = torch.cat([p5_up, p4], dim=1) # Concatenate with P4 from backbone (1/16)
        p4_out = self.neck[1](p4_fused) # C2f/C2fCIB on fused P4 (1/16)
        # print(f"p4_out shape: {p4_out.shape}")

        p4_up = self.neck[2](p4_out) # Upsample P4_out (1/16 -> 1/8)
        # print(f"p4_up shape: {p4_up.shape}") # Should match p3 shape
        p3_fused = torch.cat([p4_up, p3], dim=1) # Concatenate with P3 from backbone (1/8)
        p3_out = self.neck[3](p3_fused) # C2f on fused P3 (1/8)
        # print(f"p3_out shape: {p3_out.shape}")

        # Bottom-up path (P3 -> P4 -> P5)
        p3_down = self.neck[4](p3_out) # Downsample P3_out (1/8 -> 1/16)
        # print(f"p3_down shape: {p3_down.shape}") # Should match p4_out shape
        p4_fused_down = torch.cat([p3_down, p4_out], dim=1) # Concatenate with P4_out from top-down (1/16)
        p4_out_down = self.neck[5](p4_fused_down) # C2f/C2fCIB on fused P4 (for downsampling) (1/16)
        # print(f"p4_out_down shape: {p4_out_down.shape}")

        p4_down = self.neck[6](p4_out_down) # Downsample P4_out_down (1/16 -> 1/32)
        # print(f"p4_down shape: {p4_down.shape}") # Should match p5 shape
        p5_fused_down = torch.cat([p4_down, p5], dim=1) # Concatenate with original P5 from backbone (1/32)
        p5_out = self.neck[7](p5_fused_down) # C2fCIB on fused P5 (1/32)
        # print(f"p5_out shape: {p5_out.shape}")


        # The final neck outputs are typically p3_out, p4_out_down, p5_out
        neck_outputs = [p3_out, p4_out_down, p5_out]


        # --- Detection Head ---
        # The detection head receives the list of neck outputs
        head_outputs = self.detection_head(neck_outputs)

        # --- Return ---
        # The forward method of the DetectionModel should return the output of the detection head,
        # which is a list of tensors, one for each detection scale.
        return head_outputs


if __name__ == "__main__":
    model = DetectionModel(variant='m', num_classes=80)
    print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Test forward pass
    x = torch.randn(1, 3, 640, 640)
    try:
        outputs = model(x)
        print("\nModel outputs:")
        for i, out in enumerate(outputs):
            print(f"Output {i} shape: {out.shape}")
    except Exception as e:
        print(f"\nError during forward pass: {e}")
        # traceback.print_exc()