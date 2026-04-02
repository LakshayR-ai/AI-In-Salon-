"""
Hair segmentation using BiSeNet face parser.

Loads the official zllrunning/face-parsing.PyTorch checkpoint
(79999_iter.pth / face_parsing.pth).

Hair label = 17 in the 19-class CelebAMask-HQ scheme.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
import numpy as np


# ── Exact architecture from zllrunning/face-parsing.PyTorch ──────────────────
# This matches the key names in the downloaded checkpoint exactly.

class ConvBNReLU(nn.Module):
    def __init__(self, in_chan, out_chan, ks=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_chan, out_chan, kernel_size=ks,
                              stride=stride, padding=padding, bias=False)
        self.bn   = nn.BatchNorm2d(out_chan)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)), inplace=True)


class BiSeNetOutput(nn.Module):
    def __init__(self, in_chan, mid_chan, n_classes):
        super().__init__()
        self.conv = ConvBNReLU(in_chan, mid_chan, ks=3, stride=1, padding=1)
        self.conv_out = nn.Conv2d(mid_chan, n_classes, kernel_size=1, bias=False)

    def forward(self, x):
        return self.conv_out(self.conv(x))


class AttentionRefinementModule(nn.Module):
    def __init__(self, in_chan, out_chan):
        super().__init__()
        self.conv      = ConvBNReLU(in_chan, out_chan, ks=3, stride=1, padding=1)
        self.conv_atten = nn.Conv2d(out_chan, out_chan, kernel_size=1, bias=False)
        self.bn_atten   = nn.BatchNorm2d(out_chan)
        self.sigmoid    = nn.Sigmoid()

    def forward(self, x):
        feat  = self.conv(x)
        atten = F.avg_pool2d(feat, feat.size()[2:])
        atten = self.sigmoid(self.bn_atten(self.conv_atten(atten)))
        return torch.mul(feat, atten)


class ContextPath(nn.Module):
    def __init__(self):
        super().__init__()
        from torchvision.models import resnet18
        # checkpoint uses cp.resnet.* keys
        self.resnet  = resnet18(weights=None)
        self.arm16   = AttentionRefinementModule(256, 128)
        self.arm32   = AttentionRefinementModule(512, 128)
        self.conv_head32 = ConvBNReLU(128, 128, ks=3, stride=1, padding=1)
        self.conv_head16 = ConvBNReLU(128, 128, ks=3, stride=1, padding=1)
        self.conv_avg    = ConvBNReLU(512, 128, ks=1, stride=1, padding=0)

    def forward(self, x):
        feat = self.resnet.relu(self.resnet.bn1(self.resnet.conv1(x)))
        feat = self.resnet.maxpool(feat)
        feat = self.resnet.layer1(feat)
        feat = self.resnet.layer2(feat)
        feat16 = self.resnet.layer3(feat)
        feat32 = self.resnet.layer4(feat16)
        avg = self.conv_avg(F.avg_pool2d(feat32, feat32.size()[2:]))
        feat32_arm = self.arm32(feat32)
        feat32_sum = feat32_arm + avg
        feat32_up  = self.conv_head32(
            F.interpolate(feat32_sum, feat16.size()[2:], mode='nearest'))
        feat16_arm = self.arm16(feat16)
        feat16_sum = feat16_arm + feat32_up
        feat16_up  = self.conv_head16(
            F.interpolate(feat16_sum, feat.size()[2:], mode='nearest'))
        return feat16_up, feat32_up


class FeatureFusionModule(nn.Module):
    def __init__(self, in_chan, out_chan):
        super().__init__()
        self.convblk = ConvBNReLU(in_chan, out_chan, ks=1, stride=1, padding=0)
        self.conv1   = nn.Conv2d(out_chan, out_chan // 4, kernel_size=1, bias=False)
        self.conv2   = nn.Conv2d(out_chan // 4, out_chan, kernel_size=1, bias=False)
        self.relu    = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, fsp, fcp):
        fcat  = torch.cat([fsp, fcp], dim=1)
        feat  = self.convblk(fcat)
        atten = F.avg_pool2d(feat, feat.size()[2:])
        atten = self.relu(self.conv1(atten))
        atten = self.sigmoid(self.conv2(atten))
        return feat + feat * atten


class SpatialPath(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = ConvBNReLU(3,  64, ks=7, stride=2, padding=3)
        self.conv2 = ConvBNReLU(64, 64, ks=3, stride=2, padding=1)
        self.conv3 = ConvBNReLU(64, 64, ks=3, stride=2, padding=1)
        self.conv_out = ConvBNReLU(64, 128, ks=1, stride=1, padding=0)

    def forward(self, x):
        return self.conv_out(self.conv3(self.conv2(self.conv1(x))))


class BiSeNet(nn.Module):
    HAIR_LABEL = 17

    def __init__(self, n_classes=19):
        super().__init__()
        self.cp  = ContextPath()
        self.sp  = SpatialPath()
        self.ffm = FeatureFusionModule(256, 256)
        self.conv_out   = BiSeNetOutput(256, 256, n_classes)
        self.conv_out16 = BiSeNetOutput(128, 64,  n_classes)
        self.conv_out32 = BiSeNetOutput(128, 64,  n_classes)

    def forward(self, x):
        H, W = x.size()[2:]
        feat_sp        = self.sp(x)
        feat16, feat32 = self.cp(x)
        feat_fuse      = self.ffm(feat_sp, feat16)
        out      = self.conv_out(feat_fuse)
        out16    = self.conv_out16(feat16)
        out32    = self.conv_out32(feat32)
        out      = F.interpolate(out,   (H, W), mode='bilinear', align_corners=False)
        out16    = F.interpolate(out16, (H, W), mode='bilinear', align_corners=False)
        out32    = F.interpolate(out32, (H, W), mode='bilinear', align_corners=False)
        return out, out16, out32


# ── Public API ────────────────────────────────────────────────────────────────

class HairSegmenter:
    def __init__(self, checkpoint: str, device: str = "cpu"):
        self.device = device
        self.model  = BiSeNet(n_classes=19).to(device)
        state = torch.load(checkpoint, map_location=device)
        self.model.load_state_dict(state, strict=False)
        self.model.eval()

        self.transform = T.Compose([
            T.Resize((512, 512)),
            T.ToTensor(),
            T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

    @torch.no_grad()
    def segment(self, img: Image.Image) -> np.ndarray:
        orig_size = img.size
        x = self.transform(img.convert("RGB")).unsqueeze(0).to(self.device)
        out, _, _ = self.model(x)
        pred = out.argmax(dim=1)[0].cpu().numpy()
        mask = (pred == BiSeNet.HAIR_LABEL).astype(np.uint8) * 255
        return np.array(Image.fromarray(mask).resize(orig_size, Image.NEAREST))

    @torch.no_grad()
    def segment_tensor(self, img_tensor: torch.Tensor) -> torch.Tensor:
        x = img_tensor * 0.5 + 0.5
        mean = torch.tensor([0.485,0.456,0.406], device=x.device).view(1,3,1,1)
        std  = torch.tensor([0.229,0.224,0.225], device=x.device).view(1,3,1,1)
        x = (x - mean) / std
        out, _, _ = self.model(x)
        return (out.argmax(dim=1, keepdim=True) == BiSeNet.HAIR_LABEL).float()
