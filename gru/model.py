import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    from torchvision import models as tv_models
except Exception:
    tv_models = None


class GRUNet(nn.Module):
    """
    GRU Network: Maps 400-dimensional input vector to
    - affine_params: 3x4 affine matrix (row-wise flattened) with shape (12,)
    - tone_params: Non-linear tone mapping parameters with shape (4,)

    Expects input shape as a float tensor of (batch_size, 400).
    """


    def __init__(self, hidden_dim: int = 512, dropout: float = 0.1):
        super().__init__()
        self.input_dim = 400
        self.hidden_dim = hidden_dim

        self.backbone = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),

            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),

            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(inplace=True),
        )

        self.head_affine = nn.Linear(hidden_dim // 2, 12)
        self.head_tone = nn.Linear(hidden_dim // 2, 4)

        # Initialize output heads for more stable training
        nn.init.zeros_(self.head_affine.weight)
        nn.init.zeros_(self.head_affine.bias)
        nn.init.zeros_(self.head_tone.weight)
        nn.init.zeros_(self.head_tone.bias)

    def forward(self, x: torch.Tensor):
        if x.dim() != 2 or x.size(1) != self.input_dim:
            raise ValueError(f"Expected input shape (batch, 400), got {tuple(x.shape)}")
        features = self.backbone(x)
        affine = self.head_affine(features)
        tone = self.head_tone(features)
        return affine, tone


def build_input_vector(rgb_16x8: torch.Tensor, gray_16x1: torch.Tensor) -> torch.Tensor:
    """
    Combine 3x16x8 RGB tensor with 16x1 grayscale tensor into a 400-dimensional vector.

    - rgb_16x8: Shape (batch, 3, 16, 8)
    - gray_16x1: Shape (batch, 16, 1)

    Returns: Shape (batch, 400)
    """

    if rgb_16x8.dim() != 4 or rgb_16x8.size(1) != 3 or rgb_16x8.size(2) != 16 or rgb_16x8.size(3) != 8:
        raise ValueError(f"rgb_16x8 must be (batch,3,16,8), got {tuple(rgb_16x8.shape)}")
    if gray_16x1.dim() != 3 or gray_16x1.size(1) != 16 or gray_16x1.size(2) != 1:
        raise ValueError(f"gray_16x1 must be (batch,16,1), got {tuple(gray_16x1.shape)}")

    batch = rgb_16x8.size(0)
    rgb_flat = rgb_16x8.reshape(batch, -1)  # 3*16*8 = 384
    gray_flat = gray_16x1.reshape(batch, -1)  # 16
    combined = torch.cat([rgb_flat, gray_flat], dim=1)  # 384 + 16 = 400
    return combined


def build_4ch_image(rgb_16x8: torch.Tensor, gray_16x1: torch.Tensor) -> torch.Tensor:
    """
    Construct 4-channel image from (B,3,16,8) and (B,16,1): repeat grayscale column in width direction.

    Returns: Shape (batch, 4, 16, 8)
    """

    if rgb_16x8.dim() != 4 or rgb_16x8.size(1) != 3 or rgb_16x8.size(2) != 16 or rgb_16x8.size(3) != 8:
        raise ValueError(f"rgb_16x8 must be (batch,3,16,8), got {tuple(rgb_16x8.shape)}")
    if gray_16x1.dim() != 3 or gray_16x1.size(1) != 16 or gray_16x1.size(2) != 1:
        raise ValueError(f"gray_16x1 must be (batch,16,1), got {tuple(gray_16x1.shape)}")
    batch = rgb_16x8.size(0)
    gray_16x8 = gray_16x1.repeat(1, 1, 8)  # (B,16,8)
    gray_4d = gray_16x8.unsqueeze(1)  # (B,1,16,8)
    x4 = torch.cat([rgb_16x8, gray_4d], dim=1)  # (B,4,16,8)
    return x4


class MobileNetGRU(nn.Module):
    """
    MobileNet-based backbone network, input is 4-channel (RGB + grayscale) 16x8 image,
    output is affine parameters (12,) and tone parameters (4,).
    """


    def __init__(self, variant: str = "v2", pretrained: bool = False, dropout: float = 0.1):
        super().__init__()
        if tv_models is None:
            raise ImportError("torchvision is required for MobileNetGRU. Please install torchvision.")

        if variant == "v2":
            backbone = tv_models.mobilenet_v2(weights=None if not pretrained else tv_models.MobileNet_V2_Weights.DEFAULT)
            in_feat = backbone.last_channel
            self.features = backbone.features
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
        elif variant == "v3":
            backbone = tv_models.mobilenet_v3_small(weights=None if not pretrained else tv_models.MobileNet_V3_Small_Weights.DEFAULT)
            in_feat = backbone.classifier[0].in_features if hasattr(backbone.classifier[0], 'in_features') else 576
            self.features = backbone.features
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
        else:
            raise ValueError("variant must be 'v2' or 'v3'")

        # Modify first convolution to accept 4-channel input
        first_conv = None
        for m in self.features.modules():
            if isinstance(m, nn.Conv2d) and m.in_channels == 3:
                first_conv = m
                break
        if first_conv is None:
            raise RuntimeError("Could not find first Conv2d with 3 input channels in MobileNet features")

        new_first = nn.Conv2d(4, first_conv.out_channels, kernel_size=first_conv.kernel_size,
                              stride=first_conv.stride, padding=first_conv.padding, bias=first_conv.bias is not None, groups=first_conv.groups)
        with torch.no_grad():
            if first_conv.weight.shape[1] == 3:
                new_first.weight[:, :3] = first_conv.weight
                # Initialize the 4th channel weights as the mean of RGB weights
                new_first.weight[:, 3:4] = first_conv.weight.mean(dim=1, keepdim=True)
            else:
                nn.init.kaiming_normal_(new_first.weight, nonlinearity='relu')
            if new_first.bias is not None and first_conv.bias is not None:
                new_first.bias.copy_(first_conv.bias)

        # Replace the convolution in features
        replaced = False
        for name, module in self.features._modules.items():
            if isinstance(module, nn.Sequential):
                for sub_name, sub_module in module._modules.items():
                    if sub_module is first_conv:
                        module._modules[sub_name] = new_first
                        replaced = True
                        break
                if replaced:
                    break
            elif module is first_conv:
                self.features._modules[name] = new_first
                replaced = True
                break
        if not replaced:
            # Fallback: assume the first layer is at index 0
            self.features[0][0] = new_first  # type: ignore[index]

        hidden = 512
        self.head = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_feat, hidden),
            nn.ReLU(inplace=True),
        )
        self.affine = nn.Linear(hidden, 12)
        self.tone = nn.Linear(hidden, 4)

        nn.init.zeros_(self.affine.weight)
        nn.init.zeros_(self.affine.bias)
        nn.init.zeros_(self.tone.weight)
        nn.init.zeros_(self.tone.bias)

    def forward(self, x4: torch.Tensor):
        if x4.dim() != 4 or x4.size(1) != 4 or x4.size(2) != 16 or x4.size(3) != 8:
            raise ValueError(f"Expected (batch,4,16,8) input, got {tuple(x4.shape)}")
        feats = self.features(x4)
        feats = self.pool(feats)
        feats = feats.flatten(1)
        feats = self.head(feats)
        affine = self.affine(feats)
        tone = self.tone(feats)
        return affine, tone


