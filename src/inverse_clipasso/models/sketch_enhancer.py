"""
Sketch Enhancement Model using Pix2Pix-style architecture.

This model learns to transform "rough" sketches into "refined" sketches.
It's trained on pairs of (input_sketch, target_sketch).

Architecture:
- U-Net generator (encoder-decoder with skip connections)
- PatchGAN discriminator (optional, for adversarial training)
- Can be trained with L1 loss alone or L1 + adversarial
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List


class ConvBlock(nn.Module):
    """Convolutional block with optional normalization and activation."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 4,
        stride: int = 2,
        padding: int = 1,
        norm: bool = True,
        activation: str = "leaky_relu",
    ):
        super().__init__()
        
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=not norm)
        ]
        
        if norm:
            layers.append(nn.BatchNorm2d(out_channels))
        
        if activation == "leaky_relu":
            layers.append(nn.LeakyReLU(0.2, inplace=True))
        elif activation == "relu":
            layers.append(nn.ReLU(inplace=True))
        elif activation == "tanh":
            layers.append(nn.Tanh())
        elif activation == "sigmoid":
            layers.append(nn.Sigmoid())
        
        self.block = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.block(x)


class DeconvBlock(nn.Module):
    """Transposed convolution block for upsampling."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 4,
        stride: int = 2,
        padding: int = 1,
        norm: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        
        layers = [
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=not norm)
        ]
        
        if norm:
            layers.append(nn.BatchNorm2d(out_channels))
        
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        
        layers.append(nn.ReLU(inplace=True))
        
        self.block = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.block(x)


class UNetGenerator(nn.Module):
    """
    U-Net style generator for sketch-to-sketch translation.
    
    Features:
    - Encoder-decoder architecture with skip connections
    - Works with 256x256 images (input is resized internally)
    - Preserves spatial structure through skip connections
    """
    
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        base_filters: int = 64,
    ):
        super().__init__()
        
        # For 256x256: 256->128->64->32->16->8->4->2->1
        # Encoder
        self.enc1 = ConvBlock(in_channels, base_filters, norm=False)     # 128
        self.enc2 = ConvBlock(base_filters, base_filters * 2)            # 64
        self.enc3 = ConvBlock(base_filters * 2, base_filters * 4)        # 32
        self.enc4 = ConvBlock(base_filters * 4, base_filters * 8)        # 16
        self.enc5 = ConvBlock(base_filters * 8, base_filters * 8)        # 8
        self.enc6 = ConvBlock(base_filters * 8, base_filters * 8)        # 4
        self.enc7 = ConvBlock(base_filters * 8, base_filters * 8)        # 2
        self.enc8 = ConvBlock(base_filters * 8, base_filters * 8, norm=False)  # 1
        
        # Decoder (with skip connections)
        self.dec8 = DeconvBlock(base_filters * 8, base_filters * 8, dropout=0.5)   # 2
        self.dec7 = DeconvBlock(base_filters * 16, base_filters * 8, dropout=0.5)  # 4
        self.dec6 = DeconvBlock(base_filters * 16, base_filters * 8, dropout=0.5)  # 8
        self.dec5 = DeconvBlock(base_filters * 16, base_filters * 8)               # 16
        self.dec4 = DeconvBlock(base_filters * 16, base_filters * 4)               # 32
        self.dec3 = DeconvBlock(base_filters * 8, base_filters * 2)                # 64
        self.dec2 = DeconvBlock(base_filters * 4, base_filters)                    # 128
        
        # Final layer
        self.final = nn.Sequential(
            nn.ConvTranspose2d(base_filters * 2, out_channels, 4, 2, 1),           # 256
            nn.Tanh(),  # Output in [-1, 1]
        )
    
    def forward(self, x):
        # Resize input to 256x256 if needed
        original_size = x.shape[-2:]
        if original_size != (256, 256):
            x = F.interpolate(x, size=(256, 256), mode='bilinear', align_corners=False)
        
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        e5 = self.enc5(e4)
        e6 = self.enc6(e5)
        e7 = self.enc7(e6)
        e8 = self.enc8(e7)
        
        # Decoder with skip connections
        d8 = self.dec8(e8)
        d7 = self.dec7(torch.cat([d8, e7], dim=1))
        d6 = self.dec6(torch.cat([d7, e6], dim=1))
        d5 = self.dec5(torch.cat([d6, e5], dim=1))
        d4 = self.dec4(torch.cat([d5, e4], dim=1))
        d3 = self.dec3(torch.cat([d4, e3], dim=1))
        d2 = self.dec2(torch.cat([d3, e2], dim=1))
        
        out = self.final(torch.cat([d2, e1], dim=1))
        
        # Resize output back to original size if needed
        if original_size != (256, 256):
            out = F.interpolate(out, size=original_size, mode='bilinear', align_corners=False)
        
        return out


class PatchDiscriminator(nn.Module):
    """
    PatchGAN discriminator for adversarial training.
    
    Classifies 70x70 patches as real or fake.
    """
    
    def __init__(
        self,
        in_channels: int = 2,  # Input + target concatenated
        base_filters: int = 64,
    ):
        super().__init__()
        
        self.model = nn.Sequential(
            ConvBlock(in_channels, base_filters, norm=False),          # 112
            ConvBlock(base_filters, base_filters * 2),                 # 56
            ConvBlock(base_filters * 2, base_filters * 4),             # 28
            ConvBlock(base_filters * 4, base_filters * 8, stride=1),   # 28
            nn.Conv2d(base_filters * 8, 1, 4, 1, 1),                   # 27
        )
    
    def forward(self, x, y):
        """
        Args:
            x: Input sketch
            y: Target/generated sketch
        
        Returns:
            Patch-wise predictions
        """
        combined = torch.cat([x, y], dim=1)
        return self.model(combined)


class SketchEnhancer(nn.Module):
    """
    Complete sketch enhancement model.
    
    Can be trained with:
    - L1 loss only (simpler, faster)
    - L1 + Adversarial loss (sharper results)
    - L1 + Perceptual loss using CLIP (semantically aware)
    """
    
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        use_discriminator: bool = False,
    ):
        super().__init__()
        
        self.generator = UNetGenerator(in_channels, out_channels)
        
        self.discriminator = None
        if use_discriminator:
            self.discriminator = PatchDiscriminator(in_channels + out_channels)
    
    def forward(self, x):
        """Generate enhanced sketch."""
        return self.generator(x)
    
    def compute_generator_loss(
        self,
        input_sketch: torch.Tensor,
        target_sketch: torch.Tensor,
        lambda_l1: float = 100.0,
        lambda_adv: float = 1.0,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute generator loss.
        
        Args:
            input_sketch: Input rough sketch [B, 1, H, W]
            target_sketch: Target refined sketch [B, 1, H, W]
            lambda_l1: Weight for L1 loss
            lambda_adv: Weight for adversarial loss
        
        Returns:
            Total loss and dict of individual losses
        """
        # Generate
        generated = self.generator(input_sketch)
        
        # L1 loss
        l1_loss = F.l1_loss(generated, target_sketch)
        
        losses = {"l1": l1_loss.item()}
        total_loss = lambda_l1 * l1_loss
        
        # Adversarial loss (if using discriminator)
        if self.discriminator is not None:
            pred_fake = self.discriminator(input_sketch, generated)
            adv_loss = F.binary_cross_entropy_with_logits(
                pred_fake, torch.ones_like(pred_fake)
            )
            losses["adv"] = adv_loss.item()
            total_loss = total_loss + lambda_adv * adv_loss
        
        losses["total"] = total_loss.item()
        
        return total_loss, losses
    
    def compute_discriminator_loss(
        self,
        input_sketch: torch.Tensor,
        target_sketch: torch.Tensor,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute discriminator loss.
        
        Args:
            input_sketch: Input rough sketch [B, 1, H, W]
            target_sketch: Target refined sketch [B, 1, H, W]
        
        Returns:
            Discriminator loss and dict of metrics
        """
        if self.discriminator is None:
            return torch.tensor(0.0), {}
        
        # Generate fake
        with torch.no_grad():
            generated = self.generator(input_sketch)
        
        # Real loss
        pred_real = self.discriminator(input_sketch, target_sketch)
        loss_real = F.binary_cross_entropy_with_logits(
            pred_real, torch.ones_like(pred_real)
        )
        
        # Fake loss
        pred_fake = self.discriminator(input_sketch, generated)
        loss_fake = F.binary_cross_entropy_with_logits(
            pred_fake, torch.zeros_like(pred_fake)
        )
        
        total_loss = (loss_real + loss_fake) * 0.5
        
        return total_loss, {
            "d_real": loss_real.item(),
            "d_fake": loss_fake.item(),
            "d_total": total_loss.item(),
        }


def create_sketch_enhancer(
    use_discriminator: bool = False,
    pretrained: Optional[str] = None,
) -> SketchEnhancer:
    """
    Factory function to create a SketchEnhancer model.
    
    Args:
        use_discriminator: If True, include discriminator for adversarial training
        pretrained: Path to pretrained weights (optional)
    
    Returns:
        SketchEnhancer model
    """
    model = SketchEnhancer(
        in_channels=1,
        out_channels=1,
        use_discriminator=use_discriminator,
    )
    
    if pretrained is not None:
        state_dict = torch.load(pretrained, map_location="cpu")
        model.load_state_dict(state_dict)
        print(f"Loaded pretrained weights from {pretrained}")
    
    return model

