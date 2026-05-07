import torch
import torch.nn as nn
import torch.nn.functional as F

# ============================================================
# ARCHITECTURE DEFINITION (Must match Training Script)
# ============================================================
class ResidualBlock(nn.Module):
    """Residual block with 128 channels and reflection padding."""
    def __init__(self, c):
        super().__init__()
        self.conv = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(c, c, 3, 1, 0), nn.InstanceNorm2d(c), nn.ReLU(True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(c, c, 3, 1, 0), nn.InstanceNorm2d(c)
        )
    def forward(self, x): return x + self.conv(x)

class StyleTransferNet(nn.Module):
    """Generator network for real-time inference (7 input channels)."""
    def __init__(self, n_res=6):
        super().__init__()
        # Encoder
        self.enc = nn.Sequential(
            nn.ReflectionPad2d(4),
            nn.Conv2d(7, 32, 9, 1, 0), nn.InstanceNorm2d(32), nn.ReLU(True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(32, 64, 3, 2, 0), nn.InstanceNorm2d(64), nn.ReLU(True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(64, 128, 3, 2, 0), nn.InstanceNorm2d(128), nn.ReLU(True)
        )
        # Residual Bottleneck
        self.res = nn.Sequential(*[ResidualBlock(128) for _ in range(n_res)])
        # Decoder
        self.dec = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(128, 64, 3, 1, 0), nn.InstanceNorm2d(64), nn.ReLU(True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(64, 32, 3, 1, 0), nn.InstanceNorm2d(32), nn.ReLU(True),
            nn.ReflectionPad2d(4),
            nn.Conv2d(32, 3, 9, 1, 0)
        )
    def forward(self, x): return torch.sigmoid(self.dec(self.res(self.enc(x))))

def export():
    """Converts the trained .pth checkpoint to the ONNX format."""
    # 1. Initialize model and load weights
    model = StyleTransferNet(n_res=6)
    checkpoint = "neural_ink_mi300x.pth"
    
    print(f"Loading weights from {checkpoint}...")
    try:
        ckpt = torch.load(checkpoint, map_location='cpu')
        # Handle different checkpoint formats
        weights = ckpt['model'] if 'model' in ckpt else ckpt
        
        # Remove '_orig_mod.' prefix if torch.compile was used during training
        new_weights = {}
        for k, v in weights.items():
            name = k.replace("_orig_mod.", "")
            new_weights[name] = v
            
        model.load_state_dict(new_weights)
        print("[OK] Weights loaded successfully.")
    except Exception as e:
        print(f"[ERROR] Could not load checkpoint: {e}")
        return

    model.eval()

    # 2. Export to ONNX with dynamic axes for variable resolution
    dummy_input = torch.randn(1, 7, 512, 512)
    onnx_path = "neural_ink_mi300x.onnx"
    
    print(f"Exporting to {onnx_path}...")
    torch.onnx.export(
        model, dummy_input, onnx_path,
        opset_version=15,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch', 2: 'height', 3: 'width'},
            'output': {0: 'batch', 2: 'height', 3: 'width'}
        }
    )
    print("¡Export complete!")

if __name__ == "__main__":
    export()
