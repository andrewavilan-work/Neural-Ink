"""
Neural-Ink V5 — AMD MI300X Hackathon Training
==============================================
Target Hardware : AMD Instinct MI300X (192 GB HBM3 Unified Memory)
Software Stack  : ROCm 6.x + PyTorch ROCm build
Precision       : FP32 (Full parity with local training)
Batch Size      : 1 (Same as local)
Resolution      : 512x512 (Same as local)
ResBlocks       : 6 (Same as local)
"""

import os, sys, time
# Enable OpenEXR support for OpenCV before any other imports
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms, utils
import numpy as np
import cv2
from PIL import Image

# ============================================================
# AMD MI300X ENVIRONMENT SETUP
# ============================================================
def setup_amd_mi300x():
    """Verifies ROCm/CUDA availability for AMD Instinct hardware."""
    if not torch.cuda.is_available():
        print("[ERROR] ROCm/CUDA not detected. Ensure ROCm drivers are installed.")
        sys.exit(1)
    return torch.device("cuda")

device = setup_amd_mi300x()

# Hyperparameters optimized for MI300X High-Performance Training
BATCH_SIZE    = 1       
IMG_SIZE      = 512     
NUM_RESBLOCKS = 6      
EPOCHS        = 200
LR            = 1e-4    
CHECKPOINT_PATH    = "neural_ink_mi300x.pth"
STYLE_DIR     = "styles"

# VGG-16 Normalization constants
vgg_mean = torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1).to(device)
vgg_std  = torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1).to(device)
def vgg_norm(x): return (x - vgg_mean) / vgg_std

# ============================================================
# ARCHITECTURE (PARITY WITH UNITY SENTIS INFERENCE)
# ============================================================
class ResidualBlock(nn.Module):
    """Standard Residual Block with Reflection Padding to eliminate border artifacts."""
    def __init__(self, c):
        super().__init__()
        self.conv = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(c, c, 3, 1, 0), nn.InstanceNorm2d(c), nn.ReLU(True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(c, c, 3, 1, 0), nn.InstanceNorm2d(c)
        )
    def forward(self, x): return x + self.conv(x)

class StyleTransferNetMI300X(nn.Module):
    """Recurrent Style Transfer Generator (7 input channels: RGB + Depth + Warped Prev)."""
    def __init__(self, n_res=6):
        super().__init__()
        # Encoder: Downsampling to 128 feature maps
        self.enc = nn.Sequential(
            nn.ReflectionPad2d(4),
            nn.Conv2d(7, 32, 9, 1, 0), nn.InstanceNorm2d(32), nn.ReLU(True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(32, 64, 3, 2, 0), nn.InstanceNorm2d(64), nn.ReLU(True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(64, 128, 3, 2, 0), nn.InstanceNorm2d(128), nn.ReLU(True)
        )
        # 6 Residual Blocks for deep style synthesis
        self.res = nn.Sequential(*[ResidualBlock(128) for _ in range(n_res)])
        # Decoder: Upsampling back to 512x512 RGB
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

class VGGExtractorDeep(nn.Module):
    """VGG-16 based feature extractor for Content and Style loss calculation."""
    def __init__(self):
        super().__init__()
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features
        self.l1 = vgg[:4]   # relu1_2
        self.l2 = vgg[4:9]  # relu2_2
        self.l3 = vgg[9:16] # relu3_3
        self.l4 = vgg[16:23]# relu4_3
        for p in self.parameters(): p.requires_grad = False
    def forward(self, x):
        h1 = self.l1(x)
        h2 = self.l2(h1)
        h3 = self.l3(h2)
        h4 = self.l4(h3)
        return [h1, h2, h3, h4]

def gram(x):
    """Computes the Gram Matrix for Style Loss."""
    (b, c, h, w) = x.size()
    feat = x.view(b, c, h * w)
    return feat.bmm(feat.transpose(1, 2)) / (c * h * w)

def nnfm_loss(x, y):
    """Nearest Neighbor Feature Matching Loss for patch-based style transfer."""
    b, c, h, w = x.shape
    x_flat = x.view(b, c, -1).permute(0, 2, 1)
    y_flat = y.view(1, c, -1).permute(0, 2, 1)
    dist = torch.cdist(x_flat, y_flat)
    return dist.min(dim=2)[0].mean()

def tv_loss(x):
    """Total Variation Loss for edge smoothing and noise reduction."""
    return (torch.abs(x[:,:,:,:-1]-x[:,:,:,1:]).mean() +
            torch.abs(x[:,:,:-1,:]-x[:,:,1:,:]).mean())

def warp(x, flow):
    """Warps an image based on Optical Flow for temporal consistency."""
    b, c, h, w = x.size()
    grid_y, grid_x = torch.meshgrid(torch.arange(0, h), torch.arange(0, w), indexing='ij')
    grid = torch.stack((grid_x, grid_y), 2).float().to(x.device).unsqueeze(0).expand(b, -1, -1, -1)
    v_grid = grid + flow.permute(0, 2, 3, 1)
    v_grid[:, :, :, 0] = 2.0 * v_grid[:, :, :, 0] / (w - 1) - 1.0
    v_grid[:, :, :, 1] = 2.0 * v_grid[:, :, :, 1] / (h - 1) - 1.0
    output = F.grid_sample(x, v_grid, padding_mode='reflection', align_corners=True)
    mask = ((v_grid[:, :, :, 0] < -1) | (v_grid[:, :, :, 0] > 1) | 
            (v_grid[:, :, :, 1] < -1) | (v_grid[:, :, :, 1] > 1)).float().unsqueeze(1)
    return output, mask

# ============================================================
# TEMPORAL DATASET LOADER
# ============================================================
class TemporalDatasetMI300X(Dataset):
    """Loads RGB, Depth, and Motion buffers for 7-channel training."""
    def __init__(self, root, img_size=IMG_SIZE):
        self.root = root
        self.files = sorted(os.listdir(os.path.join(root, 'rgb')))
        self.t = transforms.Compose([transforms.Resize((img_size, img_size)), transforms.ToTensor()])
        self.sz = img_size
    def __len__(self): return len(self.files) - 1
    def _load_depth(self, path):
        """Loads R16_UNorm depth maps with full range preservation."""
        raw = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if raw is None:
            raw = np.array(Image.open(path)).astype(np.float32)
            norm = raw / raw.max() if raw.max() > 0 else raw
        else: norm = raw.astype(np.float32) / 65535.0
        t = torch.from_numpy(norm).unsqueeze(0).unsqueeze(0)
        return F.interpolate(t, size=(self.sz, self.sz), mode='bilinear').squeeze(0)
    def __getitem__(self, i):
        def path(folder, f): return os.path.join(self.root, folder, f)
        rgb = self.t(Image.open(path('rgb', self.files[i+1])).convert('RGB'))
        prev = self.t(Image.open(path('rgb', self.files[i])).convert('RGB'))
        bg_rgb = self.t(Image.open(path('bg_rgb', self.files[i+1])).convert('RGB'))
        dep = self._load_depth(path('depth', self.files[i+1]))
        bg_dep = self._load_depth(path('bg_depth', self.files[i+1]))
        flow = torch.from_numpy(cv2.resize(cv2.imread(path('motion', self.files[i+1].replace('.png', '.exr')), cv2.IMREAD_UNCHANGED)[:,:,:2], (self.sz, self.sz))).permute(2,0,1)
        return rgb, dep, prev, flow, bg_rgb, bg_dep

def load_style_grams(vgg):
    """Pre-computes average Gram Matrices for all reference style images."""
    t = transforms.Compose([transforms.Resize((IMG_SIZE, IMG_SIZE)), transforms.ToTensor()])
    imgs = []
    for f in sorted(os.listdir(STYLE_DIR)):
        if f.lower().endswith(('.png', '.jpg')):
            imgs.append(t(Image.open(os.path.join(STYLE_DIR, f)).convert('RGB')).unsqueeze(0).to(device))
    all_g = []
    for img in imgs:
        feats = vgg(vgg_norm(img))
        all_g.append([gram(f) for f in feats])
    avg_grams = [torch.stack([g[li] for g in all_g]).mean(0) for li in range(len(all_g[0]))]
    return avg_grams, vgg(vgg_norm(imgs[0]))[3].detach()

# ============================================================
# MAIN TRAINING LOOP
# ============================================================
def train():
    model = StyleTransferNetMI300X().to(device)
    vgg = VGGExtractorDeep().to(device).eval()
    
    start_epoch = 0
    if os.path.exists(CHECKPOINT_PATH):
        try:
            ckpt = torch.load(CHECKPOINT_PATH, map_location=device)
            model.load_state_dict(ckpt['model'], strict=True)
            start_epoch = ckpt.get('epoch', 0)
            print(f"[OK] Resuming from Checkpoint: Epoch {start_epoch}")
        except: print("[NEW] Starting training from scratch.")

    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
    
    loader = DataLoader(TemporalDatasetMI300X("NeuralDataset_V5"), batch_size=BATCH_SIZE, shuffle=True, num_workers=8)
    gram_targets, style_nnfm_3 = load_style_grams(vgg)
    
    # Art Configuration V5.1 (MI300X Edition) — Aggressive Style & Zero Flicker
    w_cont=1.5; w_gram=2e6; w_nnfm=0.05; w_lum=4.0; w_temp=150.0; w_tv=8e-4; w_color=50.0

    for epoch in range(start_epoch, EPOCHS):
        model.train()
        t0 = time.time()
        for i, (rgb, dep, prev_rgb, flow, bg_rgb, bg_dep) in enumerate(loader):
            rgb, dep, flow, prev_rgb = rgb.to(device), dep.to(device), flow.to(device), prev_rgb.to(device)
            bg_rgb, bg_dep = bg_rgb.to(device), bg_dep.to(device)
            
            # Hybrid Depth & Weapon Masking
            weapon_mask = (torch.abs(rgb - bg_rgb).mean(dim=1, keepdim=True) > 0.05).float()
            dep_clean = dep * weapon_mask + bg_dep * (1.0 - weapon_mask)
            
            # Optical Flow Warping for Temporal Consistency
            warped_prev, oob_mask = warp(rgb, flow) 
            warped_prev_rgb, _ = warp(prev_rgb, flow)
            temporal_mask = (torch.abs(rgb - warped_prev_rgb).mean(1, keepdim=True) < 0.1).float() * (1.0 - oob_mask)
            warped_prev_clean = warped_prev * temporal_mask + rgb * (1.0 - temporal_mask)

            # Generator Forward Pass
            output = model(torch.cat((rgb, dep_clean, warped_prev_clean), dim=1))
            
            # Feature Extraction
            feat_out = vgg(vgg_norm(output))
            feat_rgb = vgg(vgg_norm(rgb))

            # MULTI-OBJECTIVE LOSS FUNCTIONS
            loss_cont = F.mse_loss(feat_out[2], feat_rgb[2])
            loss_gram = sum(F.mse_loss(gram(feat_out[k]), gram_targets[k].expand(output.shape[0],-1,-1)) for k in range(len(gram_targets)))
            loss_nnfm = nnfm_loss(feat_out[3], style_nnfm_3)
            loss_temp = F.mse_loss(output, warped_prev_clean) # GLOBAL Temporal Stability
            loss_lum = F.mse_loss(output.mean(1,keepdim=True), rgb.mean(1,keepdim=True))
            loss_color = F.mse_loss(output, rgb)
            loss_tv = tv_loss(output)

            # Total Weighted Loss
            total = loss_cont*w_cont + loss_gram*w_gram + loss_nnfm*w_nnfm + loss_temp*w_temp + loss_lum*w_lum + loss_color*w_color + loss_tv*w_tv
            
            optimizer.zero_grad()
            total.backward()
            optimizer.step()

            if i % 100 == 0:
                print(f"E{epoch}[{i}] Loss:{total.item():.3f} T:{time.time()-t0:.1f}s")
                utils.save_image(torch.cat([rgb[0:1].cpu(), output[0:1].cpu()], dim=3), f"training_previews_mi300x/ep{epoch}_s{i}.png")
        
        scheduler.step()
        torch.save({'model': model.state_dict(), 'epoch': epoch}, CHECKPOINT_PATH)

if __name__ == "__main__":
    os.makedirs("training_previews_mi300x", exist_ok=True)
    train()
