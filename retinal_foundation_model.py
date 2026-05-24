"""
Self-Supervised Retinal Foundation Model for Joint Glaucoma Detection
and Cup-to-Disc Ratio (CDR) Estimation

Paper: "A Self-Supervised Retinal Foundation Model for Joint Glaucoma Detection
        and Cup-to-Disc Ratio Estimation"
Authors: Nivethitha N, Dr. A. Rajeswari, Sruthi Nath C, Dr. R. Amirthavalli

Architecture:
  - ViT-B/16 encoder backbone
  - Self-supervised pre-training: Masked Image Modelling (MIM) + Domain-Adaptive
    Contrastive Learning (DACL)
  - Dual-head prediction: glaucoma classification + CDR regression
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.transforms.functional as TF
from PIL import Image
import random


# ─────────────────────────────────────────────────────────────────
# 1.  PREPROCESSING PIPELINE
# ─────────────────────────────────────────────────────────────────

class CLAHETransform:
    """Apply CLAHE to the green channel (clip_limit=2.0, tile_grid=8×8)."""

    def __init__(self, clip_limit: float = 2.0, tile_grid_size: int = 8):
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size

    def __call__(self, img: Image.Image) -> Image.Image:
        try:
            import cv2
        except ImportError:
            return img  # graceful fallback if cv2 unavailable
        img_np = np.array(img)
        if img_np.ndim == 3:
            green = img_np[:, :, 1]
            clahe = cv2.createCLAHE(
                clipLimit=self.clip_limit,
                tileGridSize=(self.tile_grid_size, self.tile_grid_size),
            )
            img_np[:, :, 1] = clahe.apply(green)
        return Image.fromarray(img_np)


def build_train_transform() -> transforms.Compose:
    """Stochastic augmentation pipeline used during training (Section 3.2)."""
    return transforms.Compose([
        transforms.Resize(512, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),          # ROI crop → 224×224 for ViT
        CLAHETransform(clip_limit=2.0, tile_grid_size=8),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=30),
        transforms.ColorJitter(brightness=0.2, contrast=0.2,
                               saturation=0.2, hue=0.1),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=5)], p=0.3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.2),
    ])


def build_val_transform() -> transforms.Compose:
    """Deterministic pipeline for validation / inference (no augmentation)."""
    return transforms.Compose([
        transforms.Resize(512, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        CLAHETransform(clip_limit=2.0, tile_grid_size=8),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


# ─────────────────────────────────────────────────────────────────
# 2.  VISION TRANSFORMER BACKBONE  (ViT-B/16)
# ─────────────────────────────────────────────────────────────────

class PatchEmbed(nn.Module):
    """Split image into non-overlapping 16×16 patches and project to d_model."""

    def __init__(self, img_size: int = 224, patch_size: int = 16,
                 in_channels: int = 3, embed_dim: int = 768):
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2   # 196 for 224/16
        self.proj = nn.Conv2d(in_channels, embed_dim,
                              kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W) → (B, N, D)
        x = self.proj(x)                  # (B, D, H/P, W/P)
        x = x.flatten(2).transpose(1, 2) # (B, N, D)
        return x


class MultiHeadSelfAttention(nn.Module):
    """Scaled dot-product multi-head self-attention (Equation 4)."""

    def __init__(self, embed_dim: int = 768, num_heads: int = 12,
                 attn_drop: float = 0.0, proj_drop: float = 0.0):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=True)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)   # (3, B, H, N, d_k)
        q, k, v = qkv.unbind(0)             # each (B, H, N, d_k)

        attn = (q @ k.transpose(-2, -1)) * self.scale   # (B, H, N, N)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, D)
        x = self.proj_drop(self.proj(x))
        return x


class TransformerBlock(nn.Module):
    """Single ViT-B transformer block: MHSA + FFN with LayerNorm."""

    def __init__(self, embed_dim: int = 768, num_heads: int = 12,
                 mlp_ratio: float = 4.0, drop: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads,
                                           proj_drop=drop)
        self.norm2 = nn.LayerNorm(embed_dim)
        hidden = int(embed_dim * mlp_ratio)   # 3072 for ViT-B
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, hidden),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(hidden, embed_dim),
            nn.Dropout(drop),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class VisionTransformerEncoder(nn.Module):
    """
    ViT-B/16 encoder (Section 3.3.2).

    Outputs h_CLS ∈ R^768 — the [CLS] token from the final layer,
    used as global image representation for downstream heads.
    """

    def __init__(self, img_size: int = 224, patch_size: int = 16,
                 in_channels: int = 3, embed_dim: int = 768,
                 depth: int = 12, num_heads: int = 12,
                 mlp_ratio: float = 4.0, drop_rate: float = 0.0):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size, patch_size,
                                      in_channels, embed_dim)
        num_patches = self.patch_embed.num_patches           # 196

        # Learnable [CLS] token and positional embeddings
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, embed_dim))     # +1 for CLS
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        self.blocks = nn.Sequential(*[
            TransformerBlock(embed_dim, num_heads, mlp_ratio, drop_rate)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns h_CLS: (B, embed_dim)."""
        B = x.shape[0]
        x = self.patch_embed(x)                              # (B, 196, D)
        cls = self.cls_token.expand(B, -1, -1)               # (B, 1, D)
        x = torch.cat([cls, x], dim=1)                       # (B, 197, D)
        x = x + self.pos_embed
        x = self.blocks(x)
        x = self.norm(x)
        return x[:, 0]                                        # h_CLS


# ─────────────────────────────────────────────────────────────────
# 3.  DUAL-HEAD PREDICTION MODULE  (Section 3.3.3)
# ─────────────────────────────────────────────────────────────────

class GlaucomaClassificationHead(nn.Module):
    """
    Two-layer MLP → sigmoid binary glaucoma probability (Equation 5).
    W1 ∈ R^{256×768}, W2 ∈ R^{1×256}
    """

    def __init__(self, embed_dim: int = 768, hidden_dim: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, h_cls: torch.Tensor) -> torch.Tensor:
        return self.mlp(h_cls).squeeze(-1)   # (B,)


class CDRRegressionHead(nn.Module):
    """
    Two-layer MLP → sigmoid CDR estimate constrained to (0, 1) (Equation 6).
    W3 ∈ R^{256×768}, W4 ∈ R^{1×256}
    """

    def __init__(self, embed_dim: int = 768, hidden_dim: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, h_cls: torch.Tensor) -> torch.Tensor:
        return self.mlp(h_cls).squeeze(-1)   # (B,) in (0, 1)


class RetinalFoundationModel(nn.Module):
    """
    Full fine-tuning model: shared ViT-B/16 encoder + dual heads.
    """

    def __init__(self, **vit_kwargs):
        super().__init__()
        self.encoder = VisionTransformerEncoder(**vit_kwargs)
        self.cls_head = GlaucomaClassificationHead()
        self.cdr_head = CDRRegressionHead()

    def forward(self, x: torch.Tensor):
        h = self.encoder(x)
        return self.cls_head(h), self.cdr_head(h)   # (B,), (B,)


# ─────────────────────────────────────────────────────────────────
# 4.  SELF-SUPERVISED PRE-TRAINING MODULE  (Section 3.3.1)
# ─────────────────────────────────────────────────────────────────

class MIMDecoder(nn.Module):
    """
    Lightweight pixel-space decoder that reconstructs masked 16×16 patches
    from encoded representations (Equation 1).
    """

    def __init__(self, embed_dim: int = 768, patch_size: int = 16,
                 num_patches: int = 196):
        super().__init__()
        patch_pixels = patch_size * patch_size * 3   # 768
        self.decoder = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, patch_pixels),
        )

    def forward(self, encoded: torch.Tensor) -> torch.Tensor:
        """encoded: (B, N, D) — only masked tokens passed in practice."""
        return self.decoder(encoded)   # (B, N, patch_pixels)


def mim_loss(reconstructed: torch.Tensor,
             target: torch.Tensor) -> torch.Tensor:
    """
    L_MIM = (1/|M|) Σ_{i∈M} ||x_i − x̂_i||²₂  (Equation 1).

    reconstructed, target: (B, |M|, patch_pixels)
    """
    return F.mse_loss(reconstructed, target)


class MLPProjectionHead(nn.Module):
    """Two-layer MLP projection head for DACL (NT-Xent)."""

    def __init__(self, embed_dim: int = 768, proj_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, proj_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(x), dim=-1)


def nt_xent_loss(z: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
    """
    NT-Xent (normalised temperature-scaled cross-entropy) for DACL.
    Equation 2: L_DACL = −(1/N) Σ log[ exp(sim(zi,zj)/τ) /
                                        Σ_{k≠i} exp(sim(zi,zk)/τ) ]

    z: (2N, proj_dim) — concatenation of two augmented-view embeddings.
        rows 0..N-1 = view-A; rows N..2N-1 = view-B.
    """
    two_n = z.shape[0]
    n = two_n // 2

    sim = torch.mm(z, z.T) / temperature              # (2N, 2N)
    # Mask out self-similarity
    mask = torch.eye(two_n, dtype=torch.bool, device=z.device)
    sim = sim.masked_fill(mask, float('-inf'))

    # Positive pairs: (i, i+N) and (i+N, i)
    pos_idx = torch.cat([torch.arange(n, two_n), torch.arange(n)]).to(z.device)
    loss = F.cross_entropy(sim, pos_idx)
    return loss


class SelfSupervisedPreTrainer(nn.Module):
    """
    Pre-training wrapper: MIM + DACL compound objective (Equation 3).
    L_pretrain = L_MIM + λ · L_DACL,  λ = 0.5
    """

    def __init__(self, encoder: VisionTransformerEncoder,
                 mask_ratio: float = 0.75, lam: float = 0.5,
                 temperature: float = 0.07, proj_dim: int = 128):
        super().__init__()
        self.encoder = encoder
        self.decoder = MIMDecoder()
        self.proj_head = MLPProjectionHead(proj_dim=proj_dim)
        self.mask_ratio = mask_ratio
        self.lam = lam
        self.temperature = temperature
        self.patch_size = 16

    # ----------------------------------------------------------
    # Masking helpers
    # ----------------------------------------------------------
    def _random_mask(self, n_patches: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (visible_idx, masked_idx) with mask_ratio fraction masked."""
        n_masked = int(n_patches * self.mask_ratio)
        shuffle = torch.randperm(n_patches)
        visible = shuffle[n_masked:].sort().values
        masked = shuffle[:n_masked].sort().values
        return visible, masked

    def _patches_to_target(self, imgs: torch.Tensor,
                            masked_idx: torch.Tensor) -> torch.Tensor:
        """
        Extract normalised pixel values for masked patches.
        Returns (B, |M|, patch_pixels).
        """
        B, C, H, W = imgs.shape
        p = self.patch_size
        n_h = H // p
        # Unfold into patches: (B, C, n_h, n_w, p, p)
        patches = imgs.unfold(2, p, p).unfold(3, p, p)
        patches = patches.contiguous().view(B, C, -1, p, p)
        patches = patches.permute(0, 2, 1, 3, 4).contiguous()   # (B,N,C,p,p)
        patches = patches.view(B, -1, C * p * p)                  # (B,N,Cp²)
        return patches[:, masked_idx, :]

    # ----------------------------------------------------------
    # Forward
    # ----------------------------------------------------------
    def forward(self, img_a: torch.Tensor, img_b: torch.Tensor):
        """
        img_a, img_b: two augmented views of the same batch, shape (B,3,224,224).
        Returns scalar pre-training loss.
        """
        B = img_a.shape[0]
        num_patches = (224 // self.patch_size) ** 2    # 196

        # ── MIM on view-A ──────────────────────────────────────
        visible_idx, masked_idx = self._random_mask(num_patches)

        # Encode only visible patches (simple approximation: encode full image,
        # then select masked positions for reconstruction)
        h_full = self._encode_full(img_a)                     # (B, N, D)
        masked_enc = h_full[:, masked_idx, :]                 # (B, |M|, D)
        recon = self.decoder(masked_enc)                      # (B, |M|, Cp²)
        target = self._patches_to_target(img_a, masked_idx)  # (B, |M|, Cp²)
        l_mim = mim_loss(recon, target)

        # ── DACL on both views ─────────────────────────────────
        cls_a = self.encoder(img_a)   # (B, D)
        cls_b = self.encoder(img_b)
        z_a = self.proj_head(cls_a)   # (B, proj_dim)
        z_b = self.proj_head(cls_b)
        z = torch.cat([z_a, z_b], dim=0)   # (2B, proj_dim)
        l_dacl = nt_xent_loss(z, self.temperature)

        # ── Combined loss (Equation 3) ──────────────────────────
        l_pretrain = l_mim + self.lam * l_dacl
        return l_pretrain, l_mim, l_dacl

    def _encode_full(self, x: torch.Tensor) -> torch.Tensor:
        """
        Run the ViT patch-embedding + positional encoding + blocks and return
        ALL token representations (B, N+1, D); excludes the [CLS] token slice.
        """
        B = x.shape[0]
        feats = self.encoder.patch_embed(x)                  # (B, N, D)
        cls = self.encoder.cls_token.expand(B, -1, -1)
        feats = torch.cat([cls, feats], dim=1) + self.encoder.pos_embed
        feats = self.encoder.blocks(feats)
        feats = self.encoder.norm(feats)
        return feats[:, 1:, :]   # drop CLS, keep patch tokens (B, N, D)


# ─────────────────────────────────────────────────────────────────
# 5.  LOSS FUNCTIONS  (Section 3.4)
# ─────────────────────────────────────────────────────────────────

class FineTuningLoss(nn.Module):
    """
    L_total = α · L_BCE + (1 − α) · L_MAE    (Equation 7)
    α = 0.6 (empirically determined).
    """

    def __init__(self, alpha: float = 0.6):
        super().__init__()
        self.alpha = alpha

    def forward(self,
                y_cls_pred: torch.Tensor, y_cls_true: torch.Tensor,
                y_cdr_pred: torch.Tensor, y_cdr_true: torch.Tensor):
        l_bce = F.binary_cross_entropy(y_cls_pred, y_cls_true.float())  # Eq. 8
        l_mae = F.l1_loss(y_cdr_pred, y_cdr_true.float())               # Eq. 9
        return self.alpha * l_bce + (1 - self.alpha) * l_mae, l_bce, l_mae


# ─────────────────────────────────────────────────────────────────
# 6.  EVALUATION METRICS  (Section 3.6)
# ─────────────────────────────────────────────────────────────────

def compute_metrics(y_true: np.ndarray, y_pred_prob: np.ndarray,
                    cdr_true: np.ndarray, cdr_pred: np.ndarray,
                    threshold: float = 0.5) -> dict:
    """
    Computes AUC, F1, Accuracy (Equations 10-12),
    CDR MAE and Pearson ρ (Equations 13-14).
    """
    from sklearn.metrics import (roc_auc_score, f1_score,
                                  accuracy_score, confusion_matrix)
    from scipy.stats import pearsonr

    y_pred_bin = (y_pred_prob >= threshold).astype(int)

    auc = roc_auc_score(y_true, y_pred_prob)
    f1 = f1_score(y_true, y_pred_bin)
    acc = accuracy_score(y_true, y_pred_bin)
    mae = float(np.mean(np.abs(cdr_true - cdr_pred)))
    rho, _ = pearsonr(cdr_true, cdr_pred)

    return {"AUC": auc, "F1": f1, "Accuracy": acc,
            "CDR_MAE": mae, "CDR_Pearson": rho}


# ─────────────────────────────────────────────────────────────────
# 7.  DATASET WRAPPERS
# ─────────────────────────────────────────────────────────────────

class GlaucomaDataset(Dataset):
    """
    Generic fundus image dataset for fine-tuning.
    Expects:
        image_paths  : list of str paths to images
        labels       : list[int] (0=normal, 1=glaucoma) or None
        cdr_values   : list[float] CDR values or None
        transform    : torchvision transform
    """

    def __init__(self, image_paths, labels=None,
                 cdr_values=None, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.cdr_values = cdr_values
        self.transform = transform or build_val_transform()

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        label = torch.tensor(self.labels[idx], dtype=torch.long) \
            if self.labels else torch.tensor(-1)
        cdr = torch.tensor(self.cdr_values[idx], dtype=torch.float32) \
            if self.cdr_values else torch.tensor(-1.0)
        return img, label, cdr


class UnlabelledFundusDataset(Dataset):
    """
    Unlabelled dataset for self-supervised pre-training.
    Returns two augmented views of each image.
    """

    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform or build_train_transform()

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        return self.transform(img), self.transform(img)


# ─────────────────────────────────────────────────────────────────
# 8.  TRAINING LOOPS
# ─────────────────────────────────────────────────────────────────

def pretrain_one_epoch(model: SelfSupervisedPreTrainer,
                       loader: DataLoader,
                       optimizer: torch.optim.Optimizer,
                       device: torch.device) -> dict:
    """Single pre-training epoch. Returns avg losses."""
    model.train()
    total_loss = total_mim = total_dacl = 0.0

    for img_a, img_b in loader:
        img_a, img_b = img_a.to(device), img_b.to(device)
        optimizer.zero_grad()
        loss, l_mim, l_dacl = model(img_a, img_b)
        loss.backward()
        optimizer.step()
        n = img_a.size(0)
        total_loss += loss.item() * n
        total_mim  += l_mim.item() * n
        total_dacl += l_dacl.item() * n

    N = len(loader.dataset)
    return {"loss": total_loss / N, "mim": total_mim / N,
            "dacl": total_dacl / N}


def finetune_one_epoch(model: RetinalFoundationModel,
                       loader: DataLoader,
                       criterion: FineTuningLoss,
                       optimizer: torch.optim.Optimizer,
                       device: torch.device) -> dict:
    """Single fine-tuning epoch. Returns avg losses."""
    model.train()
    total = total_bce = total_mae = 0.0

    for imgs, labels, cdrs in loader:
        imgs = imgs.to(device)
        labels = labels.float().to(device)
        cdrs = cdrs.float().to(device)

        optimizer.zero_grad()
        pred_cls, pred_cdr = model(imgs)
        loss, l_bce, l_mae = criterion(pred_cls, labels, pred_cdr, cdrs)
        loss.backward()
        optimizer.step()
        n = imgs.size(0)
        total     += loss.item() * n
        total_bce += l_bce.item() * n
        total_mae += l_mae.item() * n

    N = len(loader.dataset)
    return {"loss": total / N, "bce": total_bce / N, "mae": total_mae / N}


@torch.no_grad()
def evaluate(model: RetinalFoundationModel,
             loader: DataLoader,
             device: torch.device) -> dict:
    """Evaluate on validation/test split, returns metrics dict."""
    model.eval()
    all_probs, all_labels, all_cdr_pred, all_cdr_true = [], [], [], []

    for imgs, labels, cdrs in loader:
        imgs = imgs.to(device)
        pred_cls, pred_cdr = model(imgs)
        all_probs.append(pred_cls.cpu().numpy())
        all_labels.append(labels.numpy())
        all_cdr_pred.append(pred_cdr.cpu().numpy())
        all_cdr_true.append(cdrs.numpy())

    probs = np.concatenate(all_probs)
    labels = np.concatenate(all_labels)
    cdr_pred = np.concatenate(all_cdr_pred)
    cdr_true = np.concatenate(all_cdr_true)

    return compute_metrics(labels, probs, cdr_true, cdr_pred)


# ─────────────────────────────────────────────────────────────────
# 9.  COSINE LR SCHEDULE WITH WARMUP
# ─────────────────────────────────────────────────────────────────

def cosine_schedule_with_warmup(optimizer: torch.optim.Optimizer,
                                 warmup_epochs: int, total_epochs: int,
                                 base_lr: float, min_lr: float = 1e-6):
    """
    Implements cosine decay with linear warm-up as described in Table 2.
    Returns a LambdaLR scheduler.
    """
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return epoch / max(1, warmup_epochs)
        progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        return max(min_lr / base_lr, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ─────────────────────────────────────────────────────────────────
# 10. FULL TRAINING PIPELINE
# ─────────────────────────────────────────────────────────────────

def pretrain(image_paths: list, device: str = "cuda",
             epochs: int = 200, batch_size: int = 256,
             lr: float = 1e-3, weight_decay: float = 0.05,
             warmup_epochs: int = 40, lam: float = 0.5,
             save_path: str = "pretrained_encoder.pth"):
    """
    Algorithm 1 — Self-supervised pre-training on unlabelled fundus images.

    Parameters
    ----------
    image_paths : paths to ~300 k unlabelled fundus images
    device      : 'cuda' or 'cpu'
    epochs      : 200 (Table 2)
    batch_size  : 256 (Table 2)
    lr          : 1e-3 (Table 2)
    weight_decay: 0.05 (Table 2)
    warmup_epochs: 40 (Table 2)
    lam         : λ for combined loss, 0.5 (Equation 3)
    save_path   : where to save the pre-trained encoder weights
    """
    dev = torch.device(device if torch.cuda.is_available() else "cpu")

    encoder = VisionTransformerEncoder()
    model = SelfSupervisedPreTrainer(encoder, lam=lam).to(dev)

    dataset = UnlabelledFundusDataset(image_paths, build_train_transform())
    loader = DataLoader(dataset, batch_size=batch_size,
                        shuffle=True, num_workers=8, pin_memory=True)

    optimizer = torch.optim.AdamW(model.parameters(),
                                   lr=lr, weight_decay=weight_decay)
    scheduler = cosine_schedule_with_warmup(
        optimizer, warmup_epochs, epochs, lr)

    print(f"Starting pre-training on {len(dataset):,} images "
          f"for {epochs} epochs …")
    for epoch in range(1, epochs + 1):
        metrics = pretrain_one_epoch(model, loader, optimizer, dev)
        scheduler.step()
        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}/{epochs} | "
                  f"loss={metrics['loss']:.4f}  "
                  f"mim={metrics['mim']:.4f}  "
                  f"dacl={metrics['dacl']:.4f}  "
                  f"lr={scheduler.get_last_lr()[0]:.2e}")

    torch.save(model.encoder.state_dict(), save_path)
    print(f"\nPre-trained encoder saved to {save_path}")
    return model.encoder


def finetune(train_paths, train_labels, train_cdrs,
             val_paths, val_labels, val_cdrs,
             pretrained_weights: str = None,
             device: str = "cuda",
             epochs: int = 50, batch_size: int = 32,
             lr: float = 1e-4, weight_decay: float = 0.01,
             alpha: float = 0.6, patience: int = 10,
             encoder_lr_scale: float = 0.1,
             save_path: str = "finetuned_model.pth") -> RetinalFoundationModel:
    """
    Fine-tuning on labelled downstream data (Section 3.4).

    Parameters
    ----------
    *_paths  : file paths for images
    *_labels : binary glaucoma labels (0/1)
    *_cdrs   : ground-truth CDR float values
    pretrained_weights : path to encoder checkpoint from pretrain()
    encoder_lr_scale   : encoder LR = lr * encoder_lr_scale (0.1 × head LR)
    patience           : early stopping patience on val AUC
    """
    dev = torch.device(device if torch.cuda.is_available() else "cpu")

    model = RetinalFoundationModel().to(dev)
    if pretrained_weights:
        model.encoder.load_state_dict(
            torch.load(pretrained_weights, map_location=dev))
        print(f"Loaded pre-trained encoder from {pretrained_weights}")

    train_ds = GlaucomaDataset(train_paths, train_labels,
                                train_cdrs, build_train_transform())
    val_ds   = GlaucomaDataset(val_paths, val_labels,
                                val_cdrs,   build_val_transform())
    train_loader = DataLoader(train_ds, batch_size=batch_size,
                               shuffle=True, num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size,
                               shuffle=False, num_workers=4, pin_memory=True)

    # Encoder LR scaled by 0.1 relative to task heads
    optimizer = torch.optim.Adam([
        {"params": model.encoder.parameters(), "lr": lr * encoder_lr_scale},
        {"params": model.cls_head.parameters()},
        {"params": model.cdr_head.parameters()},
    ], lr=lr, betas=(0.9, 0.999), weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs)
    criterion = FineTuningLoss(alpha=alpha)

    best_auc = 0.0
    no_improve = 0

    print(f"Starting fine-tuning for up to {epochs} epochs …")
    for epoch in range(1, epochs + 1):
        train_m = finetune_one_epoch(model, train_loader,
                                     criterion, optimizer, dev)
        val_m   = evaluate(model, val_loader, dev)
        scheduler.step()

        auc = val_m["AUC"]
        print(f"Epoch {epoch:2d}/{epochs} | "
              f"train_loss={train_m['loss']:.4f} | "
              f"val_AUC={auc:.4f}  F1={val_m['F1']:.4f}  "
              f"CDR_MAE={val_m['CDR_MAE']:.4f}")

        if auc > best_auc:
            best_auc = auc
            no_improve = 0
            torch.save(model.state_dict(), save_path)
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Early stopping triggered at epoch {epoch}.")
                break

    model.load_state_dict(torch.load(save_path, map_location=dev))
    print(f"\nBest val AUC: {best_auc:.4f} — model saved to {save_path}")
    return model


# ─────────────────────────────────────────────────────────────────
# 11. INFERENCE
# ─────────────────────────────────────────────────────────────────

@torch.no_grad()
def predict(model: RetinalFoundationModel,
            image_path: str,
            device: str = "cuda",
            glaucoma_threshold: float = 0.5) -> dict:
    """
    Run inference on a single fundus image.

    Returns
    -------
    dict with keys:
        glaucoma_prob  : float in (0, 1)
        glaucoma_pred  : bool
        cdr_estimate   : float in (0, 1)
    """
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    model.eval().to(dev)

    img = Image.open(image_path).convert("RGB")
    transform = build_val_transform()
    x = transform(img).unsqueeze(0).to(dev)

    cls_prob, cdr_est = model(x)
    return {
        "glaucoma_prob": float(cls_prob.item()),
        "glaucoma_pred": cls_prob.item() >= glaucoma_threshold,
        "cdr_estimate":  float(cdr_est.item()),
    }


# ─────────────────────────────────────────────────────────────────
# 12. QUICK SMOKE-TEST
# ─────────────────────────────────────────────────────────────────

def _smoke_test():
    """Verify all components run without shape errors on random tensors."""
    print("Running smoke test …")
    device = torch.device("cpu")

    # -- Encoder
    enc = VisionTransformerEncoder()
    x = torch.randn(2, 3, 224, 224)
    h = enc(x)
    assert h.shape == (2, 768), f"Encoder output shape mismatch: {h.shape}"

    # -- Dual-head model
    model = RetinalFoundationModel()
    cls_out, cdr_out = model(x)
    assert cls_out.shape == (2,), f"CLS head shape: {cls_out.shape}"
    assert cdr_out.shape == (2,), f"CDR head shape: {cdr_out.shape}"
    assert cls_out.min() >= 0 and cls_out.max() <= 1, "CLS must be in (0,1)"
    assert cdr_out.min() >= 0 and cdr_out.max() <= 1, "CDR must be in (0,1)"

    # -- Pre-training loss
    trainer = SelfSupervisedPreTrainer(enc)
    img_a = torch.randn(2, 3, 224, 224)
    img_b = torch.randn(2, 3, 224, 224)
    loss, l_mim, l_dacl = trainer(img_a, img_b)
    assert loss.item() > 0, "Pre-train loss must be positive"

    # -- Fine-tuning loss
    crit = FineTuningLoss()
    labels = torch.tensor([1.0, 0.0])
    cdrs   = torch.tensor([0.6, 0.4])
    total, bce, mae = crit(cls_out, labels, cdr_out, cdrs)
    assert total.item() > 0, "Fine-tune loss must be positive"

    print("✓  All smoke tests passed.\n")
    print(f"   Encoder h_CLS shape : {h.shape}")
    print(f"   Glaucoma prob range  : [{cls_out.min():.3f}, {cls_out.max():.3f}]")
    print(f"   CDR estimate range   : [{cdr_out.min():.3f}, {cdr_out.max():.3f}]")
    print(f"   L_pretrain           : {loss.item():.4f}  "
          f"(MIM={l_mim.item():.4f}, DACL={l_dacl.item():.4f})")
    print(f"   L_total (finetune)   : {total.item():.4f}  "
          f"(BCE={bce.item():.4f}, MAE={mae.item():.4f})")


if __name__ == "__main__":
    _smoke_test()

    # ── Example usage ───────────────────────────────────────────
    # STEP 1 – Self-supervised pre-training
    # unlabelled_paths = glob.glob("/data/fundus/unlabelled/**/*.jpg")
    # encoder = pretrain(unlabelled_paths, device="cuda")

    # STEP 2 – Fine-tuning on REFUGE / RIM-ONE / ORIGA
    # model = finetune(
    #     train_paths, train_labels, train_cdrs,
    #     val_paths, val_labels, val_cdrs,
    #     pretrained_weights="pretrained_encoder.pth",
    # )

    # STEP 3 – Inference
    # result = predict(model, "path/to/fundus_image.jpg")
    # print(result)
