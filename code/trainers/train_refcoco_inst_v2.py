# code/trainers/train_refcoco_inst_v2.py
import os
import time
import argparse
import yaml
from typing import Dict, Any, Tuple, Optional, List
import csv, json
import math
import random
import tempfile

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# =======================
# モデル本体（修正後を利用）
# =======================
# try:
#     from models.qformer_refcoco_inst_v2 import QFormerRefCOCO, OpenCLIPVisionFrozen
# except Exception:
#     from models.qformer_refcoco_inst import QFormerRefCOCO, OpenCLIPVisionFrozen
from models.qformer_refcoco_inst import QFormerRefCOCO, OpenCLIPVisionFrozen

# =========================================================
# 追加ロス・数値安定ユーティリティ
# =========================================================

def _attn_to_map(attn_maps: torch.Tensor, hw: Tuple[int, int]) -> torch.Tensor:
    B, C, L = attn_maps.shape
    H, W = int(hw[0]), int(hw[1])
    A = attn_maps.view(B, C, H, W)
    A = A / (A.amax(dim=(2, 3), keepdim=True) + 1e-6)
    return A.clamp(0, 1)

def safe_bce_from_probs(prob: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    prob   = torch.nan_to_num(prob,   nan=0.0, posinf=1.0, neginf=0.0)
    target = torch.nan_to_num(target, nan=0.0, posinf=1.0, neginf=0.0)
    return (-(target * torch.log(prob + eps) + (1.0 - target) * torch.log1p(-prob + eps))).mean()

def safe_dice_from_probs(prob: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    prob   = torch.nan_to_num(prob,   nan=0.0, posinf=1.0, neginf=0.0)
    target = torch.nan_to_num(target, nan=0.0, posinf=1.0, neginf=0.0)
    inter = (prob * target).sum(dim=(2, 3))
    denom = prob.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
    return (1.0 - (2.0 * inter + eps) / (denom + eps)).mean()

def soft_iou_loss(prob: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    prob   = torch.nan_to_num(prob,   nan=0.0, posinf=1.0, neginf=0.0)
    target = torch.nan_to_num(target, nan=0.0, posinf=1.0, neginf=0.0)
    inter = (prob * target).sum(dim=(2, 3))
    union = (prob + target - prob * target).sum(dim=(2, 3)) + eps
    return (1.0 - inter / union).mean()

# == Lovasz
def lovasz_grad(gt_sorted):
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.cumsum(0)
    union = gts + (1 - gt_sorted).cumsum(0)
    jaccard = 1. - intersection / (union + 1e-6)
    if p > 1:
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard

def lovasz_hinge_flat(logits, labels):
    if logits.numel() == 0:
        return logits.sum() * 0.
    signs = 2. * labels - 1.
    errors = 1. - logits * signs
    errors_sorted, perm = torch.sort(errors, descending=True)
    gt_sorted = labels[perm]
    grad = lovasz_grad(gt_sorted)
    loss = torch.clamp(errors_sorted, min=0.)
    return torch.dot(loss, grad)

def lovasz_sigmoid(prob, target):
    # prob, target: (B,1,H,W) in [0,1]
    B = prob.size(0)
    loss = 0.
    for b in range(B):
        p = prob[b].view(-1)
        t = target[b].view(-1)
        logit = torch.log(p.clamp(1e-6,1-1e-6)) - torch.log1p(-p.clamp(1e-6,1-1e-6))
        loss = loss + lovasz_hinge_flat(logit, t)
    return loss / B

class InfoNCELoss(nn.Module):
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
    def forward(self, v: torch.Tensor, t: torch.Tensor):
        if v.ndim == 3: v = v.mean(dim=1)
        if t.ndim == 3: t = t.mean(dim=1)
        v = F.normalize(v, dim=-1); t = F.normalize(t, dim=-1)
        logits_vt = (v @ t.t()) / max(self.temperature, 1e-6)
        logits_tv = logits_vt.t()
        targets = torch.arange(v.size(0), device=v.device)
        loss_v = F.cross_entropy(logits_vt, targets)
        loss_t = F.cross_entropy(logits_tv, targets)
        return 0.5 * (loss_v + loss_t)

def zero_if_not_finite(x: torch.Tensor) -> torch.Tensor:
    return torch.where(torch.isfinite(x), x, torch.zeros_like(x))

# --- 追加: エッジ重み & Total Variation ---

def edge_weight_map(gt: torch.Tensor) -> torch.Tensor:
    sobel_x = torch.tensor([[1,0,-1],[2,0,-2],[1,0,-1]], device=gt.device, dtype=gt.dtype).view(1,1,3,3)
    sobel_y = sobel_x.transpose(2,3)
    gx = F.conv2d(gt, sobel_x, padding=1)
    gy = F.conv2d(gt, sobel_y, padding=1)
    mag = (gx.abs() + gy.abs())
    return (mag / (mag.amax(dim=(2,3), keepdim=True) + 1e-6)).clamp(0,1)

def total_variation(x: torch.Tensor) -> torch.Tensor:
    tv_h = (x[:,:,1:,:] - x[:,:,:-1,:]).abs().mean()
    tv_w = (x[:,:,:,1:] - x[:,:,:,:-1]).abs().mean()
    return tv_h + tv_w

# --- 追加: MixUp（Phase1向け） ---

def maybe_mixup(img: torch.Tensor, mask: torch.Tensor, alpha: float = 0.2, prob: float = 0.0):
    if prob <= 0.0 or torch.rand(1).item() > prob:
        return img, mask
    lam = np.random.beta(alpha, alpha)
    B = img.size(0)
    idx = torch.randperm(B, device=img.device)
    img2, m2 = img[idx], mask[idx]
    return lam * img + (1 - lam) * img2, lam * mask + (1 - lam) * m2

# --- 追加: EMA（CPUに格納して省メモリ） ---
class ModelEMA:
    def __init__(self, model: nn.Module, decay: float = 0.999, cpu: bool = True, trainable_only: bool = True):
        self.decay = float(decay)
        self.cpu = bool(cpu)
        self.trainable_only = bool(trainable_only)
        ema: dict = {}
        if trainable_only:
            for n, p in model.named_parameters():
                if p.requires_grad:
                    t = p.detach()
                    t = t.to("cpu", dtype=torch.float32) if self.cpu else t.to(dtype=p.dtype)
                    ema[n] = t.clone()
        else:
            for k, v in model.state_dict().items():
                if not torch.is_tensor(v):
                    continue
                t = v.detach()
                if self.cpu:
                    t = t.to("cpu", dtype=torch.float32)
                ema[k] = t.clone()
        self.ema = ema

    @torch.no_grad()
    def update(self, model: nn.Module):
        if self.trainable_only:
            for n, p in model.named_parameters():
                if not p.requires_grad:
                    continue
                vv = p.detach()
                vv = vv.to("cpu", dtype=torch.float32) if self.cpu else vv.to(dtype=p.dtype)
                if n not in self.ema:
                    self.ema[n] = vv.clone()
                else:
                    self.ema[n].mul_(self.decay).add_(vv, alpha=1 - self.decay)
        else:
            for k, v in model.state_dict().items():
                if not torch.is_tensor(v):
                    continue
                vv = v.detach()
                if self.cpu:
                    vv = vv.to("cpu", dtype=torch.float32)
                if k not in self.ema:
                    self.ema[k] = vv.clone()
                else:
                    self.ema[k].mul_(self.decay).add_(vv, alpha=1 - self.decay)

    @torch.no_grad()
    def copy_to(self, model: nn.Module):
        for n, p in model.named_parameters():
            if not p.requires_grad or n not in self.ema:
                continue
            src = self.ema[n]
            p.copy_(src.to(device=p.device, dtype=p.dtype))

# ========== ここから下は保存処理を堅牢化 ==========

def atomic_torch_save(obj, path, use_legacy=False):
    d = os.path.dirname(path)
    os.makedirs(d, exist_ok=True)
    fd, tmppath = tempfile.mkstemp(prefix=".tmp_ckpt_", suffix=".pt", dir=d)
    os.close(fd)
    try:
        torch.save(obj, tmppath, _use_new_zipfile_serialization=(not use_legacy))
        os.replace(tmppath, path)
    except Exception:
        try:
            if os.path.exists(tmppath):
                os.remove(tmppath)
        except Exception:
            pass
        raise

try:
    from safetensors.torch import save_file as safe_save
    def atomic_safe_save(state_dict: dict, path: str):
        d = os.path.dirname(path)
        os.makedirs(d, exist_ok=True)
        tmppath = path + ".tmp"
        safe_save(state_dict, tmppath)
        os.replace(tmppath, path)
except Exception:
    safe_save = None
    atomic_safe_save = None

# =========================================================
# Fallback Dataset（画像フォルダ＋任意のマスクフォルダ）
# =========================================================
_CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
_CLIP_STD  = (0.26862954, 0.26130258, 0.27577711)
_IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

class _SimpleRefCOCODataset(Dataset):
    def __init__(self, img_dir: str, mask_dir: Optional[str], image_size: int = 448, dummy_text: Optional[str] = None):
        super().__init__()
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.image_size = int(image_size)
        self.dummy_text = dummy_text

        img_paths = []
        for root, _, files in os.walk(img_dir):
            for fn in files:
                ext = os.path.splitext(fn)[1].lower()
                if ext in _IMG_EXTS:
                    img_paths.append(os.path.join(root, fn))
        self.img_paths = sorted(img_paths)
        if len(self.img_paths) == 0:
            raise FileNotFoundError(f"No images found under: {img_dir}")

        self.tf = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(_CLIP_MEAN, _CLIP_STD),
        ])

    def __len__(self) -> int:
        return len(self.img_paths)

    def _try_load_mask(self, img_path: str, w: int, h: int) -> Optional[np.ndarray]:
        if self.mask_dir is None:
            return None
        base = os.path.splitext(os.path.basename(img_path))[0]
        for ext in [".npy", ".png", ".jpg", ".jpeg", ".bmp", ".webp"]:
            cand = os.path.join(self.mask_dir, base + ext)
            if os.path.exists(cand):
                if ext == ".npy":
                    arr = np.load(cand)
                    if arr.ndim == 3:
                        arr = arr[..., 0]
                    arr = arr.astype(np.float32)
                    pil = Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8))
                else:
                    pil = Image.open(cand).convert("L")
                pil = pil.resize((w, h), Image.BILINEAR)
                arr = np.asarray(pil, dtype=np.float32) / 255.0
                return arr
        return None

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        p = self.img_paths[idx]
        img = Image.open(p).convert("RGB")
        img_t = self.tf(img)

        H = W = self.image_size
        m = self._try_load_mask(p, W, H)
        if m is None:
            yy, xx = np.meshgrid(np.linspace(-1, 1, H), np.linspace(-1, 1, W), indexing="ij")
            d2 = xx**2 + yy**2
            m = np.exp(-3.0 * d2).astype(np.float32)

        m_t = torch.from_numpy(m)[None, ...]
        text = (self.dummy_text if (self.dummy_text and len(self.dummy_text) > 0)
                else f"refer to the target region in {os.path.splitext(os.path.basename(p))[0]}")
        return {"image": img_t, "soft_mask": m_t, "text": text, "path": p}

# =========================================================
# collate（可変サイズを右下ゼロパディング）
# =========================================================

def _pad_to_hw(x: torch.Tensor, target_hw):
    Ht, Wt = target_hw
    if x.ndim != 3:
        raise ValueError(f"expect (C,H,W), got {x.shape}")
    C, H, W = x.shape
    pad_h = Ht - H
    pad_w = Wt - W
    if pad_h < 0 or pad_w < 0:
        raise ValueError(f"target smaller than tensor: target {(Ht,Wt)} vs {(H,W)}")
    return F.pad(x, (0, pad_w, 0, pad_h), value=0.0)

def refcoco_collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    PATCH = 14  # ViT-*/14 前提
    out: Dict[str, Any] = {}
    if len(batch) == 0:
        return out

    Hmax, Wmax = 0, 0
    for b in batch:
        img = b.get("image", b.get("images", None))
        if img is None: continue
        _, H, W = img.shape
        Hmax = max(Hmax, H); Wmax = max(Wmax, W)

    def _ceil_to_m(x, m): return ((x + m - 1) // m) * m
    Hmax = _ceil_to_m(Hmax, PATCH)
    Wmax = _ceil_to_m(Wmax, PATCH)

    if Hmax == 0 or Wmax == 0:
        raise RuntimeError("collate: no images found in batch")

    tensor_keys = []
    for k in batch[0].keys():
        if k == "bbox":
            continue
        if torch.is_tensor(batch[0][k]):
            tensor_keys.append(k)
    if "images" in batch[0] and "image" not in batch[0]:
        tensor_keys.append("images")

    for k in tensor_keys:
        buf = []
        for b in batch:
            if k not in b:
                raise KeyError(f"collate: key '{k}' missing in a sample")
            t = b[k]
            if t.ndim == 3:
                buf.append(_pad_to_hw(t.contiguous(), (Hmax, Wmax)))
            elif t.ndim == 4 and t.shape[0] == 1:
                buf.append(_pad_to_hw(t.squeeze(0).contiguous(), (Hmax, Wmax)))
            else:
                raise ValueError(f"collate: unexpected tensor shape for key '{k}': {t.shape}")
        out[k] = torch.stack(buf, dim=0)

    for k in batch[0].keys():
        if k in tensor_keys or k == "bbox":
            continue
        vals = [b.get(k, None) for b in batch]
        out[k] = vals

    if "image" not in out and "images" in out:
        out["image"] = out.pop("images")

    B, _, Ht, Wt = out["image"].shape
    if "soft_mask" not in out:
        out["soft_mask"] = torch.zeros(B, 1, Ht, Wt, dtype=out["image"].dtype)

    has_any_bbox = any(("bbox" in b) and isinstance(b["bbox"], torch.Tensor) for b in batch)
    if has_any_bbox:
        buf = []
        dtype = out["image"].dtype
        for b in batch:
            t = b.get("bbox", None)
            if t is None:
                buf.append(torch.zeros(4, dtype=dtype))
            else:
                t = t.to(dtype=torch.float32).view(-1)
                if t.numel() != 4:
                    raise ValueError(f"collate: bbox must have 4 elements, got {t.numel()}")
                buf.append(t)
        out["bbox"] = torch.stack(buf, dim=0)

    return out

# =========================================================
# DataLoader 構築
# =========================================================

def _ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def _mask_to_feat_hw(sm: torch.Tensor, feat_hw: Tuple[int, int]) -> torch.Tensor:
    Ht, Wt = int(feat_hw[0]), int(feat_hw[1])
    out = F.interpolate(sm, size=(Ht, Wt), mode="bilinear", align_corners=False)
    return out.clamp(0, 1)

def _ensure_list_text(texts: Any, B: int, fallback: str = "refer to the target region") -> List[str]:
    if isinstance(texts, list) and len(texts) == B:
        return texts
    if isinstance(texts, str):
        return [texts for _ in range(B)]
    return [fallback for _ in range(B)]

def build_loaders(cfg: Dict[str, Any]) -> Tuple[DataLoader, DataLoader]:
    dcfg = cfg.get("dataset", {})
    RefCOCOCls = None
    import_exc = None

    try:
        import importlib
        mod = importlib.import_module("datasets.refcoco.refcoco_dataset_inst")
        for cand in ["RefCOCODataset", "RefCOCO"]:
            if hasattr(mod, cand):
                RefCOCOCls = getattr(mod, cand)
                break
        if RefCOCOCls is None:
            raise AttributeError("datasets.refcoco.refcoco_dataset に RefCOCODataset/RefCOCO が見つかりません。")
    except Exception as e:
        import_exc = e
        print("[build_loaders] RefCOCO* class import failed -> fallback to image-folder dataset.")
        print("  reason:", repr(e))

    if RefCOCOCls is not None:
        train_json = dcfg.get("json", dcfg.get("train_json"))
        val_json   = dcfg.get("val_json", None)
        if train_json is None:
            raise KeyError("dataset.train_json（または dataset.json）を指定してください。")

        img_root      = dcfg.get("img_root")
        val_img_root  = dcfg.get("val_img_root", img_root)
        mask_root     = dcfg.get("mask_root", None)
        val_mask_root = dcfg.get("val_mask_root", mask_root)
        if img_root is None:
            raise KeyError("dataset.img_root を指定してください。")

        long_side   = int(dcfg.get("long_side", 896))
        supervision = dcfg.get("supervision", "mask")
        soft_cfg    = dcfg.get("soft", None)
        augment     = dcfg.get("augment", None)

        ds_tr = RefCOCOCls(
            json_path=train_json,
            img_root=img_root,
            mask_root=mask_root,
            split=dcfg.get("split", "train"),
            long_side=long_side,
            supervision=supervision,
            soft_cfg=soft_cfg,
            augment=augment,
            text_key=dcfg.get("text_key", "text"),
            answer_key=dcfg.get("answer_key", "answer"),
            use_answers=dcfg.get("use_answers", "expand"),
            sample_ratios=dcfg.get("sample_ratios", None),
        )

        _val_mode = str(dcfg.get("val_answer_mode", "first")).lower().strip()
        _val_use_answers = "first" if _val_mode == "first" else dcfg.get("use_answers", "expand")

        ds_va = RefCOCOCls(
            json_path=val_json if val_json is not None else train_json,
            img_root=val_img_root,
            mask_root=val_mask_root,
            split="val",
            long_side=long_side,
            supervision=supervision,
            soft_cfg=soft_cfg,
            augment=augment,
            text_key=dcfg.get("text_key", "text"),
            answer_key=dcfg.get("answer_key", "answer"),
            use_answers=_val_use_answers,
            sample_ratios=dcfg.get("sample_ratios", None),
        )

        bs  = int(cfg.get("batch_size", cfg.get("train", {}).get("batch_size", 4)))
        nw  = int(cfg.get("num_workers", 0))
        pin = bool(cfg.get("pin_memory", True))
        dl_tr = DataLoader(
            ds_tr, batch_size=bs, shuffle=True,
            num_workers=nw, pin_memory=pin, drop_last=True,
            collate_fn=refcoco_collate, persistent_workers=False
        )
        dl_va = DataLoader(
            ds_va, batch_size=bs, shuffle=False,
            num_workers=nw, pin_memory=pin, drop_last=False,
            collate_fn=refcoco_collate, persistent_workers=False
        )
        return dl_tr, dl_va

    # -------- フォールバック（画像フォルダ） --------
    tr_dir = cfg.get("train_dir") or cfg.get("data", {}).get("train_dir") or dcfg.get("img_root")
    va_dir = cfg.get("val_dir")   or cfg.get("data", {}).get("val_dir")   or dcfg.get("val_img_root", dcfg.get("img_root"))

    if not tr_dir or not va_dir:
        raise KeyError(
            "cfg['train_dir'] と cfg['val_dir'] を指定してください。"
            " もしくは datasets/refcoco/refcoco_dataset.py が import 可能か確認してください。"
            f" import_reason={repr(import_exc)}"
        )

    tr_mask = cfg.get("train_mask_dir") or cfg.get("data", {}).get("train_mask_dir") or dcfg.get("mask_root")
    va_mask = cfg.get("val_mask_dir")   or cfg.get("data", {}).get("val_mask_dir")   or dcfg.get("val_mask_root", dcfg.get("mask_root"))

    bs  = int(cfg.get("batch_size", 4))
    nw  = int(cfg.get("num_workers", 0))
    ims = int(cfg.get("image_size", 448))
    pin = bool(cfg.get("pin_memory", True))
    dummy_text = cfg.get("dummy_text", "refer to the target region")

    ds_tr = _SimpleRefCOCODataset(tr_dir, tr_mask, image_size=ims, dummy_text=dummy_text)
    ds_va = _SimpleRefCOCODataset(va_dir, va_mask, image_size=ims, dummy_text=dummy_text)

    dl_tr = DataLoader(ds_tr, batch_size=bs, shuffle=True,  num_workers=nw, pin_memory=pin, drop_last=True,  collate_fn=refcoco_collate)
    dl_va = DataLoader(ds_va, batch_size=bs, shuffle=False, num_workers=nw, pin_memory=pin, drop_last=False, collate_fn=refcoco_collate)
    return dl_tr, dl_va

# === 超簡易・凍結ビジョンエンコーダ（フォールバック用） ===
class TinyFrozenConvVision(nn.Module):
    def __init__(self, out_dim: int = 768, patch: int = 16):
        super().__init__()
        self.out_dim = int(out_dim)
        self.patch = int(patch)
        self.conv = nn.Conv2d(3, self.out_dim, kernel_size=self.patch, stride=self.patch, padding=0, bias=False)
        with torch.no_grad():
            nn.init.normal_(self.conv.weight, mean=0.0, std=0.02)
        for p in self.parameters():
            p.requires_grad = False
    def forward(self, x: torch.Tensor):
        f = self.conv(x)
        B, C, Ht, Wt = f.shape
        tokens = f.flatten(2).transpose(1, 2).contiguous()
        return {"tokens": tokens, "feat_hw": (Ht, Wt)}

# === モデル構築 ===

def build_model(cfg: Dict[str, Any], device: torch.device) -> nn.Module:
    mcfg = cfg.get("model", {})
    proj_dim_in = int(mcfg.get("proj_dim_in", 768))

    vision = None
    try:
        oc_name = mcfg.get("openclip_name", "ViT-g-14")
        oc_pt   = mcfg.get("openclip_pretrained", "laion2b_s34b_b88k")
        print(f"[info] Using OpenCLIP {oc_name} ({oc_pt}).")
        vision = OpenCLIPVisionFrozen(model_name=oc_name, pretrained=oc_pt).to(device).eval()
    except Exception as e:
        print("[warn] OpenCLIPViTL14Frozen init failed -> fallback to TinyFrozenConvVision")
        print("  reason:", repr(e))
        vision = TinyFrozenConvVision(out_dim=proj_dim_in, patch=int(mcfg.get("patch", 16))).to(device).eval()

    enc_kv_dim = int(mcfg.get("encoder_hidden_size", 1408))

    model = QFormerRefCOCO(
        vision_encoder=vision,
        qformer=None,
        llm_name=mcfg.get("llm_name", "lmsys/vicuna-7b-v1.5"),
        num_queries=int(mcfg.get("num_queries", 16)),
        proj_dim_in=proj_dim_in,
        proj_dim_out=int(mcfg.get("proj_dim_out", 4096)),
        max_txt_len=int(mcfg.get("max_txt_len", 64)),
        load_vicuna=bool(mcfg.get("load_vicuna", False)),
        pretrained_qformer=mcfg.get("pretrained_qformer", None),
        text_cond_mode=str(mcfg.get("text_cond_mode", "bias")),
        num_text_kv=int(mcfg.get("num_text_kv", 0)),
        qformer_kv_dim=enc_kv_dim,
    ).to(device)
    try:
        if hasattr(model, "qformer") and hasattr(model.qformer, "bert"):
            model.qformer.bert.gradient_checkpointing_enable()
            print("[info] Q-Former gradient checkpointing: enabled")
    except Exception as e:
        print("[warn] gradient checkpointing not enabled:", repr(e))
    return model

# === Optimizer（層別LR）===

def build_optimizer(model: nn.Module, cfg: Dict[str, Any]) -> torch.optim.Optimizer:
    def _get_last_layer_idx(m):
        try:
            return len(m.qformer.bert.encoder.layer) - 1
        except Exception:
            try:
                return len(m.qformer.encoder.layer) - 1
            except Exception:
                return None

    last_idx = _get_last_layer_idx(model)
    target_layers = {last_idx, max(0, last_idx-1)}  # 最終&一個手前

    lr_head = float(cfg.get("lr_head", 1e-3))
    lr_qkv  = float(cfg.get("lr_q_kv", 2e-4))
    lr_last = float(cfg.get("lr_q_last", 5e-4))
    wd_head = float(cfg.get("wd_head", 0.0))
    wd_body = float(cfg.get("wd_body", 0.01))

    pg_head, pg_q_kv, pg_q_last, pg_vision = [], [], [], []

    # いったん全凍結
    for _, p in model.named_parameters():
        p.requires_grad = False
        
    # ---Vision: 後段ブロックだけ開放 ---
    try:
        vis = getattr(model, "vision", None)
        blocks = getattr(getattr(vis, "model", vis).visual, "transformer", None).resblocks
        N = int(cfg.get("vision_unfreeze_blocks", 3))
        N = max(1, min(N, len(blocks)))
        for blk in blocks[-N:]:
            for n, p in blk.named_parameters():
                p.requires_grad = True
                pg_vision.append(p)
        visual = getattr(getattr(vis, "model", vis), "visual", None)
        pe = getattr(visual, "positional_embedding", None)
        if isinstance(pe, torch.nn.Parameter):
            pe.requires_grad = True
            pg_vision.append(pe)
        ln_pre = getattr(visual, "ln_pre", None)
        if isinstance(ln_pre, nn.LayerNorm):
            for p in ln_pre.parameters():
                p.requires_grad = True
                pg_vision.append(p)
    except Exception as e:
        print("[warn] vision unfreeze skipped:", repr(e))

    for n, p in model.named_parameters():
        if any(k in n for k in ["projector", "projector_text", "token_proj", "query_gate", "logit_tau", "logit_bias",
                                "vis_proj", "query_embed", "pre_q_ln",
                                "_vis_adapter", "sim_proj", "text_kv_proj"]):
            p.requires_grad = True
            pg_head.append(p)
            continue

        if any(f".layer.{i}." in n for i in target_layers):
            if ".crossattention." in n and (".key." in n or ".value." in n):
                p.requires_grad = True
                pg_q_kv.append(p)
                continue

        if (f".layer.{last_idx}." in n) and (".attention.self." in n) and (".query." in n or ".output.dense." in n):
            p.requires_grad = True
            pg_q_last.append(p)
            continue

        if ("qformer.bert.encoder.layer" in n) or ("qformer.encoder.layer" in n):
            if (last_idx is not None) and (f".layer.{last_idx}." in n):
                p.requires_grad = True
                pg_q_last.append(p)
                continue
            if (".crossattention." in n) and (".key." in n or ".value." in n):
                p.requires_grad = True
                pg_q_kv.append(p)
                continue

    lr_vision = float(cfg.get("lr_vision", 5e-6))
    optim = torch.optim.AdamW(
        [
            {"params": pg_q_kv,    "lr": lr_qkv,    "weight_decay": wd_body},
            {"params": pg_q_last,  "lr": lr_last,   "weight_decay": wd_body},
            {"params": pg_head,    "lr": lr_head,   "weight_decay": wd_head},
            {"params": pg_vision,  "lr": lr_vision, "weight_decay": wd_body},
        ],
        betas=(0.9, 0.999),
    )

    num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("[sanity] trainable params(after freeze):", num_trainable)
    print(f"[sanity] pg_head={len(pg_head)} | pg_q_last={len(pg_q_last)} | pg_q_kv={len(pg_q_kv)} | pg_vision={len(pg_vision)}")
    return optim

# ---- 安全CSV追記 ----
def _csv_append(path: str, row: dict, field_order: list):
    is_new = not os.path.exists(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=field_order)
        if is_new:
            writer.writeheader()
        safe_row = {k: row.get(k, None) for k in field_order}
        writer.writerow(safe_row)

def _jsonl_append(path: str, row: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

# =========================================================
# Warmup → Cosine スケジューラ
# =========================================================

def build_warmup_cosine(optimizer, warmup_steps: int, total_steps: int):
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

def build_sgdr(optimizer, t0: int = 2, t_mult: int = 2, eta_min: float = 1e-6):
    return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=t0, T_mult=t_mult, eta_min=eta_min
    )

# =========================================================
# seed 固定
# =========================================================
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

# =========================================================
# （追加）再開ユーティリティ
# =========================================================
def _maybe_resume(model, optim, scheduler, scaler, ema, cfg) -> int:
    """returns global_step after resume (0 if fresh)"""
    resume_path = cfg.get("resume", None)
    resume_full = bool(cfg.get("resume_full", False))
    global_step = 0
    if not resume_path:
        return 0

    if resume_path.endswith(".safetensors"):
        from safetensors.torch import load_file as safe_load
        sd = safe_load(resume_path)  # CPU tensors
        print(f"[resume] safetensors loaded on CPU: {resume_path}")
        missing, unexpected = model.load_state_dict(sd, strict=False)
        if missing:   print("[resume] missing keys:", len(missing))
        if unexpected:print("[resume] unexpected keys:", len(unexpected))
        return 0

    ckpt = torch.load(resume_path, map_location="cpu", weights_only=False)
    sd = ckpt.get("model", ckpt)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"[resume] torch.load (cpu) loaded: {resume_path}")
    if missing:   print("[resume] missing keys:", len(missing))
    if unexpected:print("[resume] unexpected keys:", len(unexpected))

    if resume_full and isinstance(ckpt, dict):
        if "optimizer" in ckpt and ckpt["optimizer"] is not None:
            try: optim.load_state_dict(ckpt["optimizer"])
            except Exception as e: print("[resume] optimizer.load_state_dict failed:", repr(e))
        if "scheduler" in ckpt and ckpt["scheduler"] is not None:
            try: scheduler.load_state_dict(ckpt["scheduler"])
            except Exception as e: print("[resume] scheduler.load_state_dict failed:", repr(e))
        if "scaler" in ckpt and ckpt["scaler"] is not None and scaler is not None:
            try: scaler.load_state_dict(ckpt["scaler"])
            except Exception as e: print("[resume] scaler.load_state_dict failed:", repr(e))
        if "ema" in ckpt and isinstance(ckpt["ema"], dict) and (ema is not None):
            ema.ema = {k: (v.to("cpu") if torch.is_tensor(v) else v) for k, v in ckpt["ema"].items()}
        global_step = int(ckpt.get("global_step", 0))
        print("[resume] optimizer/scheduler/scaler/ema restored; global_step =", global_step)

    return global_step

# =========================================================
# スイッチ（重みON/OFF切替ランプ）
# =========================================================
def _ramp(step: int, on_step: Optional[int], ramp_steps: int, start: float, target: float) -> float:
    if on_step is None:
        return target
    if step < on_step:
        return start
    if ramp_steps <= 0:
        return target
    t = min(1.0, max(0.0, (step - on_step) / float(max(1, ramp_steps))))
    return start + (target - start) * t

def _gate_from_cfg(step: int, sw: Optional[dict], default: float = 1.0) -> float:
    if not isinstance(sw, dict):
        return default
    on_step = sw.get("on_step", None)
    ramp = int(sw.get("ramp_steps", 0))
    start = float(sw.get("start", default))
    target = float(sw.get("target", default))
    return _ramp(step, on_step, ramp, start, target)
# =========================================================
# 学習本体
# =========================================================

def train(cfg: Dict[str, Any]):
    torch.backends.cudnn.benchmark = True
    set_seed(int(cfg.get("seed", 42)))

    dbg_cfg = cfg.get("debug", {})
    sanity_steps = int(dbg_cfg.get("sanity_steps", 0))
    sanity_every = int(dbg_cfg.get("sanity_every", 0))
    skip_val = bool(dbg_cfg.get("skip_val", False))

    # out_dir
    out_dir = cfg.get("out_dir", cfg.get("train", {}).get("out_dir", "./outputs_refcoco"))
    os.makedirs(os.path.join(out_dir, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "attn_vis"), exist_ok=True)

    from torch.utils.tensorboard import SummaryWriter
    tb_dir = os.path.join(out_dir, "tb")
    writer = SummaryWriter(log_dir=tb_dir)

    log_dir = os.path.join(out_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    train_csv = os.path.join(log_dir, "train_iter.csv")
    epoch_csv = os.path.join(log_dir, "epoch_val.csv")
    epoch_jsonl = os.path.join(log_dir, "epoch_val.jsonl")
    # ---- timing (wall clock) ----
    wall0 = time.perf_counter()
    total_iter_ema_sec, total_iter_ema_ips = None, None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data
    dl_tr, dl_va = build_loaders(cfg)
    print("[debug] cfg.train.max_steps = ", cfg.get("train", {}).get("max_steps"))
    print("[debug] cfg.train.max_steps_per_epoch =", cfg.get("train", {}).get("max_steps_per_epoch"))
    print("[debug] len(dl_tr) =", len(dl_tr), "| len(dl_va) =", len(dl_va))

    # 入力解像度（ViT/14 のパッチ解像に直結）を YAML に合わせて統一
    long_side = int(cfg.get("dataset", {}).get("long_side", 896))

    # Model
    model = build_model(cfg, device)

    # Vision/LLM の確実な凍結（名前に依存しない）
    if hasattr(model, "vision_encoder") and (model.vision_encoder is not None):
        for p in model.vision_encoder.parameters():
            p.requires_grad = False
    if hasattr(model, "vision") and (model.vision is not None):  # 実装差異の保険
        for p in model.vision.parameters():
            p.requires_grad = False

    # Optimizer / Scheduler
    optim = build_optimizer(model, cfg)
    num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    groups = [(len(g['params']), g.get('lr', None)) for g in optim.param_groups]
    print("[sanity] trainable params:", num_trainable)
    for i,(n,lr) in enumerate(groups): print(f"[sanity] pg{i}: n={n}, lr={lr}")

    # AMP（bf16/fp16 切替）
    from torch import amp
    tc = cfg.get("train", {})
    use_amp  = bool(cfg.get("use_amp", tc.get("amp", True)))
    use_fp16 = bool(cfg.get("use_fp16", False))
    device_type = "cuda" if torch.cuda.is_available() else "cpu"
    ac_dtype = torch.float16 if (device_type == "cuda" and use_fp16) else (torch.bfloat16 if device_type=="cuda" else torch.float32)
    scaler = amp.GradScaler(enabled=use_amp and (device_type == "cuda") and use_fp16)

    # エポックなど
    epochs = int(cfg.get("epochs", tc.get("epochs", tc.get("num_epochs", 3))))
    log_interval = int(cfg.get("log_interval", tc.get("log_interval", 50)))

    # --- max_steps（0 なら無効）---
    max_steps = int(cfg.get("train", {}).get("max_steps", 0))
    len_tr = max(1, len(dl_tr))

    if max_steps > 0 and epochs * len_tr < max_steps:
        prev_epochs = epochs
        epochs = math.ceil(max_steps / len_tr)
        print(
            f"[train] epochs adjusted: {prov_epoches} -> {epochs} "
            f"to satisfy max_steps={max_steps} (len_tr={len_tr})"
        )
    # --- per-epoch 上限（0 なら無効）---
    max_steps_per_epoch = int(cfg.get("train", {}).get("max_steps_per_epoch", 0))

    # 損失重み
    lw = cfg.get("loss_weights", {"attn":0.5, "box":1.5, "contrast":0.0, "lm":0.0})
    temperature = float(cfg.get("contrast", {}).get("temperature", 0.07))
    lm_warmup = int(cfg.get("lm_warmup_steps", 2000))

    # 補助ロス重み
    aux = cfg.get("aux_loss", {})
    soft_iou_w = float(aux.get("soft_iou_w", 0.0))
    edge_bce_w = float(aux.get("edge_bce_w", 0.0))
    tv_w       = float(aux.get("tv_w", 0.0))

    # MixUp
    mixup_prob   = float(cfg.get("mixup_prob", 0.0))
    mixup_alpha  = float(cfg.get("mixup_alpha", 0.2))

    # 評価設定
    eval_q1 = float(cfg.get("eval", {}).get("quantile_p", 0.85))
    iou_col = f"iou_p{int(round(eval_q1*100))}"

    best_val = float("inf")
    best_val_iou = -1.0

    info_nce = InfoNCELoss(temperature=temperature)

    # ウォームアップ（Box重み用）
    warm_steps = int(cfg.get("box_warmup_steps", 1000))
    box_target = float(cfg.get("box_target_weight", 0.8))

    # LR Scheduler
    total_steps = epochs * len_tr
    if max_steps > 0:
        total_steps = min(total_steps, max_steps)
    lr_warmup_steps = int(cfg.get("lr_warmup_steps", 1000))
    lr_warmup_steps = min(lr_warmup_steps, max(1, total_steps // 2))
    sched_name = str(cfg.get("train", {}).get("scheduler", "cosine")).lower()
    if sched_name == "sgdr":
        sgdr_cfg = cfg.get("sgdr", {})
        t0 = int(sgdr_cfg.get("t0", 2))
        t_mult = int(sgdr_cfg.get("t_mult", 2))
        eta_min = float(sgdr_cfg.get("eta_min", 1e-6))
        scheduler = build_sgdr(optim, t0=t0, t_mult=t_mult, eta_min=eta_min)
    elif sched_name == "constant":
        scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=lambda step: 1.0)
    else:
        scheduler = build_warmup_cosine(optim, lr_warmup_steps, total_steps)

    console_compact = bool(cfg.get("train", {}).get("console_compact", True))

    # EMA（CPU/学習対象のみ）
    ema_decay = cfg.get("ema_decay", 0.999)
    if (ema_decay is None) or (isinstance(ema_decay, (int, float)) and float(ema_decay) <= 0):
        ema = None
        print("[ema] disabled")
    else:
        ema = ModelEMA(model, decay=float(ema_decay), cpu=True, trainable_only=True)
        print(f"[ema] enabled (decay={float(ema_decay)})")

    # 再開
    global_step = _maybe_resume(model, optim, scheduler, scaler, ema, cfg)

    overfit_1b = bool(cfg.get("debug", {}).get("overfit_one_batch", False))
    fixed_batch = None
    printed_hw = False

    early_cfg = cfg.get("early_stop", {})
    early_metric = str(early_cfg.get("metric", "val_iou_p"))
    early_patience = int(early_cfg.get("patience", 0))
    early_best = -1e18
    early_wait = 0

    reached_max = False  # max_steps 到達フラグ

    # ========= TRAIN =========
    for ep in range(1, epochs + 1):
        model.train()
        t0 = time.time()
        running = 0.0
        iter_ema_sec, iter_ema_ips = None, None

        for it, batch in enumerate(dl_tr, start=1):
            if it in (1, 50, 100, 101):
                print(f"[debug: ep={ep} it={it} gs={global_step} |"
                      f"path0={batch.get('path', [' '])[0]}")

            iter_t0 = time.perf_counter()
            if overfit_1b:
                if fixed_batch is None:
                    fixed_batch = batch
                else:
                    batch = fixed_batch

            images = batch.get("image", batch.get("images")).to(device, non_blocking=True)

            soft = batch.get("soft_mask", batch.get("mask", None))
            if soft is None:
                soft = torch.zeros(images.size(0), 1, images.shape[2], images.shape[3], device=images.device, dtype=torch.float32)
            else:
                soft = soft.to(device=device, dtype=torch.float32)
                if soft.ndim == 3:
                    soft = soft.unsqueeze(1)

            texts = _ensure_list_text(batch.get("text", None), images.size(0), fallback="refer to the target region")
            paths = batch.get("path", [""] * images.size(0))

            # MixUp
            if mixup_prob > 0.0:
                images, soft = maybe_mixup(images, soft, alpha=mixup_alpha, prob=mixup_prob)

            optim.zero_grad(set_to_none=True)

            # ===== forward =====
            from torch import amp as _amp
            with _amp.autocast(device_type=device_type, dtype=ac_dtype, enabled=use_amp):
                do_lm = (lw.get("lm", 0.0) > 0.0)
                out = model({"image": images, "text": texts}, compute_lm=do_lm)

                # ---- アテンションのロジット/確率取り出し ----
                if "attn_logits" in out and out["attn_logits"] is not None and out["attn_logits"].ndim == 4:
                    A_logit_hw = out["attn_logits"].to(torch.float32)
                    Ht, Wt = A_logit_hw.shape[-2:]
                elif "attn_maps" in out and out["attn_maps"] is not None:
                    A_prob_hw = torch.nan_to_num(out["attn_maps"], nan=0.0, posinf=1.0, neginf=0.0).clamp(1e-6, 1-1e-6)
                    Ht, Wt = A_prob_hw.shape[-2:]
                    A_logit_hw = torch.logit(A_prob_hw)
                else:
                    raise RuntimeError(f"Model outputs do not contain attn logits/maps. keys={list(out.keys())}")

                if not printed_hw:
                    print(f"[sanity] attn_feat_hw = {Ht}x{Wt}  (patch=14; depends on input size)")
                    printed_hw = True

                if not torch.isfinite(A_logit_hw).all():
                    print("[nan-guard] non-finite logits -> skip step")
                    if use_amp: scaler.update()
                    optim.zero_grad(set_to_none=True)
                    continue
                sm_hw = _mask_to_feat_hw(soft, (Ht, Wt)).clamp(0.0, 1.0)

                # ===== 1) L_attn =====
                with torch.no_grad():
                    pos_frac = sm_hw.mean(dim=(2, 3), keepdim=True).clamp(1e-6, 1-1e-6)
                    pos_weight = ((1.0 - pos_frac) / pos_frac).clamp(1.0, 10.0)
                bce_per_pix   = F.binary_cross_entropy_with_logits(A_logit_hw, sm_hw, reduction="none", pos_weight=pos_weight)
                loss_attn_val = zero_if_not_finite(bce_per_pix.mean())

                # ===== 2) L_box =====
                A_prob_hw  = torch.sigmoid(A_logit_hw).to(torch.float32)
                S32        = sm_hw.to(torch.float32)
                S_blur     = F.avg_pool2d(S32, kernel_size=3, stride=1, padding=1)
                S_soft     = (0.7 * S32 + 0.3 * S_blur).clamp(0.0, 1.0)
                loss_box_dice = safe_dice_from_probs(A_prob_hw, S_soft)
                loss_box_bce  = safe_bce_from_probs(A_prob_hw, S_soft)
                loss_box_val  = 1.5 * loss_box_dice + 0.5 * loss_box_bce

                # ===== 2.5) Soft-IoU =====
                loss_iou = soft_iou_loss(A_prob_hw, S_soft) if soft_iou_w > 0 else torch.tensor(0.0, device=images.device)

                # ===== 3) L_contrast =====
                ve = out.get("vision_emb", None)
                te = out.get("text_emb", None)
                if isinstance(ve, torch.Tensor): ve = ve.float().mean(dim=1)
                if isinstance(te, torch.Tensor): te = te.float()
                if (isinstance(ve, torch.Tensor) and isinstance(te, torch.Tensor) and cfg.get("loss_weights", {}).get("contrast", 0.0) > 0.0):
                    loss_ctr_val = zero_if_not_finite(info_nce(ve, te))
                else:
                    loss_ctr_val = torch.tensor(0.0, device=images.device, dtype=torch.float32)

                # ===== 4) L_lm =====
                loss_lm_val = out.get("lm_loss_raw", None)
                if not isinstance(loss_lm_val, torch.Tensor) or cfg.get("loss_weights", {}).get("lm", 0.0) == 0.0:
                    loss_lm_val = torch.tensor(0.0, device=images.device, dtype=torch.float32)
                else:
                    loss_lm_val = zero_if_not_finite(loss_lm_val)

                # ===== 5) focal 風 =====
                with torch.no_grad():
                    p = torch.sigmoid(A_logit_hw)
                    focal_w = (1 - p) * S_soft + p * (1 - S_soft)
                focal_loss = F.binary_cross_entropy_with_logits(A_logit_hw, S_soft, weight=focal_w, reduction="mean")

                # ===== 6) 境界BCE + TV =====
                ew = edge_weight_map(S_soft)
                bce_edge = F.binary_cross_entropy_with_logits(A_logit_hw, S_soft, weight=(1.0 + 2.0 * ew), reduction="mean")
                tv_val   = total_variation(A_prob_hw)

                # ------- 係数（box warmup + スイッチゲート） -------
                t_w   = min(1.0, (global_step) / max(1, warm_steps))
                w_box_base = box_target * t_w  # 0 -> box_target
                # switch（任意）
                switch = cfg.get("switch", {})
                attn_gate = _gate_from_cfg(global_step, switch.get("attn"), default=1.0)
                box_gate  = _gate_from_cfg(global_step, switch.get("box"),  default=1.0)

                # 動的係数
                focal_cfg         = cfg.get("focal", {})
                focal_w0          = float(focal_cfg.get("w0", 0.30))
                focal_w1          = float(focal_cfg.get("w1", 0.10))
                focal_decay_steps = int(focal_cfg.get("decay_steps", 150))
                alpha             = 1.0 - min(1.0, (global_step) / max(1, focal_decay_steps))
                focal_coeff       = focal_w1 + (focal_w0 - focal_w1) * alpha

                aux        = cfg.get("aux_loss", {})
                soft_iou_w = float(aux.get("soft_iou_w", 0.0))
                edge_bce_w = float(aux.get("edge_bce_w", 0.0))
                tv_w       = float(aux.get("tv_w", 0.0))

                ctr_cfg    = cfg.get("contrast", {})
                ctr_w0     = float(ctr_cfg.get("w0", lw.get("contrast", 0.0)))
                ctr_w1     = float(ctr_cfg.get("w1", lw.get("contrast", 0.0)))
                ctr_warmup = int(ctr_cfg.get("warmup_steps", 1000))
                ctr_ramp   = int(ctr_cfg.get("ramp_steps", 2000))
                ctr_prog   = 0.0
                if (ctr_w1 > ctr_w0):
                    ctr_prog = min(1.0, max(0.0, ((global_step) - ctr_warmup) / max(1, ctr_ramp)))
                ctr_coeff  = ctr_w0 + (ctr_w1 - ctr_w0) * ctr_prog

                lovasz_w_cfg = float(cfg.get("aux_loss", {}).get("lovasz_w", 0.0))
                lovasz_w_eff = lovasz_w_cfg if global_step >= lr_warmup_steps else 0.0
                loss_lovasz = lovasz_sigmoid(A_prob_hw, S_soft) if lovasz_w_eff > 0 else torch.tensor(0.0, device=images.device)

                # ------- 合算ロス -------
                attn_coeff = lw.get("attn", 0.5) * attn_gate
                box_coeff  = lw.get("box", 1.5) * w_box_base * box_gate
                lm_scale  = (min(1.0, (global_step) / max(1, lm_warmup)) if lw.get("lm", 0.0) > 0 else 0.0)

                loss = (attn_coeff * loss_attn_val +
                        box_coeff  * loss_box_val +
                        ctr_coeff  * loss_ctr_val +
                        (lw.get("lm", 0.0) * lm_scale) * loss_lm_val +
                        focal_coeff * focal_loss +
                        soft_iou_w * loss_iou +
                        lovasz_w_eff * loss_lovasz +
                        edge_bce_w * bce_edge +
                        tv_w * tv_val)

                if not torch.isfinite(loss):
                    print("[warn] non-finite total loss; skip step")
                    if use_amp: scaler.update()
                    optim.zero_grad(set_to_none=True)
                    continue
            # ===== backward / step =====
            if use_amp:
                scaler.scale(loss).backward()
                scaler.unscale_(optim)
                bad_grad = False
                for p in model.parameters():
                    if p.grad is not None and not torch.isfinite(p.grad).all():
                        bad_grad = True
                        break
                if bad_grad:
                    print("[nan-guard] non-finite grad -> zero_grad & skip update")
                    optim.zero_grad(set_to_none=True)
                    scaler.update()
                    continue
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], max_norm=1.0)
                scaler.step(optim)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], max_norm=1.0)
                optim.step()

            # スケジューラ & EMA
            scheduler.step()
            if ema is not None:
                ema.update(model)

            running += float(loss.item())
            global_step += 1

            # ===== timing / throughput =====
            iter_sec = max(1e-9, time.perf_counter() - iter_t0)
            imgs = int(images.size(0))
            ips = imgs / iter_sec
            iter_ema_sec = iter_sec if iter_ema_sec is None else (0.9 * iter_ema_sec + 0.1 * iter_sec)
            iter_ema_ips = ips      if iter_ema_ips is None else (0.9 * iter_ema_ips + 0.1 * ips)
            total_iter_ema_sec = iter_sec if total_iter_ema_sec is None else (0.99 * total_iter_ema_sec + 0.01 * iter_sec)
            total_iter_ema_ips = ips      if total_iter_ema_ips is None else (0.99 * total_iter_ema_ips + 0.01 * ips)
            steps_left_epoch = max(0, len(dl_tr) - it)
            eta_epoch_sec = (iter_ema_sec or iter_sec) * steps_left_epoch
            total_steps_all = (max_steps if max_steps > 0 else (epochs * len_tr))
            steps_done = (ep - 1) * max(1, len(dl_tr)) + it
            eta_total_sec = (total_iter_ema_sec or iter_sec) * max(0, total_steps_all - steps_done)

            # ===== max_steps 到達チェック =====
            if max_steps > 0 and global_step >= max_steps:
                print(f"[debug] hit max_steps at gs={global_step} (cfg={max_steps}")
                print(f"[train] Reached max_steps={max_steps} (global_step={global_step}). Stopping training.")
                reached_max = True
                # ここではbreakし、下の保存や評価へ
                break

            # ===== 1エポック上限 =====
            if max_steps_per_epoch > 0 and it >= max_steps_per_epoch:
                print(f"[debug] hit max_steps_per_epoch at it={it} (cfg={max_steps_per_epoch})")
                print(f"[train] Reached max_steps_per_epoch={max_steps_per_epoch} at epoch {ep}.")
                break

            # ===== メトリクス =====
            with torch.no_grad():
                probs = A_prob_hw.detach().to(torch.float32)
                gt    = sm_hw.detach().to(torch.float32)

                B, _, Ht_, Wt_ = probs.shape
                flat = probs.view(B, -1)

                qs = [0.70, 0.80, 0.90]
                iou_qs = {}
                for q in qs:
                    thr = torch.quantile(flat, q=q, dim=1, keepdim=True).view(B,1,1,1)
                    pred = (probs >= thr).float()
                    inter = (pred * gt).sum(dim=(2, 3))
                    union = (pred + gt - pred * gt).sum(dim=(


2, 3)) + 1e-6
                    iou_qs[f"iou_p{int(q*100)}"] = (inter / union).mean().item()

                pred_05 = (probs >= 0.5).float()
                inter   = (pred_05 * gt).sum(dim=(2, 3))
                union   = (pred_05 + gt - pred_05 * gt).sum(dim=(2, 3)) + 1e-6
                iou_05  = (inter / union).mean().item()

                gt_bin = (gt >= 0.5).float()
                K = gt_bin.view(B, -1).sum(dim=1).to(torch.int64)
                K = torch.clamp(K, min=1, max=Ht_ * Wt_ - 1)
                pred_sized = torch.zeros_like(flat)
                for b in range(B):
                    kb = int(K[b].item())
                    if kb > 0:
                        topk_idx = torch.topk(flat[b], k=kb, dim=0).indices
                        pred_sized[b, topk_idx] = 1.0
                pred_sized = pred_sized.view(B, 1, Ht_, Wt_)
                inter   = (pred_sized * gt_bin).sum(dim=(2, 3))
                union   = (pred_sized + gt_bin - pred_sized * gt_bin).sum(dim=(2, 3)) + 1e-6
                iou_sz  = (inter / union).mean().item()

                pos_mean = float(probs[gt >= 0.5].mean().item()) if (gt >= 0.5).any() else float('nan')
                neg_mean = float(probs[gt < 0.5].mean().item()) if (gt < 0.5).any() else float('nan')

            if it % log_interval == 0:
                # 可視化
                if cfg.get("train", {}).get("save_attn_vis", True) and (it % (log_interval * 2) == 0):
                    try:
                        import torchvision.utils as vutils
                        vis_dir = os.path.join(out_dir, "attn_vis", f"ep{ep:03d}_it{it:06d}")
                        os.makedirs(vis_dir, exist_ok=True)
                        vutils.save_image(A_prob_hw.clamp(0, 1), os.path.join(vis_dir, "pred_prob.png"))
                        vutils.save_image(sm_hw.clamp(0, 1), os.path.join(vis_dir, "gt_mask.png"))
                    except Exception as e:
                        print("[warn] save_attn_vis failed:", repr(e))
                avg_loss = running / log_interval
                base_row = {
                    "epoch": ep, "iter": it, "global_step": global_step,
                    "loss": round(float(avg_loss), 6),
                    "attn": round(float(loss_attn_val.item()), 6),
                    "box":  round(float(loss_box_val.item()), 6),
                    "ctr":  round(float(loss_ctr_val.item()), 6),
                    "lm":   round(float(loss_lm_val.item()), 6),
                    "box_w": round(float(w_box_base), 6),
                    "iou_05": round(float(iou_05), 6),
                    "iou_size": round(float(iou_sz), 6),
                    "pos_mean": round(float(pos_mean), 6),
                    "neg_mean": round(float(neg_mean), 6),
                    "p_cur": round(float(eval_q1), 4),
                }
                for k, v in iou_qs.items():
                    base_row[k] = round(float(v), 6)

                writer.add_scalar("train/loss", float(avg_loss), global_step)
                writer.add_scalar("train/sec_per_iter", float(iter_ema_sec or iter_sec), global_step)
                writer.add_scalar("train/imgs_per_sec", float(iter_ema_ips or ips), global_step)
                writer.add_scalar("train/eta_epoch_min", float(eta_epoch_sec/60.0), global_step)
                writer.add_scalar("train/eta_total_min", float(eta_total_sec/60.0), global_step)
                writer.add_scalar("train/attn", float(loss_attn_val.item()), global_step)
                writer.add_scalar("train/box", float(loss_box_val.item()), global_step)
                writer.add_scalar("train/iou@p70", float(iou_qs["iou_p70"]), global_step)
                writer.add_scalar("train/iou@p80", float(iou_qs["iou_p80"]), global_step)
                writer.add_scalar("train/iou@p90", float(iou_qs["iou_p90"]), global_step)
                writer.add_scalar("train/iou@0.5", float(iou_05), global_step)
                writer.add_scalar("train/iou@size", float(iou_sz), global_step)
                try:
                    writer.add_scalar("lr", float(scheduler.get_last_lr()[0]), global_step)
                except Exception:
                    writer.add_scalar("lr", float(optim.param_groups[0]["lr"]), global_step)
                writer.add_scalar("train/gt_pos_ratio", float((sm_hw>0.5).float().mean().item()), global_step)

                row = {
                    **base_row,
                    "sec_per_iter": round(float(iter_ema_sec or iter_sec), 6),
                    "imgs_per_sec": round(float(iter_ema_ips or ips), 2),
                    "eta_epoch_min": round(float(eta_epoch_sec/60.0), 2),
                    "eta_total_min": round(float(eta_total_sec/60.0), 2),
                }
                _csv_append(
                    train_csv, row,
                    field_order=[
                        "epoch", "iter", "global_step", "loss", "attn", "box", "ctr", "lm", "box_w",
                        "iou_p70", "iou_p80", "iou_p90", "iou_05", "iou_size", "pos_mean", "neg_mean", "p_cur",
                        "sec_per_iter", "imgs_per_sec", "eta_epoch_min", "eta_total_min"
                    ]
                )

                running = 0.0

                # コンソール
                if console_compact:
                    if it == 1 or (it % max(50, log_interval*10) == 1):
                        print("epoch\titer\tgs\tloss\tiou@0.5\tpos\tneg\tlr\tips")
                    try:
                        cur_lr = float(scheduler.get_last_lr()[0])
                    except Exception:
                        cur_lr = optim.param_groups[0]["lr"]
                    print(f"{ep}\t{it}\t{global_step}\t"
                          f"{avg_loss:.4f}\t{iou_05:.3f}\t{pos_mean:.3f}\t{neg_mean:.3f}\t{cur_lr:.2e}\t{(iter_ema_ips or ips):.2f}")

            # ===== mid-epoch 保存（軽量） =====
            save_every_steps = int(cfg.get("save_every_steps", 0))
            if save_every_steps and (global_step % save_every_steps == 0):
                light_state = {
                    "model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                    "epoch": ep, "global_step": global_step
                }
                light_path = os.path.join(out_dir, "checkpoints", f"midstep_weights_{global_step}.pt")
                try:
                    atomic_torch_save(light_state, light_path, use_legacy=False)
                except Exception as e:
                    print("[warn] mid-save zip failed, retry legacy:", repr(e))
                    atomic_torch_save(light_state, light_path, use_legacy=True)
                if atomic_safe_save is not None:
                    st_path = os.path.join(out_dir, "checkpoints", f"midstep_weights_{global_step}.safetensors")
                    try:
                        atomic_safe_save({k: v.detach().cpu() for k, v in model.state_dict().items()}, st_path)
                    except Exception as e:
                        print("[warn] safetensors mid-save failed:", repr(e))

        # ========= VALIDATION/終了処理 =========
        if reached_max:
            # このエポックの保存は後段で実行
            pass

        if not skip_val:
            model.eval()
            backup = {k: (v.detach().to("cpu", copy=True) if torch.is_tensor(v) else v) for k, v in model.state_dict().items()}
            torch.cuda.empty_cache()

            if ema is not None:
                ema.copy_to(model)
            with torch.no_grad():
                total = 0.0
                count = 0
                iou_list, iou05_list, iou035_list, iou075_list = [], [], [], []
                prob_sum = 0.0
                prob_pos_sum = 0.0
                prob_neg_sum = 0.0
                pix_pos = 0
                pix_neg = 0
                n_imgs = 0
                img_count = 0

                once = True
                for batch in dl_va:
                    images = batch.get("image", batch.get("images")).to(device, non_blocking=True)
                    soft   = batch["soft_mask"].to(device, dtype=torch.float32)

                    out = model({"image": images, "text": _ensure_list_text(batch.get("text"), images.size(0))}, compute_lm=False)

                    if "attn_logits" in out and out["attn_logits"] is not None and out["attn_logits"].ndim == 4:
                        A_logit_hw = torch.nan_to_num(out["attn_logits"], nan=0.0, posinf=0.0, neginf=0.0).to(torch.float32)
                        Ht, Wt = A_logit_hw.shape[-2:]
                    else:
                        A_prob_hw = torch.nan_to_num(out["attn_maps"], nan=0.0, posinf=1.0, neginf=0.0).clamp(1e-6, 1-1e-6)
                        Ht, Wt = A_prob_hw.shape[-2:]
                        A_logit_hw = torch.logit(A_prob_hw)
                        sm_hw = _mask_to_feat_hw(soft, (Ht, Wt))
                        probs = torch.sigmoid(A_logit_hw).to(torch.float32)
                        gt    = sm_hw.to(torch.float32)
                        B = probs.size(0)
                        img_count += B
                        flat = probs.view(B, -1)

                    prob_sum += float(probs.mean().item()) * B
                    pos_mask = (gt >= 0.5)
                    neg_mask = ~pos_mask
                    pv = probs[pos_mask]
                    nv = probs[neg_mask]
                    if pv.numel() > 0:
                        prob_pos_sum += float(pv.mean().item()) * pv.numel()
                        pix_pos += pv.numel()
                    if nv.numel() > 0:
                        prob_neg_sum += float(nv.mean().item()) * nv.numel()
                        pix_neg += nv.numel()
                    p = eval_q1
                    thr_q = torch.quantile(flat, q=p, dim=1).view(B, 1, 1, 1)
                    pred_q = (probs >= thr_q).float()

                    def _soft_iou(pred, gt):
                        inter = (pred * gt).sum((2, 3))
                        union = (pred + gt - pred * gt).sum((2, 3)) + 1e-6
                        return (inter / union).mean().item()

                    pred_05   = (probs >= 0.5).float()
                    pred_035  = (probs >= 0.35).float()
                    pred_075  = (probs >= 0.75).float()

                    iou_q     = _soft_iou(pred_q, gt)
                    iou_05    = _soft_iou(pred_05, gt)
                    iou_035   = _soft_iou(pred_035, gt)
                    iou_075   = _soft_iou(pred_075, gt)

                    iou_list.append(iou_q)
                    iou05_list.append(iou_05)
                    iou035_list.append(iou_035)
                    iou075_list.append(iou_075)

                    dicebce = safe_dice_from_probs(probs, gt) + 0.5 * safe_bce_from_probs(probs, gt)
                    loss_attn_val = nn.BCEWithLogitsLoss(reduction="mean")(A_logit_hw, gt)
                    total += (lw.get("attn", 0.5) * loss_attn_val.item() + (cfg.get("box_target_weight", 1.0)) * dicebce.item())
                    count += 1

                    if once:
                        print("[val-sanity] images mean/std:", float(images.mean()), float(images.std()))
                        sm = batch["soft_mask"].to(torch.float32)
                        print("[val-sanity] soft_mask mean/std/sum:", float(sm.mean()), float(sm.std()), float(sm.sum()))
                        print("[val-sanity] soft_mask >0.5 ratio:", float((sm > 0.5).float().mean()))
                        once = False

                val_loss   = total / max(count, 1)
                val_iou_q   = float(np.mean(iou_list))    if len(iou_list)   > 0 else 0.0
                val_iou_05  = float(np.mean(iou05_list))  if len(iou05_list) > 0 else 0.0
                val_iou_035 = float(np.mean(iou035_list)) if len(iou035_list) > 0 else 0.0
                val_iou_075 = float(np.mean(iou075_list)) if len(iou075_list) > 0 else 0.0

            model.load_state_dict(backup)

            dt = time.time() - t0
            epoch_ips = float(iter_ema_ips or 0.0)
            total_elapsed = time.perf_counter() - wall0
            print(
                f"[Epoch {ep}] val_loss={val_loss:.4f} | val_iou@p(cur)={val_iou_q:.3f} | "
                f"val_iou@0.35={val_iou_035:.3f} | val_iou@0.5={val_iou_05:.3f} | val_iou@0.75={val_iou_075:.3f}  "
                f"({dt:.1f}s) | ips~{epoch_ips:.1f} img/s | elapsed={total_elapsed/3600.0:.2f}h"
            )

            # ログ（エポック）
            epoch_row = {
                "epoch": ep,
                "val_loss": round(float(val_loss), 6),
                "val_iou_p": round(float(val_iou_q), 6),
                "val_iou_035": round(float(val_iou_035), 6),
                "val_iou_05": round(float(val_iou_05), 6),
                "val_iou_075": round(float(val_iou_075), 6),
                "elapsed_sec": round(float(dt), 1),
                "imgs_per_sec_ema": None,
                "sec_per_iter_ema": None,
            }
            writer.add_scalar("val/loss", float(val_loss), ep)
            writer.add_scalar("val/iou@p(cur)", float(val_iou_q), ep)
            writer.add_scalar("val/iou@0.35", float(val_iou_035), ep)
            writer.add_scalar("val/iou@0.5", float(val_iou_05), ep)
            writer.add_scalar("val/iou@0.75", float(val_iou_075), ep)
            writer.add_scalar("val/prob_mean", float(prob_sum / max(1, img_count)), ep)
            if pix_pos > 0:
                writer.add_scalar("val/prob_pos_mean", float(prob_pos_sum / pix_pos), ep)
            if pix_neg > 0:
                writer.add_scalar("val/prob_neg_mean", float(prob_neg_sum / pix_neg), ep)
            try: writer.flush()
            except Exception: pass

            _csv_append(epoch_csv, epoch_row, field_order=["epoch","val_loss","val_iou_p","val_iou_035","val_iou_05","val_iou_075","elapsed_sec","imgs_per_sec_ema","sec_per_iter_ema"])
            _jsonl_append(epoch_jsonl, {**epoch_row,
                                        "quantile_p": cfg.get("eval", {}).get("quantile_p", 0.85),
                                        "quantile_p0": cfg.get("eval", {}).get("quantile_p0", 0.90)})
            # 早期停止
            score = val_iou_05 if early_metric == "val_iou_05" else val_iou_q
            if score > early_best:
                early_best, early_wait = score, 0
            else:
                early_wait += 1
            if early_patience > 0 and early_wait >= early_patience:
                print(f"[early-stop] no improvement on {early_metric} for {early_patience} epochs.")
                reached_max = True  # 保存後に抜ける
        # ========== 保存（軽量→フル） ==========
        if ema is not None:
            ema.copy_to(model)
        light_sd = {
            "model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
            "cfg": cfg, "epoch": ep
        }
        light_path = os.path.join(out_dir, "checkpoints", f"weights_ep{ep}.pt")
        try:
            atomic_torch_save(light_sd, light_path, use_legacy=False)
        except Exception as e:
            print("[warn] light save zip failed, retry legacy:", repr(e))
            atomic_torch_save(light_sd, light_path, use_legacy=True)
        if atomic_safe_save is not None:
            st_path = os.path.join(out_dir, "checkpoints", f"weights_ep{ep}.safetensors")
            try:
                atomic_safe_save({k: v.detach().cpu() for k, v in model.state_dict().items()}, st_path)
            except Exception as e:
                print("[warn] safetensors light save failed:", repr(e))
        # フル
        full_ckpt = {
            "model": model.state_dict(),
            "optimizer": optim.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "scaler": scaler.state_dict() if scaler is not None else None,
            "ema": (getattr(ema, "ema", None) if ema is not None else None),
            "cfg": cfg,
            "epoch": ep,
            "global_step": global_step,
        }
        full_path = os.path.join(out_dir, "checkpoints", f"full_ep{ep}.pt")
        try:
            atomic_torch_save(full_ckpt, full_path, use_legacy=False)
        except Exception as e:
            print("[warn] full save zip failed, retry legacy:", repr(e))
            atomic_torch_save(full_ckpt, full_path, use_legacy=True)
        if max_steps > 0 and reached_max:
            print(f"[train] Training stopped after reaching max_steps={max_steps}. Final epoch: {ep}")
            break

        # 元重みへ戻す
        # skip_val のときは backup を作っていないので存在チェック
        if 'backup' in locals():
            model.load_state_dict(backup)
    writer.close()
    # ---- total wall time ----
    total_wall = time.perf_counter() - wall0
    hh = int(total_wall // 3600); mm = int((total_wall % 3600) // 60); ss = int(total_wall % 60)
    print(f"[train] total wall time: {hh:02d}:{mm:02d}:{ss:02d}  ({total_wall:.1f}s)")
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", type=str, required=True, help="yaml config")
    args = ap.parse_args()
    with open(args.cfg, "r") as f:
        cfg = yaml.safe_load(f)

    train(cfg)

if __name__ == "__main__":
    main()
