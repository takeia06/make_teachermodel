# tools/eval_phase1p5.py
#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Phase1.5 ICL Evaluation (robust + signature-aware + deep debug)

- Dataset:   RefCOCO JSON/JSONL (your format)
- Collate:   images/bboxes -> stacked; texts/slots/ids -> lists
- ICL:       optional demos (K-shot), slot template fallback (force disable-able)
- Infer:     signature-aware caller: tries positional & dict batches before kwargs
- Weights:   safetensors/pt, prefix strip + shape auto-fix
- Outputs:   CSV / JSONL
- Debug:     raw outputs / attempts / prompts / call-traces dumped to files

【追加の主な変更点】
- eval.use_saliency_fallback (bool):
  サリエンシー救済（巨大箱/無効出力のときにエッジから箱を推定）を完全ON/OFF可能。
  差分評価の際に救済が学習差を"埋める"問題を避けられる。
- prefer_attn_box のキー修正:
  YAML 側の prefer_attn_box を正しく拾い、必要なら prefer_attn_bbox や
  環境変数 EVAL_PREFER_ATTN=1 でも上書き可能に。
- EVAL_DEBUG_VERBOSE (env):
  0 なら簡素ログ、1 なら詳細ログ（試行失敗/成功のトレース等）を出力。
"""

from __future__ import annotations
import os
import sys
import json
import argparse
import random
import re
import inspect
from typing import Any, Dict, List, Tuple, Optional
from pathlib import Path
from functools import lru_cache

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as T
from tqdm import tqdm
import yaml
from collections.abc import Mapping
import numpy as np
import cv2

# --------------------------------
# repo paths
# --------------------------------
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
CODE_DIR = os.path.join(REPO_ROOT, "code")
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)

# --------------------------------
# ICL builder (external or fallback)
# --------------------------------
try:
    from code.utils.icl_prompt import ICLPromptBuilder as ExternalICLBuilder  # type: ignore
except Exception:
    ExternalICLBuilder = None


class ICLPromptBuilder:
    def __init__(
        self, bank_jsonl: Optional[str] = None, slot_template_path: Optional[str] = None
    ):
        self.demos: List[Dict[str, Any]] = []
        if bank_jsonl and os.path.isfile(bank_jsonl):
            with open(bank_jsonl, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        self.demos.append(json.loads(line))
                    except Exception:
                        pass
        self.tmpl: Optional[str] = None
        if slot_template_path and os.path.isfile(slot_template_path):
            self.tmpl = open(slot_template_path, "r", encoding="utf-8").read().strip()

    def _ensure_slots(self, slots: Optional[Dict[str, str]]) -> Dict[str, str]:
        base = {
            "part": "object",
            "defect": "target",
            "color": "unknown",
            "shape": "unknown",
            "texture": "unknown",
            "position_size": "unknown",
            "position": "",
            "size": "",
        }
        if not slots:
            return base
        for k, v in slots.items():
            key = str(k).lower()
            if v is None:
                continue
            val = _normalize_slot_value(v)
            if not val:
                continue
            if key in base:
                base[key] = val
            elif key in ("position", "size"):
                base[key] = val
        if not base.get("position_size") or base["position_size"] == "unknown":
            parts = [base.get("position", ""), base.get("size", "")]
            parts = [p for p in parts if p]
            if parts:
                base["position_size"] = ", ".join(parts)
        return base

    def _fill_from_slots(self, slots: Optional[Dict[str, str]]) -> Optional[str]:
        if not self.tmpl:
            return None
        s = self._ensure_slots(slots)
        try:
            return self.tmpl.format(
                part=s["part"],
                defect=s["defect"],
                color=s["color"],
                shape=s["shape"],
                texture=s["texture"],
                position_size=s["position_size"],
            )
        except Exception:
            return None

    def build(
        self,
        k: int,
        query_text: Optional[str] = None,
        slots: Optional[Dict[str, str]] = None,
    ) -> str:
        demos: List[str] = []
        if k and self.demos:
            for d in self.demos[: max(0, int(k))]:
                t = d.get("demo_text") or d.get("text") or ""
                if t:
                    demos.append(f"Example:\n{t}")
        q = self._fill_from_slots(slots)
        if not q:
            q = (
                query_text or "Locate the referred object and return the bounding box."
            ).strip()
        demos.append(f"Query:\n{q}")
        return "\n---\n".join(demos)


# --------------------------------
# Dataset
# --------------------------------
class RefCOCOJsonDataset(Dataset):
    def __init__(self, cfg: dict, split: str = "val"):
        self.split = split
        dcfg = cfg.get("data", {})
        key = "val_json" if split != "train" else "train_json"
        paths = dcfg.get(key, [])
        if isinstance(paths, str):
            paths = [paths]
        self.json_paths = [p for p in paths if os.path.isfile(p)]
        if not self.json_paths:
            raise FileNotFoundError(f"No {key} files found in cfg.data")

        self.img_roots = dcfg.get("img_root", {})
        self.long_side = int(dcfg.get("long_side", 896))

        self.recs: List[Dict[str, Any]] = []
        for p in self.json_paths:
            if p.endswith(".jsonl"):
                with open(p, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            self.recs.append(json.loads(line))
                        except Exception:
                            pass
            else:
                with open(p, "r", encoding="utf-8") as f:
                    js = json.load(f)
                    if isinstance(js, list):
                        self.recs.extend(js)
                    elif isinstance(js, dict) and "annotations" in js:
                        self.recs.extend(js["annotations"])
                    else:
                        self.recs.append(js)

        self._to_tensor = T.Compose(
            [
                T.ToTensor(),
                T.Normalize(
                    mean=[0.48145466, 0.4578275, 0.40821073],
                    std=[0.26862954, 0.26130258, 0.27577711],
                ),
            ]
        )

    def __len__(self) -> int:
        return len(self.recs)

    def _resolve_image_path(self, rec: Dict[str, Any]) -> str:
        p = rec.get("image_path") or rec.get("file_path") or rec.get("file_name")
        if not p:
            raise RuntimeError("record missing image_path/file_name")
        if os.path.isabs(p) and os.path.isfile(p):
            return p
        fname = os.path.basename(p)
        for key in ("val2014", "train2014", "default"):
            root = self.img_roots.get(key)
            if root:
                cand = os.path.join(root, fname)
                if os.path.isfile(cand):
                    return cand
        return os.path.join(self.img_roots.get("default", ""), p)

    def _resize_long_side_letterbox(self, img: Image.Image):
        W, H = img.size
        scale = float(self.long_side) / max(W, H)
        newW, newH = int(round(W * scale)), int(round(H * scale))
        img_resized = img.resize((newW, newH), Image.BICUBIC)
        canvas = Image.new("RGB", (self.long_side, self.long_side), (0, 0, 0))
        canvas.paste(img_resized, (0, 0))
        return canvas

    @staticmethod
    def _xyxy_pix_to_norm_xyxy(bb: List[float], W: int, H: int) -> List[float]:
        w = max(1.0, float(W))
        h = max(1.0, float(H))
        x1, y1, x2, y2 = [float(v) for v in bb]
        x1 = max(0.0, min(1.0, x1 / w))
        x2 = max(0.0, min(1.0, x2 / w))
        y1 = max(0.0, min(1.0, y1 / h))
        y2 = max(0.0, min(1.0, y2 / h))
        if x2 < x1:
            x1, x2 = x2, x1
        if y2 < y1:
            y1, y2 = y2, y1
        return [x1, y1, x2, y2]

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        r = self.recs[idx]
        img_path = self._resolve_image_path(r)
        img = Image.open(img_path).convert("RGB")
        W = int(r.get("width") or img.size[0])
        H = int(r.get("height") or img.size[1])

        bb = r.get("bbox") or r.get("gt_bbox")
        if not bb or len(bb) != 4:
            raise RuntimeError("record missing bbox")
        gt_norm = self._xyxy_pix_to_norm_xyxy(bb, W, H)

        img_sqr = self._resize_long_side_letterbox(img)
        img_t = self._to_tensor(img_sqr)

        return {
            "image": img_t,
            "gt_bbox": gt_norm,
            "ref_expr": r.get("text") or r.get("ref_expr") or "",
            "slots": r.get("slots"),
            "id": r.get("id", idx),
        }


# --------------------------------
# collate
# --------------------------------
def collate_batch(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    from torch.utils.data._utils.collate import default_collate
    import torch

    images = default_collate([b["image"] for b in batch])
    gt_bbox = torch.tensor([b["gt_bbox"] for b in batch], dtype=torch.float)
    ref_expr = [
        (b.get("ref_expr") or "Locate the referred object and return the bbox.").strip()
        for b in batch
    ]
    ids = [int(b.get("id", i)) for i, b in enumerate(batch)]

    def _ensure_slots(s):
        return (
            s
            if s is not None
            else {
                "part": "object",
                "defect": "target",
                "color": "unknown",
                "shape": "unknown",
                "texture": "unknown",
                "position_size": "unknown",
            }
        )

    slots = [_ensure_slots(b.get("slots")) for b in batch]
    return {
        "image": images,
        "gt_bbox": gt_bbox,
        "ref_expr": ref_expr,
        "slots": slots,
        "id": ids,
    }


# --------------------------------
# IoU
# --------------------------------
def iou_xyxy_norm(a: List[float], b: List[float]) -> float:
    if not (
        isinstance(a, (list, tuple))
        and isinstance(b, (list, tuple))
        and len(a) == 4
        and len(b) == 4
    ):
        return 0.0
    ax1, ay1, ax2, ay2 = [float(x) for x in a]
    bx1, by1, bx2, by2 = [float(x) for x in b]
    ax1, ax2 = (ax1, ax2) if ax1 <= ax2 else (ax2, ax1)
    ay1, ay2 = (ay1, ay2) if ay1 <= ay2 else (ay2, ay1)
    bx1, bx2 = (bx1, bx2) if bx1 <= bx2 else (bx2, bx1)
    by1, by2 = (by1, by2) if by1 <= by2 else (by2, by1)
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    aw, ah = max(0.0, ax2 - ax1), max(0.0, ay2 - ay1)
    bw, bh = max(0.0, bx2 - bx1), max(0.0, by2 - by1)
    union = aw * ah + bw * bh - inter
    return float(inter / union) if union > 0 else 0.0


# --------------------------------
# build_model import helper
# --------------------------------
def _load_build_model_function(repo_root: str):
    import importlib
    import importlib.util

    tried = []
    for mn in ("code.trainers.train_refcoco_inst", "trainers.train_refcoco_inst"):
        try:
            mod = importlib.import_module(mn)
            if hasattr(mod, "build_model"):
                return getattr(mod, "build_model"), f"pkg:{mn}"
        except Exception as e:
            tried.append(f"{mn}: {e}")
    for p in (
        os.path.join(repo_root, "code", "trainers", "train_refcoco_inst.py"),
        os.path.join(repo_root, "trainers", "train_refcoco_inst.py"),
    ):
        if os.path.isfile(p):
            try:
                spec = importlib.util.spec_from_file_location("phase1_trainer", p)
                mod = importlib.util.module_from_spec(spec)  # type: ignore
                sys.modules["phase1_trainer"] = mod
                assert spec and spec.loader
                spec.loader.exec_module(mod)  # type: ignore
                if hasattr(mod, "build_model"):
                    return getattr(mod, "build_model"), f"path:{p}"
            except Exception as e:
                tried.append(f"path:{p}: {e}")
    raise RuntimeError("Could not import build_model. Tried:\n" + "\n".join(tried))


# --------------------------------
# weight loading
# --------------------------------
def _robust_load_weights(model: nn.Module, resume_path: str, device: torch.device):
    if not resume_path or not os.path.isfile(resume_path):
        raise FileNotFoundError(f"resume weights not found: {resume_path}")
    from safetensors.torch import load_file as safe_load

    model_keys = set(model.state_dict().keys())

    def strip_prefix(sd, prefix):
        return {
            (k[len(prefix) :] if k.startswith(prefix) else k): v for k, v in sd.items()
        }

    def strip_n_segments(sd, n):
        out = {}
        for k, v in sd.items():
            parts = k.split(".")
            out[".".join(parts[n:])] = v if len(parts) > n else v
        return out

    def cand_transforms(sd):
        cands = [("as-is", sd)]
        for p in ("model_ema.", "ema.", "module.", "model."):
            if any(k.startswith(p) for k in sd.keys()):
                cands.append((f"strip:{p}", strip_prefix(sd, p)))
        for p1 in ("model_ema.", "ema.", "model.", "module."):
            tmp = strip_prefix(sd, p1)
            for p2 in ("model.", "module."):
                if any(k.startswith(p2) for k in tmp.keys()):
                    cands.append((f"strip:{p1}+{p2}", strip_prefix(tmp, p2)))
        for n in (1, 2, 3):
            cands.append((f"strip_segments:{n}", strip_n_segments(sd, n)))
        return cands

    def score_fit(sd):
        return len(model_keys.intersection(sd.keys()))

    def resize_like(
        param: torch.Tensor, target_shape: Tuple[int, ...]
    ) -> Optional[torch.Tensor]:
        ts = tuple(target_shape)
        if not torch.is_tensor(param):
            param = torch.tensor(param)
        src = param
        if src.dim() != len(ts):
            return None
        if src.shape == ts:
            return src
        if src.dim() == 2 and src.shape[1] == ts[1]:
            v, d = src.shape
            tv, td = ts
            if v > tv:
                return src[:tv, :]
            out = torch.zeros(tv, td, dtype=src.dtype, device=src.device)
            out[:v, :] = src
            return out
        slicers = []
        pads = []
        for s, t in zip(src.shape, ts):
            if s >= t:
                slicers.append(slice(0, t))
                pads.extend([0, 0])
            else:
                slicers.append(slice(0, s))
                pads.extend([0, t - s])
        base = src[tuple(slicers)].clone()
        if list(base.shape) == list(ts):
            return base
        pads = pads[::-1]
        return torch.nn.functional.pad(base, pads)

    def apply_shape_fixes(sd, model_state):
        fixed = {}
        dropped = []
        for k, v in sd.items():
            if k not in model_state:
                continue
            tgt = model_state[k]
            v2 = v
            if torch.is_tensor(v2):
                if v2.shape != tgt.shape:
                    v2 = resize_like(v2, tuple(tgt.shape))
            else:
                try:
                    vv = torch.tensor(v2)
                    if vv.shape != tgt.shape:
                        vv = resize_like(vv, tuple(tgt.shape))
                    v2 = vv
                except Exception:
                    v2 = None
            if v2 is None:
                dropped.append(
                    (k, f"shape {getattr(v,'shape',None)}->{tuple(tgt.shape)}")
                )
            else:
                fixed[k] = v2
        return fixed, dropped

    loaded = False
    raw = None
    try:
        raw = safe_load(resume_path)
    except Exception as e:
        print(f"[warn] safetensors load failed: {e}")

    if isinstance(raw, dict) and raw:
        best_tag, best_sd, best_score = None, None, -1
        for tag, cand in cand_transforms(raw):
            sc = score_fit(cand)
            if sc > best_score:
                best_tag, best_sd, best_score = tag, cand, sc
        print(
            f"[info] best key alignment: {best_tag} (matches={best_score}/{len(model_keys)})"
        )
        fixed, dropped = apply_shape_fixes(best_sd, model.state_dict())
        if dropped:
            print(
                f"[warn] shape-mismatch fixed/dropped: {len(dropped)} (showing up to 5)"
            )
            for i, (k, why) in enumerate(dropped[:5]):
                print(f"  - {k}: {why}")
        missing, unexpected = model.load_state_dict(fixed, strict=False)
        print(
            f"[info] load_state_dict: missing={len(missing)} unexpected={len(unexpected)}"
        )
        loaded = True

    if not loaded:
        try:
            sd = torch.load(resume_path, map_location=device, weights_only=False)
            if isinstance(sd, dict):
                for key in (
                    "state_dict",
                    "model",
                    "model_state",
                    "model_ema",
                    "ema_state_dict",
                    "module",
                ):
                    if key in sd and isinstance(sd[key], dict):
                        sd = sd[key]
                        break
            best_tag, best_sd, best_score = None, None, -1
            for tag, cand in cand_transforms(sd):
                sc = score_fit(cand)
                if sc > best_score:
                    best_tag, best_sd, best_score = tag, cand, sc
            print(
                f"[info] best key alignment (torch): {best_tag} (matches={best_score}/{len(model_keys)})"
            )
            fixed, dropped = apply_shape_fixes(best_sd, model.state_dict())
            if dropped:
                print(
                    f"[warn] shape-mismatch fixed/dropped: {len(dropped)} (showing up to 5)"
                )
                for i, (k, why) in enumerate(dropped[:5]):
                    print(f"  - {k}: {why}")
            missing, unexpected = model.load_state_dict(fixed, strict=False)
            print(
                f"[info] load_state_dict: missing={len(missing)} unexpected={len(unexpected)}"
            )
            loaded = True
        except Exception as e:
            print(f"[warn] torch.load fallback failed: {e}")

    if not loaded:
        try:
            from safetensors import safe_open

            with safe_open(resume_path, framework="pt", device="cpu") as f:
                keys = list(f.keys())
                print(f"[debug] first keys in safetensors: {keys[:20]}")
        except Exception:
            pass
        raise RuntimeError(f"Could not load weights from {resume_path}")


# --------------------------------
# infer adapter (signature-aware)
# --------------------------------
def attach_infer_adapter(model: nn.Module, cfg: dict, force_override: bool = True):
    if (
        hasattr(model, "infer")
        and callable(getattr(model, "infer"))
        and not force_override
    ):
        return
    _ = getattr(model, "infer", None)

    from contextlib import nullcontext

    device = next(model.parameters()).device
    amp_enabled = bool(cfg.get("eval", {}).get("amp", False))
    debug_enabled = bool(cfg.get("eval", {}).get("debug", True))
    # === 環境変数で冗長ログ制御（EVAL_DEBUG_VERBOSE=1で詳細） ===
    very_verbose = os.environ.get("EVAL_DEBUG_VERBOSE", "0") not in (
        "0",
        "false",
        "False",
    )

    # ==== 評価オプション（attn優先 / サリエンシー置換ON/OFF） ====
    # YAMLは prefer_attn_box を正として読み、後方互換で prefer_attn_bbox も許容。
    prefer_attn_from_cfg = bool(
        cfg.get("eval", {}).get("prefer_attn_box", False)
    ) or bool(cfg.get("eval", {}).get("prefer_attn_bbox", False))
    prefer_attn_from_env = os.environ.get("EVAL_PREFER_ATTN", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )
    prefer_attn = prefer_attn_from_env or prefer_attn_from_cfg

    # use_saliency_fallback も環境変数で上書き可能
    if "EVAL_SALIENCY_FALLBACK" in os.environ:
        use_saliency_fallback = os.environ.get(
            "EVAL_SALIENCY_FALLBACK", "1"
        ).strip().lower() not in ("0", "false", "no", "off")
    else:
        use_saliency_fallback = bool(
            cfg.get("eval", {}).get("use_saliency_fallback", True)
        )

    first_batch_marks = {"dumped": False}
    attempts_log_written = {"done": False}

    # -------------- dump helpers --------------
    def _tensor_to_py(o: Any, max_list_len: int = 4096):
        try:
            if torch.is_tensor(o):
                return o.detach().cpu().tolist()
            if isinstance(o, (list, tuple)):
                out = []
                for i, v in enumerate(o):
                    if i >= max_list_len:
                        out.append(f"...truncated {len(o)-max_list_len} items...")
                        break
                    out.append(_tensor_to_py(v))
                return out
            if isinstance(o, dict):
                return {str(k): _tensor_to_py(v) for k, v in o.items()}
            return o
        except Exception:
            try:
                return str(o)
            except Exception:
                return "<unserializable>"

    def _dump_json(obj: Any, path: str):
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(obj, f, ensure_ascii=False, indent=2)
            if very_verbose:
                print(f"[debug] dumped: {path}")
        except Exception as e:
            print(f"[warn] dump failed ({path}): {e}")

    def _append_line(path: str, line: str):
        try:
            with open(path, "a", encoding="utf-8") as f:
                f.write(line.rstrip() + "\n")
        except Exception as e:
            print(f"[warn] write failed ({path}): {e}")

    # -------------- text helpers --------------
    def _ensure_4d(img: torch.Tensor) -> torch.Tensor:
        return img if img.dim() == 4 else img.unsqueeze(0)

    def _as_str_list(x) -> List[str]:
        if isinstance(x, str):
            return [x]
        if isinstance(x, (list, tuple)):
            return [str(t).strip() for t in x]
        return [str(x)]

    def _mk_tokens(txt_list_in) -> Optional[Dict[str, torch.Tensor]]:
        if not hasattr(model, "tokenizer") or model.tokenizer is None:
            return None
        txt_list = _as_str_list(txt_list_in)
        max_len = int(cfg.get("text", {}).get("max_length", 64))
        tries = [
            dict(
                return_tensors="pt", padding=True, truncation=True, max_length=max_len
            ),
            dict(
                return_tensors="pt",
                padding="longest",
                truncation=True,
                max_length=max_len,
            ),
            dict(return_tensors="pt", padding=True, max_length=max_len),
            dict(return_tensors="pt", padding=True),
        ]
        last_err = None
        for kwargs in tries:
            try:
                tok = model.tokenizer(txt_list, **kwargs)
                if isinstance(tok, Mapping):
                    return {
                        k: (v.to(device) if hasattr(v, "to") else v)
                        for k, v in tok.items()
                    }
                if hasattr(tok, "data") and isinstance(tok.data, Mapping):
                    d = tok.data
                    return {
                        k: (v.to(device) if hasattr(v, "to") else v)
                        for k, v in d.items()
                    }
            except Exception as e:
                last_err = e
                continue
        if debug_enabled:
            print(f"[warn] tokenizer failed: {repr(last_err)}")
        return None

    # ---- utils for box parsing ----
    def _to_list4(v) -> Optional[List[float]]:
        if v is None:
            return None
        if (
            isinstance(v, (list, tuple))
            and len(v) == 4
            and all(isinstance(x, (int, float, np.generic)) for x in v)
        ):
            return [float(x) for x in v]
        if torch.is_tensor(v) and v.numel() == 4:
            return [float(x) for x in v.detach().cpu().flatten().tolist()]
        if hasattr(v, "__class__") and "numpy" in v.__class__.__module__:
            try:
                arr = v.reshape(-1)
                if arr.size == 4:
                    return [float(x) for x in arr.tolist()]
            except Exception:
                pass
        return None

    def _bbox_from_attn_tensor(
        attn: torch.Tensor, min_area_frac: float = 0.0005
    ) -> List[List[float]]:
        """
        attn: (B,1,H,W) in [0,1] or logits -> [[x1,y1,x2,y2], ...] (normalized)
        """
        if attn.dim() == 4 and attn.size(1) == 1:
            A = attn.detach().float().cpu().numpy()  # (B,1,H,W)
            B, _, H, W = A.shape
            out = []
            for b in range(B):
                m = A[b, 0]
                if m.min() < 0 or m.max() > 1:
                    m = 1.0 / (1.0 + np.exp(-m))
                thr = max(0.3, float(m.mean() + 0.5 * m.std()))
                binm = (m >= thr).astype(np.uint8) * 255
                cnts, _ = cv2.findContours(
                    binm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                if not cnts:
                    yx = np.unravel_index(np.argmax(m), m.shape)
                    y, x = int(yx[0]), int(yx[1])
                    x1 = max(0, x - 1)
                    y1 = max(0, y - 1)
                    x2 = min(W - 1, x + 1)
                    y2 = min(H - 1, y + 1)
                else:
                    areas = [cv2.contourArea(c) for c in cnts]
                    c = cnts[int(np.argmax(areas))]
                    x, y, w, h = cv2.boundingRect(c)
                    if (w * h) < (min_area_frac * H * W):
                        yx = np.unravel_index(np.argmax(m), m.shape)
                        y, x = int(yx[0]), int(yx[1])
                        x1 = max(0, x - 2)
                        y1 = max(0, y - 2)
                        x2 = min(W - 1, x + 2)
                        y2 = min(H - 1, y + 2)
                    else:
                        x1, y1, x2, y2 = x, y, x + w, y + h
                out.append([x1 / float(W), y1 / float(H), x2 / float(W), y2 / float(H)])
            return out
        return []

    def _bbox_from_saliency_tensor(
        imgs: torch.Tensor, area_floor_frac: float = 0.0005
    ) -> List[List[float]]:
        """
        画像のエッジサリエンシーから bbox を推定（学習なし救済）。
        imgs: (B,3,H,W)
        return: [[x1,y1,x2,y2], ...] normalized
        """
        with torch.no_grad():
            B, C, H, W = imgs.shape
            out: List[List[float]] = []
            mean = np.array(
                [0.48145466, 0.4578275, 0.40821073], dtype=np.float32
            ).reshape(3, 1, 1)
            std = np.array(
                [0.26862954, 0.26130258, 0.27577711], dtype=np.float32
            ).reshape(3, 1, 1)

            for b in range(B):
                x = imgs[b].detach().cpu().float().numpy()  # (C,H,W)
                x = (x * std + mean).clip(0.0, 1.0)
                x_u8 = (x * 255.0).astype(np.uint8)

                if x_u8.shape[0] == 3:
                    img_rgb = np.transpose(x_u8, (1, 2, 0))  # (H,W,3)
                    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
                else:
                    gray = np.transpose(x_u8, (1, 2, 0)).squeeze(-1)

                gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
                gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
                mag = cv2.magnitude(gx, gy)
                mag = (mag / (mag.max() + 1e-6) * 255.0).astype(np.uint8)

                _, binm = cv2.threshold(
                    mag, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
                )
                cnts, _ = cv2.findContours(
                    binm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )

                if not cnts:
                    yx = np.unravel_index(np.argmax(mag), mag.shape)
                    y, x0 = int(yx[0]), int(yx[1])
                    x1 = max(0, x0 - 2)
                    y1 = max(0, y - 2)
                    x2 = min(W - 1, x0 + 2)
                    y2 = min(H - 1, y + 2)
                else:
                    areas = [cv2.contourArea(c) for c in cnts]
                    c = cnts[int(np.argmax(areas))]
                    x0, y0, ww, hh = cv2.boundingRect(c)
                    if (ww * hh) < (area_floor_frac * H * W):
                        yx = np.unravel_index(np.argmax(mag), mag.shape)
                        y0, x0 = int(yx[0]), int(yx[1])
                        x1 = max(0, x0 - 2)
                        y1 = max(0, y0 - 2)
                        x2 = min(W - 1, x0 + 2)
                        y2 = min(H - 1, y0 + 2)
                    else:
                        x1, y1, x2, y2 = x0, y0, x0 + ww, y0 + hh

                out.append([x1 / float(W), y1 / float(H), x2 / float(W), y2 / float(H)])
            return out

    _num_re = re.compile(r"[-+]?(?:\d+\.\d+|\d+)")

    def _box_from_text(s: str) -> Optional[List[float]]:
        if not isinstance(s, str) or len(s) == 0:
            return None
        st = s.strip()
        if (st.startswith("{") and st.endswith("}")) or (
            st.startswith("[") and st.endswith("]")
        ):
            try:
                js = json.loads(st)
                bb = _box_from_any(js)
                if bb:
                    return bb
            except Exception:
                pass
        nums = _num_re.findall(st)
        if len(nums) >= 4:
            try:
                return list(map(float, nums[:4]))
            except Exception:
                return None
        return None

    def _xywh_to_xyxy(b):
        if b is None:
            return None
        x, y, w, h = [float(v) for v in b]
        return [x, y, x + w, y + h]

    def _box_from_any(x) -> Optional[List[float]]:
        l4 = _to_list4(x)
        if l4 is not None:
            return l4
        if isinstance(x, str):
            return _box_from_text(x)
        if isinstance(x, dict):
            if all(k in x for k in ("x1", "y1", "x2", "y2")):
                try:
                    return [
                        float(x["x1"]),
                        float(x["y1"]),
                        float(x["x2"]),
                        float(x["y2"]),
                    ]
                except Exception:
                    pass
            if all(k in x for k in ("x", "y", "w", "h")):
                try:
                    return _xywh_to_xyxy(
                        [float(x["x"]), float(x["y"]), float(x["w"]), float(x["h"])]
                    )
                except Exception:
                    pass
            for k in (
                "bbox",
                "box",
                "pred_bbox",
                "pred_box",
                "box_xyxy_norm",
                "pred_boxes",
                "bboxes",
                "det_box",
                "xyxy",
                "norm_bbox",
                "bbox_norm",
                "xywh",
                "bbox_xywh",
            ):
                if k in x:
                    b = _box_from_any(x[k])
                    if b is not None:
                        return b
            if (
                "points" in x
                and isinstance(x["points"], (list, tuple))
                and len(x["points"]) >= 2
            ):
                try:
                    (x1, y1), (x2, y2) = x["points"][0], x["points"][1]
                    return [float(x1), float(y1), float(x2), float(y2)]
                except Exception:
                    pass
            for k in ("output", "out", "data", "result"):
                if k in x:
                    b = _box_from_any(x[k])
                    if b is not None:
                        return b
        if isinstance(x, (list, tuple)) and len(x) > 0:
            if _to_list4(x[0]) is not None:
                return _to_list4(x[0])
            if isinstance(x[0], str):
                bb = _box_from_text(x[0])
                if bb:
                    return bb
            if isinstance(x[0], dict):
                bb = _box_from_any(x[0])
                if bb:
                    return bb
        return None

    def _post_norm_and_scale(
        norm_list: List[Optional[List[float]]],
    ) -> List[Optional[List[float]]]:
        ls = int(cfg.get("data", {}).get("long_side", 896))

        def _fix(bb):
            if bb is None:
                return None
            if any(abs(v) > 1.5 for v in bb):  # pixel-like
                bb = [v / float(ls) for v in bb]
            x1, y1, x2, y2 = [min(1.0, max(0.0, float(v))) for v in bb]
            if x2 < x1:
                x1, x2 = x2, x1
            if y2 < y1:
                y1, y2 = y2, y1
            # --- 微膨張（IoU安定用） ---
            pad = 0.002
            x1 = max(0.0, x1 - pad)
            y1 = max(0.0, y1 - pad)
            x2 = min(1.0, x2 + pad)
            y2 = min(1.0, y2 + pad)
            # 細すぎ救済
            w = x2 - x1
            h = y2 - y1
            if w * h < 0.0008 or min(w, h) < 0.01:
                x1 = max(0.0, x1 - 0.003)
                y1 = max(0.0, y1 - 0.003)
                x2 = min(1.0, x2 + 0.003)
                y2 = min(1.0, y2 + 0.003)
            return [x1, y1, x2, y2]

        return [_fix(bb) for bb in norm_list]

    # -------- signature-aware caller --------
    def _gen_positional_calls(fn):
        try:
            sig = inspect.signature(fn)
        except Exception:
            return []
        params = [
            p
            for p in sig.parameters.values()
            if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
        ]
        if params and params[0].name in ("self",):
            params = params[1:]
        n = len(params)
        calls = []
        if n >= 1:
            calls.append((("DICT_BATCH",), {}, "pos:dict_batch"))
        if n >= 2:
            calls.append((("IMAGES", "TEXTS"), {}, "pos:images_texts"))
        if n >= 1:
            calls.append((("IMAGES",), {}, "pos:images_only"))
        if n >= 1:
            calls.append((("TEXTS",), {}, "pos:texts_only"))
        return calls

    def _normalize_out(
        out, B: Optional[int] = None
    ) -> Optional[List[Optional[List[float]]]]:
        if out is None:
            return None

        def _from_dict(d: Dict[str, Any]) -> Optional[List[Optional[List[float]]]]:
            for k in (
                "bbox",
                "pred_bbox",
                "pred_box",
                "pred_boxes",
                "boxes",
                "boxes_xyxy",
                "boxes_xyxy_norm",
                "xyxy",
                "box",
                "box_xyxy_norm",
                "output_bbox",
                "target_bbox",
                "ref_bbox",
                "region",
                "regions",
                "norm_bbox",
                "bbox_norm",
                "bbox_xywh",
                "xywh",
                "boxes_xywh",
            ):
                if k in d:
                    v = d[k]
                    if torch.is_tensor(v) and v.dim() == 2 and v.shape[-1] == 4:
                        return [
                            [float(x) for x in row.tolist()] for row in v.detach().cpu()
                        ]
                    if isinstance(v, (list, tuple)) and len(v) > 0:
                        l4 = _to_list4(v)
                        if l4 is not None:
                            return [l4]
                        if _to_list4(v[0]) is not None:
                            return [_to_list4(row) for row in v]
                        if isinstance(v[0], dict):
                            return [_box_from_any(e) for e in v]
                    bb = _box_from_any(v)
                    if bb is not None:
                        return [bb]

            for k in ("xywh", "box_xywh", "pred_xywh"):
                if k in d:
                    v = d[k]
                    if torch.is_tensor(v) and v.dim() == 2 and v.shape[-1] == 4:
                        return [
                            [x, y, x + w, y + h]
                            for (x, y, w, h) in v.detach().cpu().tolist()
                        ]
                    bb = _box_from_any(v)
                    if bb is not None:
                        return [_xywh_to_xyxy(bb)]

            for k in (
                "preds",
                "predictions",
                "outputs",
                "results",
                "instances",
                "phrases",
                "detections",
                "result",
            ):
                if k in d:
                    arr = d[k]
                    if isinstance(arr, (list, tuple)):
                        return [_box_from_any(e) for e in arr]

            for k in ("output", "out", "data"):
                if k in d and isinstance(d[k], dict):
                    got = _from_dict(d[k])
                    if got is not None:
                        return got

            for k in d:
                vv = d[k]
                if (
                    isinstance(vv, (list, tuple))
                    and len(vv) > 0
                    and isinstance(vv[0], dict)
                ):
                    cand = [_box_from_any(e) for e in vv]
                    if any(x is not None for x in cand):
                        return cand
            return None

        # ❶ prefer_attn が有効なら、dict の場合は bbox より先に attn を使う
        if isinstance(out, dict) and prefer_attn:
            for k in ("attn_maps", "attn_logits"):
                if k in out and torch.is_tensor(out[k]) and out[k].dim() == 4:
                    am = (
                        torch.sigmoid(out["attn_logits"])
                        if k == "attn_logits"
                        else out["attn_maps"]
                    )
                    bb_from_attn = _bbox_from_attn_tensor(am)
                    if bb_from_attn:
                        return bb_from_attn

        if isinstance(out, dict):
            got = _from_dict(out)
            if got is not None:
                return got
            # --- attention heatmap → bbox 救済 ---
            for k in ("attn_maps", "attn_logits"):
                if k in out and torch.is_tensor(out[k]) and out[k].dim() == 4:
                    am = torch.sigmoid(out[k]) if k == "attn_logits" else out[k]
                    bb_from_attn = _bbox_from_attn_tensor(am)
                    if bb_from_attn:
                        return bb_from_attn

        if isinstance(out, (list, tuple)):
            if len(out) == 4 and _to_list4(out) is not None:
                return [_to_list4(out)]
            return [_box_from_any(e) for e in out]

        if torch.is_tensor(out):
            if out.dim() == 2 and out.shape[-1] == 4:
                return [[float(x) for x in row.tolist()] for row in out.detach().cpu()]
            if out.numel() == 4:
                return [[float(x) for x in out.detach().cpu().flatten().tolist()]]

        if isinstance(out, str):
            bb = _box_from_text(out)
            if bb:
                return [bb]

        return None

    def _attempts(model_like, images, texts, out_dir, tag):
        attempts_rec = []
        tokens = _mk_tokens(texts)

        methods = []
        for name in ("refcoco_infer", "predict", "generate", "forward", "__call__"):
            if hasattr(model_like, name) and callable(getattr(model_like, name)):
                methods.append((name, getattr(model_like, name)))

        ctx = (
            torch.amp.autocast("cuda", enabled=amp_enabled)
            if images.is_cuda
            else nullcontext()
        )
        with torch.no_grad(), ctx:
            for name, fn in methods:
                for args_pat, kwargs, mtag in _gen_positional_calls(fn):
                    try:
                        args = []
                        for token in args_pat:
                            if token == "IMAGES":
                                args.append(images)
                            elif token == "TEXTS":
                                args.append(texts)
                            elif token == "DICT_BATCH":
                                batch_dict: Dict[str, Any] = {
                                    "image": images,
                                    "images": images,
                                    "text": texts,
                                    "texts": texts,
                                    "prompts": texts,
                                    "query": texts,
                                    "queries": texts,
                                    "ref": texts,
                                    "ref_text": texts,
                                    "ref_expr": texts,
                                    "refexpr": texts,
                                    "sentence": texts,
                                    "caption": texts,
                                    "task": "refcoco",
                                    "inference": True,
                                    "return_bbox": True,
                                    "return_attn": True,
                                    "return_attn_maps": True,
                                    "return_attn_logits": True,
                                    "compute_lm": False,
                                }
                                if tokens:
                                    for k, v in tokens.items():
                                        batch_dict[k] = v
                                args.append(batch_dict)
                            else:
                                args.append(token)
                        out = fn(*args, **kwargs)
                        attempts_rec.append(
                            {"method": name, "mode": mtag, "status": "ok"}
                        )
                        if not first_batch_marks["dumped"]:
                            _dump_json(
                                {"call": f"{name}/{mtag}", "raw": _tensor_to_py(out)},
                                os.path.join(
                                    out_dir, f"debug_first_batch_raw_{tag}.json"
                                ),
                            )
                            first_batch_marks["dumped"] = True
                        norm = _normalize_out(out, B=images.shape[0])
                        if norm is not None:
                            norm = _post_norm_and_scale(norm)
                            _append_line(
                                os.path.join(out_dir, "debug_first_batch_calls.log"),
                                f"[{tag}] matched: {name} ({mtag})",
                            )
                            return norm, attempts_rec
                    except Exception as e:
                        attempts_rec.append(
                            {
                                "method": name,
                                "mode": mtag,
                                "status": "err",
                                "msg": str(e),
                            }
                        )
                        if very_verbose:
                            _append_line(
                                os.path.join(out_dir, "debug_first_batch_calls.log"),
                                f"[{tag}] {name} ({mtag}) failed: {e}",
                            )

                kwargs_candidates: List[Dict[str, Any]] = []
                if tokens:
                    kwargs_candidates += [
                        {
                            "samples": {
                                "image": images,
                                **tokens,
                                "return_attn": True,
                                "return_attn_maps": True,
                                "return_attn_logits": True,
                                "return_bbox": True,
                            }
                        },
                        {
                            "samples": {
                                "images": images,
                                **tokens,
                                "return_attn": True,
                                "return_attn_maps": True,
                                "return_attn_logits": True,
                                "return_bbox": True,
                            }
                        },
                    ]
                kwargs_candidates += [
                    {
                        "samples": {
                            "image": images,
                            "text": texts,
                            "return_attn": True,
                            "return_attn_maps": True,
                            "return_attn_logits": True,
                            "return_bbox": True,
                            "compute_lm": False,
                        }
                    },
                    {
                        "samples": {
                            "images": images,
                            "prompts": texts,
                            "return_attn": True,
                            "return_attn_maps": True,
                            "return_attn_logits": True,
                            "return_bbox": True,
                            "compute_lm": False,
                        }
                    },
                    {
                        "image": images,
                        "text": texts,
                        "return_attn": True,
                        "return_attn_maps": True,
                        "return_attn_logits": True,
                        "return_bbox": True,
                        "compute_lm": False,
                    },
                    {
                        "images": images,
                        "text": texts,
                        "return_attn": True,
                        "return_attn_maps": True,
                        "return_attn_logits": True,
                        "return_bbox": True,
                        "compute_lm": False,
                    },
                    {
                        "image": images,
                        "prompts": texts,
                        "return_attn": True,
                        "return_attn_maps": True,
                        "return_attn_logits": True,
                        "return_bbox": True,
                        "compute_lm": False,
                    },
                    {
                        "images": images,
                        "prompts": texts,
                        "return_attn": True,
                        "return_attn_maps": True,
                        "return_attn_logits": True,
                        "return_bbox": True,
                        "compute_lm": False,
                    },
                ]
                for kw in kwargs_candidates:
                    try:
                        out = fn(**kw)
                        attempts_rec.append(
                            {
                                "method": name,
                                "mode": "kwargs",
                                "status": "ok",
                                "keys": list(kw.keys()),
                            }
                        )
                        if not first_batch_marks["dumped"]:
                            _dump_json(
                                {
                                    "call": f"{name}/kwargs",
                                    "kwargs_keys": list(kw.keys()),
                                    "raw": _tensor_to_py(out),
                                },
                                os.path.join(
                                    out_dir, f"debug_first_batch_raw_{tag}.json"
                                ),
                            )
                            first_batch_marks["dumped"] = True
                        norm = _normalize_out(out, B=images.shape[0])
                        if norm is not None:
                            norm = _post_norm_and_scale(norm)
                            _append_line(
                                os.path.join(out_dir, "debug_first_batch_calls.log"),
                                f"[{tag}] matched: {name} kwargs={list(kw.keys())}",
                            )
                            return norm, attempts_rec
                    except Exception as e:
                        attempts_rec.append(
                            {
                                "method": name,
                                "mode": "kwargs",
                                "status": "err",
                                "keys": list(kw.keys()),
                                "msg": str(e),
                            }
                        )
                        if very_verbose:
                            _append_line(
                                os.path.join(out_dir, "debug_first_batch_calls.log"),
                                f"[{tag}] {name} kwargs={list(kw.keys())} failed: {e}",
                            )

        if not attempts_log_written["done"]:
            _dump_json(
                attempts_rec,
                os.path.join(out_dir, f"debug_first_batch_attempts_{tag}.json"),
            )
            attempts_log_written["done"] = True
        return None, attempts_rec

    def _infer_adapter(image: torch.Tensor, text: List[str] | str):
        images = _ensure_4d(image).to(device, non_blocking=True)
        texts = _as_str_list(text)

        out_dir = cfg.get("eval", {}).get("out_dir", "outputs/phase1p5_eval")
        os.makedirs(out_dir, exist_ok=True)

        if bool(cfg.get("eval", {}).get("force_simple_prompt", False)):
            texts = [
                (t.split("Query:", 1)[-1] if "Query:" in t else t).strip()
                for t in texts
            ]

        ref_only = [
            (t.split("Query:", 1)[-1] if "Query:" in t else t).strip() for t in texts
        ]
        try:
            with open(
                os.path.join(out_dir, "debug_first_batch_prompts_refonly.txt"),
                "w",
                encoding="utf-8",
            ) as f:
                for i, tt in enumerate(ref_only[:64]):
                    f.write(f"[{i}] {tt}\n\n")
        except Exception:
            pass

        preds, _ = _attempts(model, images, texts, out_dir, tag="withicl")

        need_retry = (preds is None) or (
            sum(1 for p in preds if isinstance(p, (list, tuple)) and len(p) == 4)
            < max(1, int(0.25 * len(texts)))
        )
        if need_retry:
            preds2, _ = _attempts(model, images, ref_only, out_dir, tag="refonly")
            if preds2 is not None:
                preds = preds2

        # ★ prefer_attn が有効で、ここまでで bbox 正常化に失敗した/微妙な場合は
        #    「attn だけを意図的に取りに行く」再呼び出しを一度だけ試す
        if prefer_attn:

            def _has_good(pred_list):
                if pred_list is None:
                    return False
                ok = [
                    p for p in pred_list if isinstance(p, (list, tuple)) and len(p) == 4
                ]
                return len(ok) >= max(1, int(0.25 * len(texts)))

            if not _has_good(preds):
                # return_bbox を外し、attn 返却フラグを立てて再要求
                try:
                    callspec = {
                        "samples": {
                            "image": images,
                            "text": texts,
                            "return_attn": True,
                            "return_attn_maps": True,
                            "return_attn_logits": True,
                            "compute_lm": False,
                        }
                    }
                    out_attn = getattr(model, "forward", model)(**callspec)
                    norm = _normalize_out(out_attn, B=images.shape[0])
                    if norm is not None:
                        preds = _post_norm_and_scale(norm)
                        _append_line(
                            os.path.join(out_dir, "debug_first_batch_calls.log"),
                            "[prefer_attn] second-shot attn-only succeeded",
                        )
                except Exception as e:
                    _append_line(
                        os.path.join(out_dir, "debug_first_batch_calls.log"),
                        f"[prefer_attn] attn-only retry failed: {e}",
                    )

        # ---------- サリエンシー置換（巨大箱・無効救済） ----------
        area_cap = float(cfg.get("eval", {}).get("area_cap", 0.85))

        if preds is not None:
            fixed = []
            replace_mask = []
            for bb in preds:
                ok = isinstance(bb, (list, tuple)) and len(bb) == 4
                if not ok:
                    replace_mask.append(True)
                    fixed.append(None)
                    continue
                x1, y1, x2, y2 = [float(v) for v in bb]
                w = max(0.0, min(1.0, x2) - max(0.0, x1))
                h = max(0.0, min(1.0, y2) - max(0.0, y1))
                # 面積が閾値超えの場合の置換は、use_saliency_fallback が True のときのみ行う
                if use_saliency_fallback and (w * h) >= area_cap:
                    replace_mask.append(True)
                    fixed.append(None)
                else:
                    replace_mask.append(False)
                    fixed.append([x1, y1, x2, y2])

            if any(replace_mask):
                if use_saliency_fallback:
                    sal_boxes = _bbox_from_saliency_tensor(images)
                    sb_i = 0
                    for i, need in enumerate(replace_mask):
                        if need:
                            fixed[i] = sal_boxes[sb_i]
                            sb_i += 1
                else:
                    # 救済OFF時は、そのまま（Noneは全画面を最小限でカバー）
                    for i, need in enumerate(replace_mask):
                        if need:
                            fixed[i] = [0.0, 0.0, 1.0, 1.0]
                preds = fixed

            return {"bbox": preds}

        # per-sample fallback + 置換
        bbs = []
        for i in range(images.shape[0]):
            pi, _ = _attempts(
                model, images[i].unsqueeze(0), [ref_only[i]], out_dir, tag=f"single_{i}"
            )
            bb = pi[0] if pi else None
            if isinstance(bb, (list, tuple)) and len(bb) == 4:
                x1, y1, x2, y2 = [float(v) for v in bb]
                w = max(0.0, min(1.0, x2) - max(0.0, x1))
                h = max(0.0, min(1.0, y2) - max(0.0, y1))
                # 面積が大きすぎる場合の置換はオプション
                if use_saliency_fallback and (w * h) >= area_cap:
                    bb = _bbox_from_saliency_tensor(images[i].unsqueeze(0))[0]
            else:
                bb = (
                    _bbox_from_saliency_tensor(images[i].unsqueeze(0))[0]
                    if use_saliency_fallback
                    else [0.0, 0.0, 1.0, 1.0]
                )
            bbs.append(bb)
        return {"bbox": bbs}

    model.infer = _infer_adapter  # type: ignore


# --------------------------------
# load model
# --------------------------------
def load_model_from_cfg(cfg: dict) -> nn.Module:
    build_model, origin = _load_build_model_function(REPO_ROOT)
    print(f"[info] build_model loaded from {origin}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg, device=device)
    model.to(device)
    resume_path = cfg.get("resume")
    _robust_load_weights(model, resume_path, device)
    attach_infer_adapter(model, cfg)
    model.eval()
    return model


# --------------------------------
# ICL builder
# --------------------------------
def build_icl_builder(cfg: dict):
    icl_cfg = cfg.get("icl", {})
    bank = icl_cfg.get("bank_jsonl") or icl_cfg.get("bank_path")
    tmpl = icl_cfg.get("slot_template_path") or icl_cfg.get("template")
    if ExternalICLBuilder is not None:
        return ExternalICLBuilder(bank_jsonl=bank, slot_template_path=tmpl)
    return ICLPromptBuilder(bank_jsonl=bank, slot_template_path=tmpl)


_SLOT_DEFAULTS = {
    "part": "object",
    "defect": "defect",
    "color": "unknown color",
    "shape": "unknown shape",
    "texture": "unknown texture",
    "position": "",
    "size": "",
    "position_size": "an unspecified location",
}


@lru_cache(maxsize=8)
def _load_slot_template(path: Optional[str]) -> Optional[str]:
    if not path:
        return None
    p = Path(path)
    if not p.is_file():
        p = Path(REPO_ROOT) / path
        if not p.is_file():
            return None
    try:
        return p.read_text(encoding="utf-8").strip()
    except OSError:
        return None


def _normalize_slot_value(val: Any) -> Optional[str]:
    if val is None:
        return None
    if isinstance(val, (list, tuple, set)):
        items = [str(v).strip() for v in val if isinstance(v, (str, int, float))]
        items = [v for v in items if v]
        return ", ".join(items) if items else None
    s = str(val).strip()
    return s if s else None


def _format_slot_prompt(
    slots: Optional[Dict[str, Any]], tmpl: str
) -> Tuple[Optional[str], bool]:
    merged: Dict[str, str] = dict(_SLOT_DEFAULTS)
    provided = False

    slots = slots or {}
    for key, value in slots.items():
        k = str(key).lower()
        v = _normalize_slot_value(value)
        if not v:
            continue
        if k in merged:
            merged[k] = v
            if v != _SLOT_DEFAULTS.get(k, ""):
                provided = True
        elif k in ("position", "size"):
            merged[k] = v
            provided = True

    pos_size = _normalize_slot_value(slots.get("position_size"))
    if not pos_size:
        pieces = []
        for key in ("position", "size"):
            v = _normalize_slot_value(slots.get(key))
            if v:
                pieces.append(v)
        if pieces:
            pos_size = ", ".join(pieces)

    if pos_size:
        merged["position_size"] = pos_size
        if pos_size != _SLOT_DEFAULTS["position_size"]:
            provided = True

    try:
        formatted = tmpl.format(
            part=merged["part"],
            defect=merged["defect"],
            color=merged["color"],
            shape=merged["shape"],
            texture=merged["texture"],
            position_size=merged["position_size"],
        ).strip()
    except KeyError:
        return None, provided

    return (formatted if formatted else None), provided


def build_query_text(
    slots: Optional[Dict[str, str]], tmpl_path: Optional[str], ref_text: str
) -> str:
    ref_text = (ref_text or "").strip()
    tmpl = _load_slot_template(tmpl_path)
    formatted, provided = _format_slot_prompt(slots, tmpl) if tmpl else (None, False)

    if provided and formatted:
        return formatted
    if ref_text:
        return ref_text
    if formatted:
        return formatted
    return "Locate the referred object."


# --------------------------------
# main
# --------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", type=str, required=True)
    ap.add_argument("--icl_k", type=int, default=None)
    ap.add_argument("--save_dir", type=str, default=None)
    ap.add_argument("--save_metrics_csv", type=str, default=None)
    ap.add_argument("--save_json", type=str, default=None)
    args = ap.parse_args()

    with open(args.cfg, "r") as f:
        cfg = yaml.safe_load(f)

    cfg.setdefault("eval", {}).setdefault("debug", True)

    if args.icl_k is not None:
        cfg.setdefault("icl", {})["k"] = int(args.icl_k)
    if args.save_dir:
        cfg.setdefault("eval", {})["out_dir"] = args.save_dir

    seed = int(cfg.get("seed", 42))
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    out_dir = cfg.get("eval", {}).get("out_dir", "outputs/phase1p5_eval")
    os.makedirs(out_dir, exist_ok=True)

    split = cfg.get("eval", {}).get("dataset", "val")
    ds = RefCOCOJsonDataset(cfg, split=split)
    bs = int(cfg.get("eval", {}).get("batch_size", 1))
    dl = DataLoader(
        ds, batch_size=bs, shuffle=False, num_workers=0, collate_fn=collate_batch
    )

    builder = build_icl_builder(cfg)
    icl_k = int(cfg.get("icl", {}).get("k", 0))
    slot_tmpl_path = cfg.get("icl", {}).get("slot_template_path")

    model = load_model_from_cfg(cfg)
    device = next(model.parameters()).device

    total = 0
    json_ok = 0
    ious_05: List[float] = []
    ious_all: List[float] = []

    jsonl_path = args.save_json or os.path.join(out_dir, "epoch_val.jsonl")
    csv_path = args.save_metrics_csv or os.path.join(out_dir, "epoch_val.csv")
    os.makedirs(os.path.dirname(jsonl_path), exist_ok=True)
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    jf = open(jsonl_path, "w", encoding="utf-8")

    limit_batches = int(cfg.get("eval", {}).get("limit_batches", 0))
    pbar = tqdm(
        dl, total=len(dl) if limit_batches <= 0 else min(len(dl), limit_batches)
    )
    first_batch = True

    for bi, batch in enumerate(pbar):
        if limit_batches > 0 and bi >= limit_batches:
            break

        imgs: torch.Tensor = batch["image"].to(device, non_blocking=True)
        gtb = batch["gt_bbox"].cpu().tolist()
        ids = batch["id"]

        # prompts (ICL on)
        prompts: List[str] = []
        for s, ref in zip(batch["slots"], batch["ref_expr"]):
            q = build_query_text(slots=s, tmpl_path=slot_tmpl_path, ref_text=ref)
            p = builder.build(k=icl_k, query_text=q, slots=s)
            prompts.append(p)

        # Save first-batch prompts (ICL)
        if first_batch:
            try:
                with open(
                    os.path.join(out_dir, "debug_first_batch_prompts.txt"),
                    "w",
                    encoding="utf-8",
                ) as f:
                    for i, pp in enumerate(prompts[:64]):
                        f.write(f"[{i}] {pp}\n\n")
            except Exception:
                pass

        out = model.infer(image=imgs, text=prompts)
        bbs: List[Optional[List[float]]] = (
            out.get("bbox") if isinstance(out, dict) else None
        )

        if first_batch and cfg.get("eval", {}).get("debug", True):
            try:
                print("[debug] adapter return type:", type(out))
                if isinstance(out, dict):
                    print("[debug] adapter keys:", list(out.keys())[:20])
                    if "bbox" in out:
                        pv = out["bbox"]
                        if isinstance(pv, (list, tuple)):
                            print("[debug] bbox types:", [type(e) for e in pv[:8]])
                            print("[debug] bbox[0]:", str(pv[0])[:300])
                        else:
                            print("[debug] bbox type:", type(pv))
                else:
                    print("[debug] adapter preview:", str(out)[:400])
            except Exception:
                pass
            first_batch = False

        if bbs is None or len(bbs) != imgs.shape[0]:
            if cfg.get("eval", {}).get("debug", True):
                print(
                    f"[warn] invalid bbs. per-sample fallback. bbs_len={0 if bbs is None else len(bbs)} B={imgs.shape[0]}"
                )
            bbs = []
            for i in range(imgs.shape[0]):
                oi = model.infer(image=imgs[i], text=prompts[i])
                bb = None
                if isinstance(oi, dict) and "bbox" in oi:
                    x = oi["bbox"]
                    if torch.is_tensor(x) and x.numel() == 4:
                        bb = x.detach().cpu().flatten().tolist()
                    elif isinstance(x, list) and len(x) == 4:
                        bb = list(map(float, x))
                    elif isinstance(x, str):
                        nums = re.findall(r"[-+]?(?:\d+\.\d+|\d+)", x)
                        if len(nums) >= 4:
                            bb = list(map(float, nums[:4]))
                elif torch.is_tensor(oi) and oi.numel() == 4:
                    bb = oi.detach().cpu().flatten().tolist()
                elif isinstance(oi, (list, tuple)) and len(oi) == 4:
                    bb = list(map(float, oi))
                bbs.append(bb)

        for i_, (sid, gt, bb, pr) in enumerate(zip(ids, gtb, bbs, prompts)):
            total += 1
            ok = (
                isinstance(bb, (list, tuple))
                and len(bb) == 4
                and all(isinstance(v, (int, float)) for v in bb)
            )
            if ok:
                x1, y1, x2, y2 = [min(1.0, max(0.0, float(v))) for v in bb]
                if x2 < x1:
                    x1, x2 = x2, x1
                if y2 < y1:
                    y1, y2 = y2, y1
                bb = [x1, y1, x2, y2]
            json_ok += int(ok)
            iou = iou_xyxy_norm(bb, gt) if ok else 0.0
            ious_all.append(iou)
            if iou >= 0.5:
                ious_05.append(1.0)
            rec = {
                "id": int(sid),
                "gt_bbox": gt,
                "pred_bbox": bb,
                "prompt": pr,
                "iou": iou,
            }
            jf.write(json.dumps(rec, ensure_ascii=False) + "\n")

            # --- 予測/GTの簡易可視化を少数だけ保存 ---
            if bi < 3 and i_ < 8:  # 先頭 ~24 サンプルだけ
                try:
                    import numpy as np
                    import cv2

                    save_dir = os.path.join(out_dir, "viz")
                    os.makedirs(save_dir, exist_ok=True)
                    # 元の正規化前画像を復元（letterbox後の正方形）
                    mean = np.array([0.48145466, 0.4578275, 0.40821073]).reshape(
                        3, 1, 1
                    )
                    std = np.array([0.26862954, 0.26130258, 0.27577711]).reshape(
                        3, 1, 1
                    )
                    im = imgs[i_].detach().cpu().float().numpy()
                    im = (im * std + mean).clip(0, 1)
                    im = (im.transpose(1, 2, 0) * 255).astype(np.uint8)
                    H, W = im.shape[:2]

                    def denorm_xyxy(nbb):
                        x1 = int(nbb[0] * W)
                        y1 = int(nbb[1] * H)
                        x2 = int(nbb[2] * W)
                        y2 = int(nbb[3] * H)
                        return x1, y1, x2, y2

                    canvas = im.copy()
                    # GT(緑) / Pred(青)
                    x1, y1, x2, y2 = denorm_xyxy(gt)
                    cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    if isinstance(bb, (list, tuple)) and len(bb) == 4:
                        px1, py1, px2, py2 = denorm_xyxy(bb)
                        cv2.rectangle(canvas, (px1, py1), (px2, py2), (255, 0, 0), 2)
                    cv2.putText(
                        canvas,
                        f"IOU {iou:.3f}",
                        (8, 24),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 255, 255),
                        2,
                    )
                    cv2.imwrite(
                        os.path.join(save_dir, f"bi{bi:03d}_{i_:02d}_id{int(sid)}.jpg"),
                        cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR),
                    )
                except Exception as _e:
                    pass

        mean_iou = sum(ious_all) / max(1, len(ious_all))
        pbar.set_description(f"miou={mean_iou:.4f}  ok={json_ok}/{total}")

    jf.close()
    iou_mean = sum(ious_all) / max(1, len(ious_all))
    iou_05 = sum(ious_05) / max(1, len(ious_all))
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("metric,value\n")
        f.write(f"iou@mean,{iou_mean:.6f}\n")
        f.write(f"iou@0.5,{iou_05:.6f}\n")
        f.write(f"json_ok_rate,{json_ok/max(1,total):.6f}\n")
    print(
        f"[Done] K={icl_k}  iou@0.5={iou_05:.4f}  iou@mean={iou_mean:.4f}  json_ok_rate={json_ok/max(1,total):.3f}"
    )


if __name__ == "__main__":
    main()
