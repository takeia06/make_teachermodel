# datasets/refcoco/refcoco_dataset_inst.py
import os, json, random
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from PIL import Image, ImageDraw

# --- Pillow 互換（Resampling がない旧版でも動くように）---
_RESAMPLING = getattr(Image, "Resampling", None)
_BICUBIC    = _RESAMPLING.BICUBIC if _RESAMPLING else Image.BICUBIC
_NEAREST    = _RESAMPLING.NEAREST if _RESAMPLING else Image.NEAREST
try:
    # COCO RLE を扱う公式実装
    from pycocotools import mask as coco_mask
    _HAS_PYCOCO = True
except Exception:
    _HAS_PYCOCO = False

import torch
from torch.utils.data import Dataset
from torchvision import transforms

_CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
_CLIP_STD  = (0.26862954, 0.26130258, 0.27577711)
_PATCH = 14 #ViT-*/14

# -----------------------------
# helpers: polygons / masks
# -----------------------------
def _seg_to_polys(seg: Union[List[float], List[List[float]]]) -> List[List[Tuple[float, float]]]:
    polys: List[List[Tuple[float, float]]] = []
    if not seg:
        return polys
    if isinstance(seg[0], list):
        for s in seg:
            if not isinstance(s, list):
                continue
            n = (len(s) // 2) * 2
            if n < 6:
                continue
            pts = [(float(s[i]), float(s[i + 1])) for i in range(0, n, 2)]
            polys.append(pts)
    else:
        n = (len(seg) // 2) * 2
        if n >= 6:
            s = seg[:n]
            pts = [(float(s[i]), float(s[i + 1])) for i in range(0, n, 2)]
            polys.append(pts)
    return polys

def _clip_poly(poly: List[Tuple[float, float]], w: int, h: int) -> List[Tuple[float, float]]:
    out = []
    for x, y in poly:
        xx = float(np.clip(x, 0, max(0, w - 1)))
        yy = float(np.clip(y, 0, max(0, h - 1)))
        out.append((xx, yy))
    return out

def _poly_mask(w: int, h: int, polys: List[List[Tuple[float, float]]]) -> Image.Image:
    m = Image.new("L", (w, h), 0)
    if not polys:
        return m
    d = ImageDraw.Draw(m)
    for p in polys:
        if len(p) < 3:
            continue
        d.polygon(_clip_poly(p, w, h), outline=255, fill=255)
    return m

def _rle_to_mask(rle_obj: Any) -> Optional[Image.Image]:
    """
    COCO RLE / compressed RLE を Pillow マスク(L)に変換。
    rle_obj: {"size":[h,w], "counts": ...} または list[RLE] などを許容。
    """
    if not _HAS_PYCOCO:
        return None
    try:
        # 単一/複数 RLE の両方に対応
        if isinstance(rle_obj, dict) and "counts" in rle_obj and "size" in rle_obj:
            rle = rle_obj
        elif isinstance(rle_obj, list) and len(rle_obj) > 0:
            # 複数RLE → 合成
            rle = coco_mask.merge(rle_obj, intersect=False)
        else:
            return None
        m = coco_mask.decode(rle)  # (H, W) uint8/uint32
        if m is None:
            return None
        if m.ndim == 3:
            # (H, W, N) の場合は max でまとめる
            import numpy as np
            m = (m.max(axis=2) > 0).astype("uint8")
        from PIL import Image
        return Image.fromarray((m.astype("uint8") * 255), mode="L")
    except Exception:
        return None

def _bbox_mask(w: int, h: int, box: Union[List[float], Tuple[float, float, float, float], None]) -> Image.Image:
    if not box:
        return Image.new("L", (w, h), 0)
    x, y, bw, bh = [float(v) for v in box]
    x0 = float(np.clip(x, 0, max(0, w - 1)))
    y0 = float(np.clip(y, 0, max(0, h - 1)))
    x1 = float(np.clip(x + bw, 0, max(0, w - 1)))
    y1 = float(np.clip(y + bh, 0, max(0, h - 1)))
    if x1 <= x0 or y1 <= y0:
        return Image.new("L", (w, h), 0)
    m = Image.new("L", (w, h), 0)
    ImageDraw.Draw(m).rectangle([x0, y0, x1, y1], outline=255, fill=255)
    return m

# -----------------------------
# helpers: json loading / path
# -----------------------------
def _is_json_array(file_head: str) -> bool:
    return file_head.lstrip().startswith("[")

def _load_records_any_one(json_path: str) -> List[dict]:
    with open(json_path, "r") as f:
        head = f.read(2048)
        f.seek(0)
        if _is_json_array(head):
            data = json.load(f)
            if not isinstance(data, list):
                raise ValueError(f"{json_path} should be a list JSON.")
            return data
        records = []
        for line in f:
            s = line.strip()
            if not s:
                continue
            records.append(json.loads(s))
        return records

def _resolve_img_path(it: Dict[str, Any], img_root: Optional[Union[str, Dict[str, str]]]) -> str:
    """
    画像パスの解決。image_path(絶対/相対) を最優先。
    file_name の場合は COCO の split 推定や orig_split を使い、img_root(dict) のエイリアスも吸収。
    """
    p = it.get("image_path")
    if isinstance(p, str) and p:
        if os.path.isabs(p):
            return p
        if isinstance(img_root, str):
            return os.path.join(img_root, p)

    fname = it.get("file_name") or (os.path.basename(p) if p else None)
    if not isinstance(fname, str):
        raise FileNotFoundError("Record must have image_path or file_name.")

    # split 推定（orig_split > file_nameの含意）
    split = (it.get("orig_split") or
             ("train2014" if "train2014" in fname.lower() else
              ("val2014" if "val2014" in fname.lower() else "train2014"))).lower()

    # img_root が dict の場合のエイリアスマップ
    if isinstance(img_root, dict):
        if split in ("train", "train2014"):
            key = "train2014"
        elif split in ("val", "val2014"):
            key = "val2014"
        elif split in ("test", "test2014", "test2017"):
            key = "val2014"  # 無いことが多いので val に寄せる
        else:
            key = "default"
        root = img_root.get(key) or img_root.get("default") or ""
        return os.path.join(root, fname)

    # 文字列パスのとき
    if isinstance(img_root, str):
        sub = "train2014" if "train2014" in fname.lower() else ("val2014" if "val2014" in fname.lower() else "")
        return os.path.join(img_root, sub, fname) if sub else os.path.join(img_root, fname)

    return fname

# -----------------------------
# Dataset
# -----------------------------
class RefCOCODataset(Dataset):
    """
    RefCOCO系（RefCOCO/+/g）をまとめて扱うための Dataset 実装。
    - 監督: segmentation(優先) → bbox → ゼロマスク
    - テキスト: "text" or "answer"(配列) から選択（use_answers）
    - ③対応: マスクは「オリジナル解像度」で生成→最終画像サイズへリサイズ（座標系の厳密一致）
    """
    def __init__(
        self,
        json_path: Union[str, List[str]],
        img_root: Optional[Union[str, Dict[str, str]]] = None,
        mask_root: Optional[str] = None,  # 未使用（将来の拡張用）
        split: str = "train",
        long_side: int = 896,
        supervision: str = "mask",
        soft_cfg: Optional[dict] = None,  # 未使用（将来の拡張用）
        augment: Optional[dict] = None,
        text_key: str = "text",
        answer_key: str = "answer",
        use_answers: str = "expand",      # "expand" | "random" | "first"
        sample_ratios: Optional[List[float]] = None,
        mixture_weights: Optional[List[float]] = None,  # 未使用（将来の拡張用）
        seed: int = 42,
        # --- 小規模検証 / 過学習テスト補助用 ---
        limit_items: Optional[int] = None,          # 全体から先頭N件
        limit_per_source: Optional[int] = None,     # 各JSONごとの上限
        shuffle: bool = True,                       # マージ後にシャッフル
        strict_exist: bool = False,                 # 画像実在チェックで欠損除外
    ):
        super().__init__()
        rng = random.Random(seed)
        self.split = split
        self.long_side = int(long_side)
        self.patch = _PATCH  # ViT-*/14 前提（必要ならcfgから受け取るよう拡張可）
        self.supervision = supervision
        self.text_key = text_key
        self.answer_key = answer_key
        self.use_answers = (use_answers or "expand").lower().strip()
        self.img_root = img_root
        self.aug = augment or {}
        # 追加引数をインスタンスに保持（必要ならどこでも参照できる）
        self.limit_items = limit_items
        self.limit_per_source = limit_per_source
        self.shuffle = shuffle
        self.strict_exist = strict_exist

        # === train/val/test で Aug 適用を切り替えるためのフラグ ===
        _sp = (self.split or "").lower()
        self._is_train_split = (_sp in ("train", "tr", "training"))
        # それ以外（"val", "valid", "test" など）は False（= Aug 無効）

        # 複数json可
        paths = [json_path] if isinstance(json_path, str) else list(json_path)
        if len(paths) == 0:
            raise FileNotFoundError("json_path is empty.")

        raw_sources: List[List[Dict[str, Any]]] = []
        for p in paths:
            items = _load_records_any_one(p)
            raw_sources.append(items)

        # サンプリング（任意）
        if sample_ratios:
            assert len(sample_ratios) == len(raw_sources)
            for i, r in enumerate(sample_ratios):
                if r is None or r >= 1.0:
                    continue
                src = raw_sources[i]
                k = max(1, int(len(src) * float(r)))
                raw_sources[i] = rng.sample(src, k)

        # 各ソースごとの上限（小規模データでの素早い検証に便利）
        if self.limit_per_source is not None and int(self.limit_per_source) > 0:
            n = int(self.limit_per_source)
            for i, src in enumerate(raw_sources):
                raw_sources[i] = src[:n]

        # answers の展開or選択
        expanded_by_src: List[List[Dict[str, Any]]] = []
        for src in raw_sources:
            ex: List[Dict[str, Any]] = []
            if self.use_answers == "expand":
                for it in src:
                    it = dict(it)
                    it["image_path"] = _resolve_img_path(it, self.img_root)
                    answers = it.get(self.answer_key) or it.get(self.answer_key + "s")
                    if isinstance(answers, list) and len(answers) > 0:
                        for a in answers:
                            dit = dict(it)
                            dit["__text_from__"] = "answer"
                            dit[self.text_key] = str(a)
                            ex.append(dit)
                    else:
                        it["__text_from__"] = "text"
                        it[self.text_key] = str(it.get(self.text_key, ""))
                        ex.append(it)
            else:
                for it in src:
                    dit = dict(it)
                    dit["image_path"] = _resolve_img_path(dit, self.img_root)
                    ex.append(dit)
            expanded_by_src.append(ex)

        merged = [it for src in expanded_by_src for it in src]
        # 画像の実在チェック（必要時のみ）
        if strict_exist:
            merged = [it for it in merged if isinstance(it.get("image_path"), str) and os.path.exists(it["image_path"])]
        # シャッフル（reproducible）
        if self.shuffle:
            rng.shuffle(merged)

        # 全体上限（小規模検証用）
        if self.limit_items is not None and int(self.limit_items) > 0:
            merged = merged[: int(self.limit_items)]
        if len(merged) == 0:
            raise FileNotFoundError("No records after merge/expand.")

        self.items = merged

        # 画像前処理（OpenCLIPに合わせる）
        self.tf_img = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(_CLIP_MEAN, _CLIP_STD)
        ])

        # Aug
        self._cj = None
        self._rrc = None
        if isinstance(self.aug, dict) and self.aug.get("color_jitter", False):
            p = self.aug.get("color_jitter_params", {})
            # 必要なら RandomApply で確率化してもOK（メモリには無関係）
            self._cj = transforms.ColorJitter(
                brightness=float(p.get("brightness", 0.2)),
                contrast=float(p.get("contrast", 0.2)),
                saturation=float(p.get("saturation", 0.2)),
                hue=float(p.get("hue", 0.02)),
            )
        # random_resized_crop を YAML で有効化している場合に対応
        if isinstance(self.aug, dict):
            rrc = self.aug.get("random_resized_crop", {})
            if rrc and rrc.get("enabled", False):
                self._rrc = {
                    "scale": tuple(rrc.get("scale", [0.8, 1.0])),
                    "ratio": tuple(rrc.get("ratio", [0.9, 1.1]))
                }
        self._rng = rng

    def __len__(self) -> int:
        return len(self.items)

    @staticmethod
    def _snap_to_multiple(x: int, m: int, mode: str = "floor") -> int:
        if m <= 1: return x
        if mode == "ceil":
            return int(((x + m - 1) // m) * m)
        return int((x // m) * m)  # floor

    def _resize_keep_long_side(self, pil: Image.Image, long_side: int) -> Image.Image:
        """アスペクト保持で long_side に合わせた後、patch(=14) の倍数にスナップ（floor）。"""

        w, h = pil.size
        if max(w, h) == long_side:
            # そのままでも、14の倍数にスナップ（必要なら微小リサイズ）
            nw, nh = w, h
        else:
            if w >= h:
                nw = long_side
                nh = int(round(h * long_side / max(w, 1)))
            else:
                nh = long_side
                nw = int(round(w * long_side / max(h, 1)))
        # ここで14の倍数に（小さく切り下げ）。最低でも14を確保。
        nw = max(self.patch, self._snap_to_multiple(nw, self.patch, mode="floor"))
        nh = max(self.patch, self._snap_to_multiple(nh, self.patch, mode="floor"))
        if (nw, nh) == (w, h):
            return pil  # 変更なし
        # 互換用に冒頭で定義済みの _BICUBIC を使う（Pillow新旧どちらでもOK）
        return pil.resize((nw, nh), _BICUBIC)

    @staticmethod
    def _pad_to_square(pil, target, fill=None):
        w, h = pil.size
        if w == target and h == target:
            return pil
        # fill=None のときだけ「画像用の CLIP 平均」で埋める
        if fill is None:
            if pil.mode == "RGB":
                fill = tuple(int(round(m*255)) for m in _CLIP_MEAN)  # 画像: CLIP 平均
            else:
                fill = 0                                           # マスク(L等): 0 を既定
        new_img = Image.new(pil.mode, (target, target), color=fill)
        # 中央寄せで貼り付け（右下パディングの偏りを避ける）
        left = (target - w) // 2
        top  = (target - h) // 2
        new_img.paste(pil, (left, top))
        return new_img

    def _choose_text(self, it: Dict[str, Any]) -> str:
        if self.use_answers == "expand":
            return str(it.get(self.text_key, ""))
        if self.use_answers == "random":
            answers = it.get(self.answer_key) or it.get(self.answer_key + "s")
            if isinstance(answers, list) and len(answers) > 0:
                return str(self._rng.choice(answers))
            return str(it.get(self.text_key, ""))
        if self.use_answers == "first":
            answers = it.get(self.answer_key) or it.get(self.answer_key + "s")
            if isinstance(answers, list) and len(answers) > 0:
                return str(answers[0])
            return str(it.get(self.text_key, ""))
        return str(it.get(self.text_key, ""))

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        it = self.items[idx]
        img_path = it["image_path"]

        # 画像読込（PIL）。元サイズを保持（③の要点）
        img = Image.open(img_path).convert("RGB")
        W0, H0 = img.size

        # モデル入力サイズに合わせたリサイズ（長辺＝long_side, アスペクト保持）
        img_resized = self._resize_keep_long_side(img, self.long_side)
        W2, H2 = img_resized.size

        # 監督マスクは必ず「オリジナル座標系(W0,H0) or (width,height)」で生成してから最終にリサイズ
        w_meta, h_meta = int(it.get("width", 0)), int(it.get("height", 0))
        base_w = w_meta if w_meta > 0 else W0
        base_h = h_meta if h_meta > 0 else H0

        mask_pil: Optional[Image.Image] = None
        if self.supervision in ("mask", "auto"):
            seg = it.get("segmentation")
            if seg:
                # 1) RLE 優先（COCO準拠）
                mask_pil = _rle_to_mask(seg)
                if mask_pil is None:
                    # 2) ポリゴンも試す
                    polys = _seg_to_polys(seg)
                    if polys:
                        mask_pil = _poly_mask(base_w, base_h, polys)

        if mask_pil is None:
            box = it.get("bbox")
            if box:
                mask_pil = _bbox_mask(base_w, base_h, box)

        if mask_pil is None:
            mask_pil = Image.new("L", (base_w, base_h), 0)

        # 画像の最終サイズ(W2,H2)に揃える（ここで初めてリサイズ）
        # NOTE: マスクは最近傍で座標忠実に（冒頭定義の _NEAREST を使用）
        mask_resized = mask_pil.resize((W2, H2), _NEAREST)

        # --- Aug: flip（画像とマスクを同期）※ train split のみ ---
        if self._is_train_split and isinstance(self.aug, dict) and self.aug.get("flip", False):
            if self._rng.random() < 0.5:
                img_resized = img_resized.transpose(Image.FLIP_LEFT_RIGHT)
                mask_resized = mask_resized.transpose(Image.FLIP_LEFT_RIGHT)

        # --- Aug: random_resized_crop（画像とマスクを同期）※ train split のみ ---
        if self._is_train_split and self._rrc is not None:
            scale = self._rrc["scale"]; ratio = self._rrc["ratio"]
            # サンプルウィンドウを手動で切り出して一致させる
            w, h = img_resized.size
            for _ in range(10):
                area = w * h
                target_area = self._rng.uniform(*scale) * area
                log_ratio = (np.log(ratio[0]), np.log(ratio[1]))
                aspect = np.exp(self._rng.uniform(*log_ratio))
                nw = int(round((target_area * aspect) ** 0.5))
                nh = int(round((target_area / aspect) ** 0.5))
                if nw <= w and nh <= h and nw > 0 and nh > 0:
                    x1 = self._rng.randint(0, w - nw)
                    y1 = self._rng.randint(0, h - nh)
                    box = (x1, y1, x1 + nw, y1 + nh)
                    # crop → そのまま利用（resizeは不要：crop後のサイズは (nw, nh) ）
                    # crop後のサイズが (nw, nh) になるので追加resizeは不要
                    img_resized = img_resized.crop(box)
                    mask_resized = mask_resized.crop(box)
                    break
            # 失敗時はスキップ（そのまま）

        # --- Aug: color jitter（画像のみ）※ train split のみ ---
        if self._is_train_split and self._cj is not None:
            img_resized = self._cj(img_resized)

        # === 正方にパディング（アスペクト保持, 歪みなし）===
        target = self._snap_to_multiple(self.long_side, self.patch, mode="ceil")
        img_sq  = self._pad_to_square(img_resized,  target, fill=None)  # 画像は CLIP 平均でパディング
        mask_sq = self._pad_to_square(mask_resized, target, fill=0)     # マスクは 0（背景）
        # --- 追加: valid マスク（元画像=1, パディング=0） ---
        # img_resized の実サイズ領域を 1、その外側（パディング分）を 0 とする
        valid_core = Image.new("L", img_resized.size, color=255)       # (W2,H2) で 1
        valid_sq_pil = self._pad_to_square(valid_core, target, fill=0) # 正方パディングで 0 埋め
        valid_np = (np.asarray(valid_sq_pil, dtype=np.float32) / 255.0)
        valid_t = torch.from_numpy(valid_np)[None, ...]  # (1, H, W) 0/1

        # Tensor 化（ここで初めて配列化）
        img_t = self.tf_img(img_sq)
        m_np = (np.asarray(mask_sq, dtype=np.float32) / 255.0)
        m_t = torch.from_numpy(m_np)[None, ...]  # (1,H,W)

        txt = self._choose_text(it)

        return {
            "image": img_t,          # (3, target, target) normalized
            "soft_mask": m_t,        # (1, target, target) in [0,1]
            "valid": valid_t,
            "text": txt,
            "path": img_path,
        }
