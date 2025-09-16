# models/qformer_refcoco.py
import math
from typing import Dict, Any, Tuple, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# ========== Vision: OpenCLIP ==========
import open_clip

# ========== Q-Former (BERT with cross-attn) ==========
from transformers import BertConfig, BertModel

# ========== Vicuna (LLaMA系) ==========
from transformers import AutoTokenizer, AutoModelForCausalLM


# ---------------------------
# Vision encoder (frozen)
# ---------------------------
class OpenCLIPViTL14Frozen(nn.Module):
    """
    OpenCLIP ViT-L/14 をトークン列として取り出すラッパ
    出力:
      tokens: (B, L, 768)   ※ [CLS] を除いた patch token 列（768 に射影済み）
      feat_hw: (Ht, Wt)     ※ Ht*Wt = L
    """
    def __init__(self, model_name: str = "ViT-L-14", pretrained: str = "openai"):
        super().__init__()
        model, _, _ = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
        self.model = model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

        # ViT-L/14: patch size = 14
        self.patch_size = 14

        # 入力埋め込み次元の推定（OpenCLIPでは class_embedding の次元が最も安全）
        vis = self.model.visual
        if hasattr(vis, "class_embedding"):
            in_dim = int(vis.class_embedding.shape[-1])
        elif hasattr(vis, "width") and isinstance(vis.width, (int, float)):
            in_dim = int(vis.width)
        elif hasattr(vis, "embed_dim") and isinstance(vis.embed_dim, (int, float)):
            in_dim = int(vis.embed_dim)
        else:
            # 最後の保険（ViT-L/14 は通常 1024）
            in_dim = 1024

        self.in_dim = in_dim        # 例: 1024（ViT-L/14）
        self.out_dim = in_dim       # 互換のため残す（未使用）

        # Q-Former 側の想定 768 に合わせる（すでに 768 なら恒等）
        if self.in_dim == 768:
            self.to_qformer = nn.Identity()
        else:
            self.to_qformer = nn.Linear(self.in_dim, 768, bias=False)
        for p in self.to_qformer.parameters():
            p.requires_grad = False

        print(f"[info] OpenCLIP visual dim inferred: {self.in_dim} -> proj_to 768 ({self.to_qformer.__class__.__name__})")

    # ---- 微解凍（任意） ----
    def unfreeze_to_qformer(self):
        for p in self.to_qformer.parameters():
            p.requires_grad = True
        print("[info] to_qformer unfrozen (low-LR fine-tuning recommended)")

    # ---- 位置埋め込みの補助 ----
    @staticmethod
    def _ensure_3d_pos_embed(pe: torch.Tensor) -> torch.Tensor:
        """positional_embedding を [1, L+1, C] に正規化"""
        if pe.dim() == 2:  # [L+1, C]
            pe = pe.unsqueeze(0)  # -> [1, L+1, C]
        return pe  # [1, L+1, C] を想定

    @staticmethod
    def _resize_pos_embed(pe: torch.Tensor, Ht: int, Wt: int) -> torch.Tensor:
        """
        pe: [1, 1+Lorig, C]
        先頭1トークンをCLSとして保持し、グリッド部を (Ht,Wt) に補間して返す: [1, 1+Ht*Wt, C]
        """
        pe = OpenCLIPViTL14Frozen._ensure_3d_pos_embed(pe)  # [1, 1+L, C]
        cls = pe[:, :1, :]            # [1,1,C]
        grid = pe[:, 1:, :]           # [1,L,C]
        L = grid.shape[1]
        Gh = int(round(math.sqrt(L)))
        Gw = Gh if Gh > 0 else 1
        if Gh * Gw != L and Gh > 0:
            Gw = max(1, L // Gh)

        # [1,L,C] -> [1,C,Gh,Gw]
        grid = grid.reshape(1, Gh, Gw, grid.shape[-1]).permute(0, 3, 1, 2).contiguous()
        # 補間
        grid = F.interpolate(grid, size=(Ht, Wt), mode="bicubic", align_corners=False)
        # [1,C,Ht,Wt] -> [1,Ht*Wt,C]
        grid = grid.permute(0, 2, 3, 1).reshape(1, Ht * Wt, -1).contiguous()
        return torch.cat([cls, grid], dim=1)  # [1, 1+Ht*Wt, C]

    @torch.no_grad()
    def forward(self, images: torch.Tensor) -> Dict[str, Any]:
        """
        images: (B,3,H,W), CLIP正規化済み
        返り: {"tokens": (B,L,768), "feat_hw": (Ht,Wt)}
        """
        vis = self.model.visual  # OpenCLIP VisionTransformer

        # --- stem ---
        x = vis.conv1(images)            # (B, C, H/patch, W/patch)
        B, C, Ht, Wt = x.shape
        L = Ht * Wt

        # (B, C, Ht, Wt) -> (B, L, C)
        x = x.reshape(B, C, L).permute(0, 2, 1)  # (B, L, C)

        # --- class token 付与 ---
        cls_tok = vis.class_embedding.to(x.dtype)
        cls_tok = cls_tok + torch.zeros(B, 1, C, device=x.device, dtype=x.dtype)
        x = torch.cat([cls_tok, x], dim=1)  # (B, 1+L, C)

        # --- 位置埋め込み ---
        pe = getattr(vis, "positional_embedding", None)
        if pe is None:
            raise AttributeError("visual.positional_embedding is missing in OpenCLIP vision backbone.")
        pe = pe.to(x.dtype).to(x.device)
        pe = self._ensure_3d_pos_embed(pe)  # [1, 1+Lorig, C]
        if pe.shape[1] != x.shape[1]:
            # 入力解像度と異なる場合は補間
            pe = self._resize_pos_embed(pe, Ht, Wt)  # [1, 1+L, C]
        x = x + pe

        # --- 前処理 LN + Transformer ---
        x = vis.ln_pre(x)
        x = x.permute(1, 0, 2)           # NLD -> LND
        x = vis.transformer(x)
        x = x.permute(1, 0, 2)           # LND -> NLD

        # --- CLS 除去 ---
        x = x[:, 1:, :]                  # (B, L, C)

        # --- 768 へ射影（Q-Former 入力）---
        x = self.to_qformer(x)           # (B, L, 768)

        return {"tokens": x, "feat_hw": (Ht, Wt)}


# ---------------------------
# Q-Former (BERT+cross-attn)
# ---------------------------
class BertQFormer(nn.Module):
    """
    BERT Encoder を cross-attention 有効 (add_cross_attention=True) で Q-Former 的に使用。
    - 入力: 学習可能 query embedding を inputs_embeds として与える
    - encoder_hidden_states に vision tokens を与える
    返り:
      out_q: (B, Nq, Hq)
      attn_logits: (B, 1, Ht, Wt)  ※クエリ平均の指示マップ（学習用ロジット）
    """
    def __init__(self,
                 hidden_size: int = 768,
                 num_hidden_layers: int = 6,
                 num_attention_heads: int = 12,
                 use_xattn: bool = True,
                 xattn_last_k: int = 2,
                 mix_alpha: float = 0.5):
        super().__init__()
        cfg = BertConfig(
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            intermediate_size=hidden_size * 4,
            add_cross_attention=True,
            is_decoder=True,
        )
        self.bert = BertModel(cfg)
        # 学習可能クエリ
        self.num_queries = 16
        self.query_embed = nn.Parameter(torch.randn(self.num_queries, hidden_size) * 0.02)

        # ロジット温度・オフセット・クエリ重み
        self.logit_tau = nn.Parameter(torch.tensor(1.0))   # 実際は exp(tau) を温度に使用
        self.logit_bias = nn.Parameter(torch.zeros(1))
        self.query_gate = nn.Parameter(torch.ones(self.num_queries))  # 各クエリの重み

        # x-attn 利用設定
        self.use_xattn = bool(use_xattn)
        self.xattn_last_k = int(xattn_last_k)
        self.mix_alpha = float(mix_alpha)  # 1.0=類似のみ, 0.0=x-attnのみ

    def forward(self, vision_tokens: torch.Tensor, feat_hw: Tuple[int, int]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        vision_tokens: (B, L, C=hidden_size)
        feat_hw: (Ht, Wt)
        """
        B, L, C = vision_tokens.shape
        queries = self.query_embed.unsqueeze(0).expand(B, -1, -1)  # (B, Nq, C)
        q_mask  = torch.ones(B, self.num_queries, device=queries.device, dtype=torch.long)
        v_mask  = torch.ones(B, L, device=queries.device, dtype=torch.long)

        out = self.bert(
            inputs_embeds=queries,
            attention_mask=q_mask,
            encoder_hidden_states=vision_tokens,
            encoder_attention_mask=v_mask,
            return_dict=True,
            output_attentions=self.use_xattn,   # ★ cross_attentions を得る
        )
        out_q = out.last_hidden_state  # (B, Nq, C)

        # --- ① 類似ベース（cosine）---
        qw = torch.softmax(self.query_gate, dim=0)            # (Nq,)
        sim = torch.einsum(
            "bnc,blc->bnl",
            F.normalize(out_q, dim=-1),
            F.normalize(vision_tokens, dim=-1),
        )  # (B,Nq,L)
        sim = (sim * qw.view(1, -1, 1)).sum(dim=1) / (qw.sum() + 1e-6)  # (B,L)
        sim = (sim - sim.mean(dim=1, keepdim=True)) / (sim.std(dim=1, keepdim=True) + 1e-6)  # z-score

        # --- ② x-attnベース（最後k層の平均, ヘッド&クエリ重み平均）---
        if self.use_xattn and hasattr(out, "cross_attentions") and (out.cross_attentions is not None):
            cws = out.cross_attentions  # tuple(len=num_layers) of (B, heads, Nq, L)
            kk = min(self.xattn_last_k, len(cws))
            xs: List[torch.Tensor] = []
            for t in cws[-kk:]:
                # ヘッド平均, クエリ重み平均
                t = t.mean(dim=1)  # (B,Nq,L)
                t = (t * qw.view(1, -1, 1)).sum(dim=1) / (qw.sum() + 1e-6)  # (B,L)
                xs.append(t)
            xattn = torch.stack(xs, dim=0).mean(dim=0)  # (B,L)
            xattn = (xattn - xattn.mean(dim=1, keepdim=True)) / (xattn.std(dim=1, keepdim=True) + 1e-6)
        else:
            xattn = sim  # フォールバック

        # --- ③ 混合 & 温度/バイアス ---
        z = self.mix_alpha * sim + (1.0 - self.mix_alpha) * xattn
        tau = torch.exp(self.logit_tau).clamp(1e-3, 100.0)
        logits_flat = tau * z + self.logit_bias  # (B,L)

        Ht, Wt = feat_hw
        logits_flat = logits_flat[..., :Ht * Wt]
        attn_logits = logits_flat.view(B, 1, Ht, Wt)
        return out_q, attn_logits


# ---------------------------
# Projector (to Vicuna dim)
# ---------------------------
class ProjectorMLP(nn.Module):
    def __init__(self, dim_in: int, dim_out: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim_in, dim_out),
            nn.GELU(),
            nn.Linear(dim_out, dim_out),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------
# Main Model
# ---------------------------
class QFormerRefCOCO(nn.Module):
    """
    returns:
      {
        "attn_logits": (B,1,Ht,Wt),          # 学習はこれを使用
        "attn_maps":   (B,1,Ht,Wt),          # = sigmoid(attn_logits)（可視化・評価用）
        "feat_hw":     torch.LongTensor([Ht,Wt]),
        "vision_emb":  (B,Nq,768),
        "vis_prefix":  (B,Nq,proj_dim_out),
        "text_emb":    (B, 768) or None,     # ★ 射影済み（InfoNCE 用）
        "text_emb_raw":(B, hidden) or None,  #   元のLM隠れ次元（ログ等）
        "lm_loss_raw": Tensor() or 0.,
        "vision_tokens": (B,L,768),          # ★ 監視・デバッグ用
      }
    """
    def __init__(self,
                 vision_encoder: Optional[nn.Module] = None,
                 qformer: Optional[nn.Module] = None,
                 llm_name: str = "lmsys/vicuna-7b-v1.5",
                 num_queries: int = 16,
                 proj_dim_in: int = 768,
                 proj_dim_out: int = 4096,
                 max_txt_len: int = 64,
                 load_vicuna: bool = False):
        super().__init__()
        # Vision
        self.vision = vision_encoder if vision_encoder is not None else OpenCLIPViTL14Frozen()
        for p in self.vision.parameters():
            p.requires_grad = False
        self.vision.eval()

        # Q-Former
        self.qformer = qformer if qformer is not None else BertQFormer(hidden_size=proj_dim_in)
        if hasattr(self.qformer, "num_queries") and self.qformer.num_queries != num_queries:
            C = self.qformer.query_embed.shape[-1]
            self.qformer.query_embed = nn.Parameter(torch.randn(num_queries, C) * 0.02)
            self.qformer.num_queries = num_queries

        # Projector（vision -> LLM hidden）
        self.projector = ProjectorMLP(proj_dim_in, proj_dim_out)

        # 入力トークンを安定化（簡易 LN）
        self.pre_q_ln = nn.LayerNorm(proj_dim_in)

        self.projector_text = nn.Linear(
            getattr(self, "llm_hidden", 4096) if hasattr(self, "llm_hidden") else proj_dim_out,  # 後で上書き
            proj_dim_in,
            bias=False,
        )

        # Vicuna (optional)
        self.llm_name = llm_name
        self.max_txt_len = int(max_txt_len)
        self.load_vicuna = bool(load_vicuna)
        if self.load_vicuna:
            self.tokenizer = AutoTokenizer.from_pretrained(self.llm_name, use_fast=False)
            self.lm = AutoModelForCausalLM.from_pretrained(
                self.llm_name,
                torch_dtype=torch.float16,
                device_map="auto"
            )
            self.lm.eval()
            for p in self.lm.parameters():
                p.requires_grad = False

            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            if getattr(self.lm.config, "pad_token_id", None) is None:
                self.lm.config.pad_token_id = self.tokenizer.eos_token_id

            # ★ テキストを 4096 -> 768 に射影（InfoNCE 用）
            self.llm_hidden = int(self.lm.config.hidden_size)  # 例: 4096（7B）
            self.projector_text = nn.Linear(self.llm_hidden, proj_dim_in, bias=False)
            nn.init.xavier_uniform_(self.projector_text.weight)
            print(f"[info] text projector: {self.llm_hidden} -> 768 (Linear)")
        else:
            self.tokenizer = None
            self.lm = None
            self.projector_text = None  # type: ignore

    @torch.no_grad()
    def _encode_vision(self, images: torch.Tensor) -> Dict[str, Any]:
        return self.vision(images)

    def _encode_text(self, texts: List[str]) -> Optional[torch.Tensor]:
        """
        戻り値: (B, hidden)  … LM の最後層隠れ状態（proj前）
        """
        if not self.load_vicuna:
            return None
        tok = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_txt_len,
            return_tensors="pt"
        )
        # モデルが device_map="auto" のため、入力は最初のモジュール側へ移される。
        dev = next(self.parameters()).device
        tok = {k: v.to(dev) for k, v in tok.items()}

        with torch.no_grad():
            # LLaMA系は .model を掘る場合がある（hidden_states 用）
            out = self.lm.model(**tok, output_hidden_states=True, return_dict=True)
            last = out.hidden_states[-1]  # (B,T,hidden)

            # 簡易 pooling: 最後トークン
            lengths = tok["attention_mask"].sum(dim=1) - 1
            pooled = last[torch.arange(last.size(0), device=last.device), lengths]  # (B, hidden)
        return pooled.to(dtype=self.projector_text.weight.dtype)

    def _compute_lm_loss(self, texts: List[str]) -> torch.Tensor:
        if not self.load_vicuna:
            return torch.tensor(0.0, device=next(self.parameters()).device)
        tok = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_txt_len,
            return_tensors="pt"
        )
        dev = next(self.parameters()).device
        tok = {k: v.to(dev) for k, v in tok.items()}
        labels = tok["input_ids"].clone()
        labels[labels == self.tokenizer.pad_token_id] = -100
        with torch.no_grad():
            out = self.lm(**tok, labels=labels, return_dict=True)
            loss = out.loss  # scalar
        return loss

    def forward(self, inputs: Dict[str, Any], compute_lm: bool = False) -> Dict[str, Any]:
        """
        inputs: {"image": Tensor(B,3,H,W), "text": List[str]}
        """
        images: torch.Tensor = inputs["image"]
        texts: List[str] = inputs.get("text", [""] * images.size(0))

        # 1) Vision → tokens
        v = self._encode_vision(images)  # {"tokens": (B,L,768), "feat_hw": (Ht,Wt)}
        tokens: torch.Tensor = v["tokens"]
        feat_hw: Tuple[int, int] = v["feat_hw"]

        # 1.5) 事前正規化
        tokens = self.pre_q_ln(tokens)

        # 2) Q-Former（BERT cross-attn）: ★AMPの外でfp32強制（NaN対策）
        from torch import amp
        with amp.autocast(device_type="cuda", enabled=False):
            out_q, attn_logits = self.qformer(tokens.float(), feat_hw)  # (B,Nq,768), (B,1,Ht,Wt)

        # 3) Project to Vicuna hidden (vision prefix)
        vis_prefix = self.projector(out_q)  # (B,Nq,proj_dim_out)

        # 4) Optional: text embedding / LM loss
        text_emb_raw = self._encode_text(texts)  # (B, hidden) or None
        if text_emb_raw is not None:
            text_emb = self.projector_text(
                text_emb_raw.to(dtype=self.projector_text.weight.dtype)
            ) 
        else:
            text_emb = None

        lm_loss = self._compute_lm_loss(texts) if compute_lm else torch.tensor(0.0, device=images.device)

        out: Dict[str, Any] = {
            "attn_logits": attn_logits,                                            # 学習はこれ
            "attn_maps": torch.sigmoid(attn_logits),                               # 可視化・評価用
            "feat_hw": torch.tensor([feat_hw[0], feat_hw[1]], device=images.device, dtype=torch.long),
            "vision_emb": out_q,        # (B,Nq,768)
            "vis_prefix": vis_prefix,   # (B,Nq,proj_dim_out)
            "text_emb": text_emb,       # (B,768) or None  ← InfoNCE 用
            "lm_loss_raw": lm_loss,     # scalar tensor (0 もあり)
            "vision_tokens": tokens,    # 監視・デバッグ用
        }
        return out
