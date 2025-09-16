# models/qformer_refcoco_inst.py
import os
import re
import math
from typing import Dict, Any, Tuple, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# ========== Vision: OpenCLIP ==========
import open_clip

# ========== Q-Former (BERT with cross-attn) ==========
from transformers import BertConfig, BertModel

# ---- Optional loaders (prefer InstructBLIP, then BLIP-2) ----
try:
    from transformers import InstructBlipQFormerModel
    _HAS_HF_INSTRUCT_QFORMER = True
except Exception:
    _HAS_HF_INSTRUCT_QFORMER = False

try:
    from transformers import Blip2QFormerModel
    _HAS_HF_BLIP2_QFORMER = True
except Exception:
    _HAS_HF_BLIP2_QFORMER = False

# ========== Vicuna (LLaMA系) ==========
from transformers import AutoTokenizer, AutoModelForCausalLM


# ---------------------------
# Vision encoder (frozen)
# ---------------------------
class OpenCLIPVisionFrozen(nn.Module):
    """
    OpenCLIP ViT をトークン列として取り出すラッパ
    出力:
      tokens: (B, L, C_vis)   ※ [CLS] を除いた patch token 列（射影なし）
      feat_hw: (Ht, Wt)       ※ Ht*Wt = L
      vis_dim: int            ※ C_vis（例: ViT-L/14 -> 1024, ViT-g/14 -> 1408）
    """
    def __init__(self, model_name: str = "ViT-g-14", pretrained: str = "laion2b_s34b_b88k"):
        super().__init__()
        model, _, _ = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
        self.model = model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

        vis = self.model.visual
        in_dim = int(getattr(vis, "class_embedding", torch.empty(1,1408)).shape[-1]) if hasattr(vis,"class_embedding") else 1408
        self.vis_dim = in_dim            # ← ここが 1408 に
        self.patch_size = 14             # ViT-*/14 はパッチ14固定
        print(f"[info] Using OpenCLIP {model_name} ({pretrained}).")
        print(f"[info] OpenCLIP visual dim inferred: {self.vis_dim} (no projection)")

    @staticmethod
    def _ensure_3d_pos_embed(pe: torch.Tensor) -> torch.Tensor:
        if pe.dim() == 2:
            pe = pe.unsqueeze(0)
        return pe

    @staticmethod
    def _resize_pos_embed(pe: torch.Tensor, Ht: int, Wt: int) -> torch.Tensor:
        pe = OpenCLIPVisionFrozen._ensure_3d_pos_embed(pe)
        cls = pe[:, :1, :]
        grid = pe[:, 1:, :]
        L = grid.shape[1]
        Gh = int(round(math.sqrt(L)))
        Gw = Gh if Gh > 0 else 1
        if Gh * Gw != L and Gh > 0:
            Gw = max(1, L // Gh)

        grid = grid.reshape(1, Gh, Gw, grid.shape[-1]).permute(0, 3, 1, 2).contiguous()
        grid = F.interpolate(grid, size=(Ht, Wt), mode="bicubic", align_corners=False)
        grid = grid.permute(0, 2, 3, 1).reshape(1, Ht * Wt, -1).contiguous()
        return torch.cat([cls, grid], dim=1)

    @torch.no_grad()
    def forward(self, images: torch.Tensor) -> Dict[str, Any]:
        vis = self.model.visual

        x = vis.conv1(images)            # (B, C, H/patch, W/patch)
        B, C, Ht, Wt = x.shape
        L = Ht * Wt

        x = x.reshape(B, C, L).permute(0, 2, 1)  # (B, L, C)

        cls_tok = vis.class_embedding.to(x.dtype)
        cls_tok = cls_tok + torch.zeros(B, 1, C, device=x.device, dtype=x.dtype)
        x = torch.cat([cls_tok, x], dim=1)  # (B, 1+L, C)

        pe = getattr(vis, "positional_embedding", None)
        if pe is None:
            raise AttributeError("visual.positional_embedding is missing in OpenCLIP vision backbone.")
        pe = pe.to(x.dtype).to(x.device)
        pe = self._ensure_3d_pos_embed(pe)
        if pe.shape[1] != x.shape[1]:
            pe = self._resize_pos_embed(pe, Ht, Wt)
        x = x + pe

        x = vis.ln_pre(x)
        x = x.permute(1, 0, 2)           # NLD -> LND
        x = vis.transformer(x)
        x = x.permute(1, 0, 2)           # LND -> NLD

        x = x[:, 1:, :]                  # (B, L, C)

        return {"tokens": x, "feat_hw": (Ht, Wt), "vis_dim": self.vis_dim}

OpenCLIPViTL14Frozen = OpenCLIPVisionFrozen


# ---------------------------
# Q-Former (BERT+cross-attn)
# ---------------------------
class BertQFormer(nn.Module):
    def __init__(
        self,
        hidden_size: int = 768,
        num_hidden_layers: int = 6,
        num_attention_heads: int = 12,
        encoder_hidden_size: Optional[int] = None,   # ← cross-attn の K/V 入力次元（例: 1408）
    ):
        super().__init__()
        cfg = BertConfig(
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            intermediate_size=hidden_size * 4,
            add_cross_attention=True,
            is_decoder=True,
        )
        cfg.attn_implementation = "eager"

        if encoder_hidden_size is None:
            encoder_hidden_size = hidden_size
        cfg.encoder_hidden_size = int(encoder_hidden_size)
        setattr(cfg, "cross_attention_hidden_size", int(encoder_hidden_size))

        self.bert = BertModel(cfg)

        # ★ transformers の一部バージョンで encoder_hidden_size を無視する対策：
        #    cross-attn の key/value Linear を in_features=encoder_hidden_size で作り直す
        self._maybe_rebuild_cross_kv(int(encoder_hidden_size))

        self.num_queries = 16
        self.query_embed = nn.Parameter(torch.randn(self.num_queries, hidden_size) * 0.02)
        self.sim_proj: Optional[nn.Linear] = None

    def _maybe_rebuild_cross_kv(self, kv_in_features: int):
        try:
            layers = getattr(self.bert.encoder, "layer", None)
            if layers is None:
                return
            rebuilt = 0
            for li, layer in enumerate(layers):
                attn = getattr(layer, "crossattention", None)
                if attn is None:
                    continue
                attn_self = getattr(attn, "self", None)
                if attn_self is None:
                    continue

                for name in ("key", "value"):
                    lin = getattr(attn_self, name, None)
                    if not isinstance(lin, nn.Linear):
                        continue
                    want_in = kv_in_features
                    cur_in = lin.in_features
                    if cur_in == want_in:
                        continue
                    # 再構築
                    with torch.no_grad():
                        new_lin = nn.Linear(want_in, lin.out_features, bias=(lin.bias is not None))
                        nn.init.xavier_uniform_(new_lin.weight)
                        if new_lin.bias is not None:
                            nn.init.zeros_(new_lin.bias)
                        # できるだけコピー（共通次元のみ）
                        common = min(cur_in, want_in)
                        new_lin.weight[:, :common].copy_(lin.weight[:, :common])
                        if new_lin.bias is not None and lin.bias is not None:
                            new_lin.bias.copy_(lin.bias)
                    # 置換（デバイス/dtype合わせ）
                    new_lin.to(lin.weight.device, dtype=lin.weight.dtype)
                    setattr(attn_self, name, new_lin)
                    rebuilt += 1
            if rebuilt > 0:
                print(f"[qformer] rebuilt cross-attn K/V linears with in_features={kv_in_features} (layers touched={rebuilt})")
        except Exception as e:
            print(f"[warn] cross-attn K/V rebuild skipped due to: {repr(e)}")

    # ---- pretrained loader (InstructBLIP優先) ----
    def load_pretrained_from_blip2(
        self,
        source: str,
        num_queries_target: Optional[int] = None,
        map_location: str = "cpu",
        prefer_ckpt_kv_dim: bool = True,
    ) -> Dict[str, Any]:
        print(f"[qformer] loading pretrained from: {source}")
        print(f"[qformer] prefer_ckpt_kv_dim={prefer_ckpt_kv_dim}")

        sd = None
        used_hf_loader = False

        def _load_torch_file(fp: str):
            if fp.endswith(".safetensors"):
                try:
                    from safetensors.torch import load_file as _safe_load
                    return _safe_load(fp, device=map_location)
                except Exception as e:
                    raise RuntimeError(f"Failed to load safetensors: {e}")
            else:
                return torch.load(fp, map_location=map_location)

        # 0) local single file
        if sd is None and os.path.isfile(source):
            try:
                sd = _load_torch_file(source)
                print(f"[qformer] loaded local file: {source}")
            except Exception as e:
                print(f"[warn] local file load failed: {e}")

        # 1) local dir
        if sd is None and os.path.isdir(source):
            cands = [
                os.path.join(source, "pytorch_model.bin"),
                os.path.join(source, "model.safetensors"),
                os.path.join(source, "blip2_pretrain.pth"),
                os.path.join(source, "checkpoint.pth"),
            ]
            for fp in cands:
                if os.path.exists(fp):
                    print(f"[qformer] found checkpoint file: {fp}")
                    sd = _load_torch_file(fp)
                    break
            if sd is None:
                for fn in sorted(os.listdir(source)):
                    if fn.endswith((".bin", ".safetensors", ".pth")):
                        sd = _load_torch_file(os.path.join(source, fn))
                        print(f"[qformer] found checkpoint file: {fn}")
                        break

        # 2) HF direct Q-Former
        if sd is None and _HAS_HF_INSTRUCT_QFORMER:
            for sub in ("qformer", None):
                try:
                    hf_q = InstructBlipQFormerModel.from_pretrained(
                        source, subfolder=sub, torch_dtype="auto", trust_remote_code=True
                    )
                    sd = hf_q.state_dict()
                    used_hf_loader = True
                    print("[qformer] loaded via transformers.InstructBlipQFormerModel", "subfolder=", sub)
                    break
                except Exception as e:
                    print(f"[warn] InstructBlipQFormerModel load failed (subfolder={sub}): {e}")

        if sd is None and _HAS_HF_BLIP2_QFORMER:
            for sub in ("qformer", None):
                try:
                    hf_q = Blip2QFormerModel.from_pretrained(
                        source, subfolder=sub, torch_dtype="auto", trust_remote_code=True
                    )
                    sd = hf_q.state_dict()
                    used_hf_loader = True
                    print("[qformer] loaded via transformers.Blip2QFormerModel", "subfolder=", sub)
                    break
                except Exception as e:
                    print(f"[warn] Blip2QFormerModel load failed (subfolder={sub}): {e}")

        # 3) HF full model extract
        if sd is None:
            try:
                from transformers import InstructBlipForConditionalGeneration
                full = InstructBlipForConditionalGeneration.from_pretrained(
                    source, torch_dtype="auto", trust_remote_code=True
                )
                raw = full.state_dict()
                sub_sd = {k: v for k, v in raw.items() if k.startswith("qformer.")}
                if sub_sd:
                    sd = sub_sd
                    used_hf_loader = True
                    print("[qformer] extracted qformer/* from full InstructBLIP")
            except Exception as e:
                print(f"[warn] InstructBLIP full-model extract failed: {e}")

        if sd is None:
            raise FileNotFoundError(
                f"Could not load state_dict from '{source}'. "
                f"Pass a HF repo id or a local file/dir."
            )

        if isinstance(sd, dict) and "model" in sd and isinstance(sd["model"], dict):
            sd = sd["model"]

        def _clean_prefix(k: str) -> str:
            for p in ("model.", "module.", "state_dict."):
                if k.startswith(p):
                    k = k[len(p):]
            return k

        new_sd: Dict[str, torch.Tensor] = {}
        q_src: Optional[torch.Tensor] = None

        for k, v in sd.items():
            k = _clean_prefix(k)
            if k.startswith("qformer."):
                k = k[len("qformer."):]
            if k.startswith("bert."):
                kk = k[len("bert."):]
            elif k.startswith(("embeddings.", "encoder.", "pooler.")):
                kk = k
            else:
                kk = None

            if kk is not None:
                kk = kk.replace("embeddings.layernorm.", "embeddings.LayerNorm.")
                kk = kk.replace(".attention.attention.", ".attention.self.")
                kk = kk.replace(".crossattention.attention.", ".crossattention.self.")
                kk = kk.replace(".LayerNorm.gamma", ".LayerNorm.weight")
                kk = kk.replace(".LayerNorm.beta", ".LayerNorm.bias")
                new_sd[kk] = v
                continue

            base = k.split(".")[0]
            if base in {"query_tokens", "query_token", "query_embed", "query"} and isinstance(v, torch.Tensor):
                if v.dim() == 3:
                    v = v.squeeze(0)
                if v.dim() == 2:
                    q_src = v

        # vocab resize (先に)
        emb_key = "embeddings.word_embeddings.weight"
        if emb_key in new_sd:
            new_vocab = new_sd[emb_key].size(0)
            cur_vocab = self.bert.embeddings.word_embeddings.weight.size(0)
            if new_vocab != cur_vocab:
                self.bert.resize_token_embeddings(new_vocab)
                self.bert.config.vocab_size = new_vocab
                print(f"[qformer] resized token embeddings: {cur_vocab} -> {new_vocab}")

        # kv dims
        kv_dim_ckpt = None
        # まず ".self.key" を優先
        self_keys = [
            "encoder.layer.0.crossattention.self.key.weight",
            "encoder.layer.2.crossattention.self.key.weight",
            "encoder.layer.4.crossattention.self.key.weight",
        ]
        for name in self_keys:
            if name in new_sd and isinstance(new_sd[name], torch.Tensor):
                kv_dim_ckpt = int(new_sd[name].shape[1]); break
        # 見つからなければ ".attention.key"（HF命名）も許容
        if kv_dim_ckpt is None:
            attn_keys = [
                "encoder.layer.0.crossattention.attention.key.weight",
                "encoder.layer.2.crossattention.attention.key.weight",
                "encoder.layer.4.crossattention.attention.key.weight",
            ]
            for name in attn_keys:
                if name in new_sd and isinstance(new_sd[name], torch.Tensor):
                    kv_dim_ckpt = int(new_sd[name].shape[1]); break
        kv_dim_cur = int(getattr(self.bert.config, "encoder_hidden_size",
                                 getattr(self.bert.config, "cross_attention_hidden_size",
                                         self.bert.config.hidden_size)))

        # ★ オプション化：ckpt側K/V次元に“合わせて”再構築するかはフラグで制御
        #    デフォルト(False)では、現在の設定を優先しミスマッチ重みは読み込みスキップ。
        if prefer_ckpt_kv_dim and kv_dim_ckpt is not None and kv_dim_ckpt != kv_dim_cur:
            self._maybe_rebuild_cross_kv(kv_dim_ckpt)
            kv_dim_cur = kv_dim_ckpt
            print(f"[qformer] prefer_ckpt_kv_dim=True -> rebuilt cross-attn K/V to {kv_dim_cur}")
        else:
            if kv_dim_ckpt is not None and kv_dim_ckpt != kv_dim_cur:
                print(f"[qformer] prefer_ckpt_kv_dim=False -> skip K/V reshape (ckpt={kv_dim_ckpt}, cur={kv_dim_cur})")

        # 形状不一致の cross-attn K/V を自動スキップ
        candidate = {}
        for k, v in new_sd.items():
            if re.search(r"^encoder\.layer\.\d+\.crossattention\.self\.(key|value)\.weight$", k):
                if v.dim() == 2 and v.shape[1] != kv_dim_cur:
                    continue
            # ".attention.(key|value).weight" もスキップ判定に含める
            if re.search(r"^encoder\.layer\.\d+\.crossattention\.attention\.(key|value)\.weight$", k):
                if v.dim() == 2 and v.shape[1] != kv_dim_cur:
                    continue
            candidate[k] = v

        own_b = set(self.bert.state_dict().keys())
        bert_sub = {k: v for k, v in candidate.items() if k in own_b}
        res = self.bert.load_state_dict(bert_sub, strict=False)
        print(f"[qformer] bert_loaded tensors={len(bert_sub)} | missing={len(res.missing_keys)} | unexpected={len(res.unexpected_keys)}")
        if res.missing_keys:
            for k in list(res.missing_keys)[:5]:
                print("  missing:", k)

        q_used = 0
        if isinstance(q_src, torch.Tensor):
            nq_src, dim_src = q_src.shape
            nq_tgt, dim_tgt = self.query_embed.shape
            if dim_src != dim_tgt:
                if dim_src >= dim_tgt:
                    q_src = q_src[:, :dim_tgt]
                else:
                    pad = torch.zeros(nq_src, dim_tgt - dim_src, dtype=q_src.dtype)
                    q_src = torch.cat([q_src, pad], dim=1)
            n_target = num_queries_target or nq_tgt
            if nq_src >= n_target:
                q_final = q_src[:n_target].clone()
            else:
                reps = (n_target + nq_src - 1) // nq_src
                q_final = q_src.repeat(reps, 1)[:n_target].clone()
            with torch.no_grad():
                self.query_embed.data.copy_(q_final.to(self.query_embed.dtype))
            q_used = n_target

        print(f"[qformer] pretrained loaded (approx): bert={len(bert_sub)} tensors, query_tokens_used={q_used}, via_hf={used_hf_loader}")
        return {"loaded": len(bert_sub), "q_tokens_used": q_used}

    def forward(
        self,
        vision_tokens: torch.Tensor,          # (B, L_total, C_vis)  ※上位で text_kv を結合済み
        feat_hw: Tuple[int, int],
        text_bias: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, L, C_vis = vision_tokens.shape
        queries = self.query_embed.unsqueeze(0).expand(B, -1, -1)
        if text_bias is not None:
            queries = queries + text_bias.unsqueeze(1)

        enc = vision_tokens  # 画像トークン + （必要なら）text_kv を上位で連結済みと想定
        attention_mask = torch.ones(B, self.num_queries, device=queries.device, dtype=torch.long)
        enc_mask       = torch.ones(B, enc.size(1),        device=queries.device, dtype=torch.long)

        out = self.bert(
            inputs_embeds=queries,
            attention_mask=attention_mask,
            encoder_hidden_states=enc,
            encoder_attention_mask=enc_mask,
            return_dict=True,
            output_attentions=True,
        )
        out_q = out.last_hidden_state      # (B, Nq, 768)

        # --- 類似度ヒートマップ（補助）。768/1408 の不一致に対応 ---
        Ht, Wt = feat_hw
        d_q = out_q.shape[-1]              # 768
        d_v = vision_tokens.shape[-1]      # 1024/1408 など

        if d_q != d_v:
            # lazy に Linear(d_v -> d_q) を用意
            if (self.sim_proj is None or
                self.sim_proj.in_features  != d_v or
                self.sim_proj.out_features != d_q):
                self.sim_proj = nn.Linear(d_v, d_q, bias=False).to(vision_tokens.device, dtype=vision_tokens.dtype)
                nn.init.xavier_uniform_(self.sim_proj.weight)

            # === AMP 下で入力と重みの dtype を必ず一致させる ===
            if vision_tokens.dtype != self.sim_proj.weight.dtype:
                vt_in = vision_tokens.to(self.sim_proj.weight.dtype)
            else:
                vt_in = vision_tokens

            vt = self.sim_proj(vt_in)           # (B, L, d_q)
        else:
            vt = vision_tokens                  # (B, L, d_q)

        # einsum 前に out_q 側の dtype に合わせておくと更に堅牢
        if vt.dtype != out_q.dtype:
            vt = vt.to(out_q.dtype)

        sim = torch.einsum(
            "bnc,blc->bnl",
            F.normalize(out_q, dim=-1),
            F.normalize(vt,     dim=-1),
        ).mean(dim=1)  # (B, L)

        # sim は類似度（相対的な可視化用）。bbox 抽出には使わない。
        # ここでは単に 0-1 に min-max 正規化したマップを「補助的に」返す。
        sim_img = sim[..., : Ht * Wt].view(B, Ht, Wt)
        sim_min = sim_img.amin(dim=(1,2), keepdim=True)
        sim_max = sim_img.amax(dim=(1,2), keepdim=True)
        sim_norm = (sim_img - sim_min) / (sim_max - sim_min + 1e-6)
        attn_logits_sim = sim_norm.unsqueeze(1)  # (B,1,Ht,Wt)  ※補助マップ（確率スケール）

        # --- Cross-Attn ヘッド重みからのヒートマップ（優先） ---
        if hasattr(out, "cross_attentions") and out.cross_attentions:
            # (B, heads, Nq, L_total). 画像トークン部分のみ取り出して head と query で平均。
            last = out.cross_attentions[-1]
            L_vis = Ht * Wt
            last_vis = last[..., :L_vis]                  # (B, heads, Nq, L_vis)
            w = last_vis.mean(dim=1).mean(dim=1)          # (B, L_vis), softmax済みの重みで 0-1
            w_img = w.view(B, Ht, Wt)
            # 0-1 に min-max 正規化（数値安定のため）
            w_min = w_img.amin(dim=(1,2), keepdim=True)
            w_max = w_img.amax(dim=(1,2), keepdim=True)
            w_norm = (w_img - w_min) / (w_max - w_min + 1e-6)
            attn_maps_head = w_norm.unsqueeze(1)          # (B,1,Ht,Wt) ← 確率マップ
        else:
            attn_maps_head = attn_logits_sim

        # 返り値は「確率マップ（0-1）」を第2引数として返す
        return out_q, attn_maps_head, attn_logits_sim


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
        "attn_logits": (B,1,Ht,Wt),
        "attn_maps":   (B,1,Ht,Wt),
        "feat_hw":     torch.LongTensor([Ht,Wt]),
        "vision_emb":  (B,Nq,768),
        "vis_prefix":  (B,Nq,proj_dim_out),
        "text_emb":    (B, 768) or None,
        "lm_loss_raw": Tensor() or 0.,
        "attn_logits_sim": (B,1,Ht,Wt)
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
                 load_vicuna: bool = False,
                 pretrained_qformer: Optional[str] = None,
                 text_cond_mode: str = "bias",   # "off" | "bias" | "bias+kv"
                 num_text_kv: int = 0,
                 qformer_kv_dim: Optional[int] = None):
        super().__init__()
        # Vision
        self.vision = vision_encoder if vision_encoder is not None else OpenCLIPVisionFrozen(model_name="ViT-g-14", pretrained="laion2b_s34b_b88k")
        for p in self.vision.parameters():
            p.requires_grad = False
        self.vision.eval()

        # Vision トークン実寸（OpenCLIP の出力次元）
        if hasattr(self.vision, "vis_dim"):
            vis_dim = int(self.vision.vis_dim)
        else:
            # fallback 推定
            sample = torch.zeros(1, 3, 224, 224)
            with torch.no_grad():
                out = self.vision(sample)
            vis_dim = int(out["tokens"].shape[-1])

        # Q-Former 側が期待する K/V 次元（Option B: 1408 を推奨）
        kv_dim_for_qformer = int(qformer_kv_dim) if (qformer_kv_dim is not None) else vis_dim

        # Q-Former（クエリ次元=proj_dim_in、K/V次元=kv_dim_for_qformer）
        if qformer is None:
            self.qformer = BertQFormer(hidden_size=proj_dim_in, encoder_hidden_size=kv_dim_for_qformer)
        else:
            self.qformer = qformer
            kv_dim_for_qformer = int(getattr(self.qformer.bert.config, "encoder_hidden_size",
                                             getattr(self.qformer.bert.config, "cross_attention_hidden_size",
                                                     proj_dim_in)))

        # num_queries を合わせる
        if hasattr(self.qformer, "num_queries") and self.qformer.num_queries != num_queries:
            C = self.qformer.query_embed.shape[-1]
            self.qformer.query_embed = nn.Parameter(torch.randn(num_queries, C) * 0.02)
            self.qformer.num_queries = num_queries

        # 事前学習Q-Formerロード（任意）
        if isinstance(pretrained_qformer, str) and len(pretrained_qformer) > 0:
            try:
                stats = self.qformer.load_pretrained_from_blip2(
                    pretrained_qformer,
                    num_queries_target=num_queries,
                    map_location="cpu",
                    prefer_ckpt_kv_dim=True,
                )
                print(f"[info] Q-Former pretrained loaded: {stats}")

                # --- Debug: cross-attn Q/K ロード状態を確認 ---
                def _stat(m):
                    with torch.no_grad():
                        return float(m.weight.abs().mean().item())

                enc = self.qformer.bert.encoder
                for i, layer in enumerate(enc.layer):
                    ca = layer.crossattention.self
                    print(f"[probe layer.{i}: |Q|={_stat(ca.quary):.6f} |K|={_stat(ca.key):.6f}")

                # ---K/V実寸の強制検証 ---
                try:
                    expect = int(getattr(self.qformer.bert.config, "encoder_hidden_size",
                                 getattr(self.qformer.bert.config, "cross_attention_hiddeen_size", 1408)))
                    k0 = self.qformer.bert.encoder.layer[0].crossattention.self.key
                    v0 = self.qformer.bert.encoder.layer[0].crossattention.self.value
                    kin = getattr(k0, "in_features", k0.weight.shape[1])
                    vin = getattr(v0, "in_features", v0.weight.shape[1])
                    print(f"[sanity] cross-attn K/V in_features={kin}/{vin} (expect={expect})")
                    assert kin == expect and vin == expect, f"cross-attn K/V in_features mismatch: {kin}/{vin} vs expect {expect}"
                except Exception as _e:
                    print(f"[warn] cross-attn K/V assertion skipped: {repr(_e)}")
            except Exception as e:
                print(f"[warn] failed to load pretrained Q-Former ({pretrained_qformer}): {e}")

        # ★ 重要: ロード後に最終の期待KV次元を取り直す
        try:
            kv_dim_for_qformer = int(getattr(self.qformer.bert.config, "encoder_hidden_size",
                                             getattr(self.qformer.bert.config, "cross_attention_hidden_size",
                                                     proj_dim_in)))
        except Exception:
            kv_dim_for_qformer = vis_dim

        # Projector（vision -> LLM hidden）
        self.projector = ProjectorMLP(proj_dim_in, proj_dim_out)

        # Vision トークン前の LayerNorm（実寸に合わせ forward 時に自動更新）
        self.pre_q_ln_vis = nn.LayerNorm(vis_dim)
        self.pre_q_ln_vis.requires_grad_(False)  # Vision 正規化は常に凍結

        # adapter を「ロード後の最終KV確定」後に判定・生成（ここが安全）
        self._vis_adapter: Optional[nn.Linear] = None
        if vis_dim != kv_dim_for_qformer:
            self._vis_adapter = nn.Linear(vis_dim, kv_dim_for_qformer, bias=False)
            nn.init.xavier_uniform_(self._vis_adapter.weight)
            print(f"[info] vis_adapter enabled: {vis_dim} -> {kv_dim_for_qformer}")
        else:
            print(f"[info] vis_adapter not required (vis_dim == q_kv_dim == {vis_dim})")

        # テキスト射影（Vicuna hidden -> 768）
        self.projector_text = None

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

            self.llm_hidden = int(self.lm.config.hidden_size)
            self.projector_text = nn.Linear(self.llm_hidden, proj_dim_in, bias=False)
            nn.init.xavier_uniform_(self.projector_text.weight)
            print(f"[info] text projector: {self.llm_hidden} -> 768 (Linear)")
        else:
            self.tokenizer = None
            self.lm = None

        self.text_kv_proj: Optional[nn.Linear] = None

        self.text_cond_mode = str(text_cond_mode).lower().strip()
        self.num_text_kv = int(num_text_kv)
        if self.text_cond_mode == "bias+kv":
            # text_emb は projector_text で 768 に揃う想定
            self.text_kv_proj = nn.Linear(proj_dim_in, kv_dim_for_qformer, bias=False)
            nn.init.xavier_uniform_(self.text_kv_proj.weight)

        # 保存：Q-Former 側の期待KV
        self._expected_kv_dim = int(kv_dim_for_qformer)

    @torch.no_grad()
    def _encode_vision(self, images: torch.Tensor) -> Dict[str, Any]:
        return self.vision(images)

    def _encode_text(self, texts: List[str]) -> Optional[torch.Tensor]:
        if not self.load_vicuna:
            return None
        tok = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_txt_len,
            return_tensors="pt"
        )
        dev = next(self.parameters()).device
        tok = {k: v.to(dev) for k, v in tok.items()}

        with torch.no_grad():
            out = self.lm.model(**tok, output_hidden_states=True, return_dict=True)
            last = out.hidden_states[-1]  # (B,T,hidden)
            lengths = tok["attention_mask"].sum(dim=1) - 1
            pooled = last[torch.arange(last.size(0), device=last.device), lengths]  # (B, hidden)
        # projector_text.weight の dtype に揃える（AMP 下でも安定）
        return pooled.to(dtype=self.projector_text.weight.dtype) if self.projector_text is not None else pooled

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
            loss = out.loss
        return loss

    def forward(self, inputs: Dict[str, Any], compute_lm: bool = False) -> Dict[str, Any]:
        images: torch.Tensor = inputs["image"]
        # text の型を吸収（str でも list でもOK）
        _t = inputs.get("text", [""] * images.size(0))
        if isinstance(_t, str):
            texts: List[str] = [_t] * images.size(0)
        else:
            texts = list(_t)

        # 1) Vision → tokens
        v = self._encode_vision(images)
        tokens: torch.Tensor = v["tokens"]
        feat_hw: Tuple[int, int] = v["feat_hw"]

        # 2) LN を実寸に合わせて張り替え（凍結のまま）
        Dv = int(tokens.size(-1))
        if (not hasattr(self, "pre_q_ln_vis")) or tuple(getattr(self.pre_q_ln_vis, "normalized_shape", ())) != (Dv,):
            self.pre_q_ln_vis = nn.LayerNorm(Dv).to(tokens.device, dtype=tokens.dtype) # AMP下でもdtypeをtokensに合わせる
            self.pre_q_ln_vis.requires_grad_(False)  # ここでも明示しておく
        tokens = self.pre_q_ln_vis(tokens)

        # 3) cross-attn が期待する K/V 入力次元（参照用）
        expected_kv = None
        try:
            ca_key = self.qformer.bert.encoder.layer[0].crossattention.self.key
            expected_kv = int(getattr(ca_key, "in_features", ca_key.weight.shape[1]))
        except Exception:
            pass
        if expected_kv is None:
            expected_kv = int(getattr(self.qformer.bert.config, "encoder_hidden_size",
                            getattr(self.qformer.bert.config, "cross_attention_hidden_size",
                                    self.qformer.bert.config.hidden_size)))

        # 4) Adapter 1024→1408（必要時）
        if self._vis_adapter is not None:
            tokens = self._vis_adapter(tokens)

        # forward() 内、アダプタ適用の直後
        if self._vis_adapter is None:
            assert tokens.size(-1) == self._expected_kv_dim, \
                f"vision tokens dim={tokens.size(-1)} but qformer expects {self._expected_kv_dim}"
        else:
            # adapter 経由時も実寸を確認
            assert tokens.size(-1) == self._expected_kv_dim, \
                f"[adapter] vision tokens dim={tokens.size(-1)} but qformer expects {self._expected_kv_dim}"

        # 5) LM loss
        lm_loss = self._compute_lm_loss(texts) if compute_lm else torch.tensor(0.0, device=images.device)

        # 6) テキスト条件化
        text_emb: Optional[torch.Tensor] = None
        text_bias: Optional[torch.Tensor] = None
        text_kv: Optional[torch.Tensor] = None

        if self.text_cond_mode in ("bias", "bias+kv"):
            if self.load_vicuna:
                raw = self._encode_text(texts)
                text_emb = raw
                if self.projector_text is not None and isinstance(raw, torch.Tensor):
                    text_emb = self.projector_text(raw)
                text_bias = text_emb

        # === AMP(bf16/fp16) 下で Linear に入れる前に dtype を一致させる ===
        try:
            w_dtype_kv = getattr(getattr(self, "text_kv_proj", None), "weight", None)
            w_dtype_kv = w_dtype_kv.dtype if w_dtype_kv is not None else None
        except Exception:
            w_dtype_kv = None
        try:
            w_dtype_q = None
            if hasattr(self.qformer, "bert"):
                w_dtype_q = next(self.qformer.bert.parameters()).dtype
        except Exception:
            w_dtype_q = None

        if isinstance(text_bias, torch.Tensor):
            if w_dtype_kv is not None and text_bias.dtype != w_dtype_kv:
                text_bias = text_bias.to(dtype=w_dtype_kv)
            elif w_dtype_kv is None and w_dtype_q is not None and text_bias.dtype != w_dtype_q:
                text_bias = text_bias.to(dtype=w_dtype_q)

        # bias+kv: テキストを cross-attn の K/V にも供給（上位で連結する）
        if (self.text_cond_mode == "bias+kv") and isinstance(text_bias, torch.Tensor) and (self.text_kv_proj is not None):
            T_kv = max(1, int(self.num_text_kv))
            kv_in = text_bias.to(self.text_kv_proj.weight.dtype)
            kv_tok = self.text_kv_proj(kv_in).unsqueeze(1).repeat(1, T_kv, 1)
            text_kv = kv_tok.to(dtype=tokens.dtype)

        # 7) Q-Former へ供給（text_kv はここで連結）
        tokens_for_ca = tokens if (text_kv is None) else torch.cat([tokens, text_kv], dim=1)

        if not hasattr(self, "_debug_once"):
            if os.getenv("DEBUG_QF", "0") == "1":
                k_lin = self.qformer.bert.encoder.layer[0].crossattention.self.key
                v_lin = self.qformer.bert.encoder.layer[0].crossattention.self.value
                print("[debug] enc_dim=", tokens_for_ca.size(-1),
                      "| key.in_features=", getattr(k_lin, "in_features", None),
                      "| value.in_features=", getattr(v_lin, "in_features", None))
                print("[debug] vis_tok_dim(before_cat)=", tokens.size(-1))
            self._debug_once = True

        out_q, attn_maps_head, attn_logits_sim = self.qformer(
            vision_tokens=tokens_for_ca,
            feat_hw=feat_hw,
            text_bias=text_bias,
        )

        # 8) LLM prefix
        vis_prefix = self.projector(out_q)

        # 以降は確率マップ（0-1）を使う
        attn_maps = attn_maps_head
        # ログ出力互換のために「ロジット風」も作っておく（評価側が logit を期待する場合に対応）
        attn_logits = torch.log(attn_maps.clamp(1e-6, 1-1e-6)) - torch.log1p(-attn_maps.clamp(1e-6, 1-1e-6))

        out: Dict[str, Any] = {
            "attn_logits": attn_logits,
            "attn_maps": torch.sigmoid(attn_logits),
            "feat_hw": torch.tensor([feat_hw[0], feat_hw[1]], device=images.device, dtype=torch.long),
            "vision_emb": out_q,
            "vis_prefix": vis_prefix,
            "text_emb": text_emb,
            "lm_loss_raw": lm_loss,
            "attn_logits_sim": attn_logits_sim,
            "vision_tokens": tokens,
        }
        return out
