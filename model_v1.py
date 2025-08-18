import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM

# --- GroundingDINO & ImageBind モジュールのインポート ---
# これらのインポートが成功するためには、GroundingDINOのセットアップが完了している必要があります
try:
    from GroundingDINO.groundingdino.util.inference import load_model
    from GroundingDINO.groundingdino.util.misc import NestedTensor
    from GroundingDINO.groundingdino.models.GroundingDINO.backbone.position_encoding import PositionEmbeddingSine
    from model.ImageBind.models.imagebind_model import imagebind_huge, ModalityType
    from model.ImageBind.data import load_and_transform_vision_data
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Please ensure GroundingDINO and its dependencies are correctly installed and accessible in your PYTHONPATH.")
    raise

# --- アダプター層の定義 ---
class AdapterLayer(nn.Module):
    def __init__(self, input_dim=1024, output_dim=256, sequence_length=256):
        super().__init__()
        self.projection = nn.Linear(input_dim, output_dim * sequence_length)
        self.output_dim = output_dim
        self.sequence_length = sequence_length
        self.layer_norm = nn.LayerNorm(output_dim)
        print(f"✅ Adapter Layer initialized: maps ({input_dim}) -> ({sequence_length}, {output_dim})")

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        projected = self.projection(features)
        reshaped = projected.view(-1, self.sequence_length, self.output_dim)
        return self.layer_norm(reshaped)

# --- 新アーキテクチャの組み立て ---
class GroundingDINO_AnomalyGPT_v2(nn.Module):
    def __init__(self, grounding_dino_model, image_encoder, llm, llm_tokenizer, device='cuda'):
        super().__init__()
        self.device = device
        self.grounding_dino = grounding_dino_model
        self.image_encoder = image_encoder
        self.llm = llm
        self.llm_tokenizer = llm_tokenizer
        
        # --- 各コンポーネントをアーキテクチャに統合 ---
        self.adapter = AdapterLayer().to(device)
        self.feature_projection = nn.Linear(256, self.llm.config.hidden_size).to(device)
        self.pos_embed_generator = PositionEmbeddingSine(num_pos_feats=128, temperature=20, normalize=True).to(device)
        
        print("✅ GroundingDINO_AnomalyGPT_v2 architecture is ready.")

    def forward(self, image_path: str, text_prompt: str):
        """
        推論（Inference）用のフォワードパス
        """
        print("\n--- 🚀 Starting Inference Forward Pass 🚀 ---")
        
        with torch.no_grad():
            # 1. ImageBindで画像特徴を抽出
            vision_inputs = load_and_transform_vision_data([image_path], self.device)
            vision_outputs = self.image_encoder({ModalityType.VISION: vision_inputs})[ModalityType.VISION]
            image_features_1024d = vision_outputs[0]
            print(f"   1. ImageBind output shape: {image_features_1024d.shape}")

            # 2. Adapterで特徴量をGroundingDINOの形式に変換
            adapted_features_256d = self.adapter(image_features_1024d)
            print(f"   2. Adapter output shape: {adapted_features_256d.shape}")

            # 3. GroundingDINOのTransformerで画像とテキストを融合
            # ... (学習ロジックとは別の、推論用の詳細な処理) ...
            # この部分は推論時に実装しますが、まずは学習を優先します
            
            # (仮の出力)
            fused_features = adapted_features_256d # 本来はDINOのTransformerからの出力
            print(f"   ✅ (Placeholder) Features fused. Shape: {fused_features.shape}")

            # 4. LLMに入力して応答を生成
            avg_features = fused_features.mean(dim=1)
            projected_for_llm = self.feature_projection(avg_features).unsqueeze(1)

            # プロンプトを組み立て
            PROMPT_START = "###Human: <Img>"
            PROMPT_END   = "</Img> " + text_prompt + "###Assistant:"
            start_tokens = self.llm_tokenizer(PROMPT_START, return_tensors="pt")
            end_tokens = self.llm_tokenizer(PROMPT_END, return_tensors="pt")
            start_embeds = self.llm.get_input_embeddings()(start_tokens.input_ids.to(self.device))
            end_embeds = self.llm.get_input_embeddings()(end_tokens.input_ids.to(self.device))
            
            inputs_embeds = torch.cat([start_embeds, projected_for_llm, end_embeds], dim=1)
            
            # LLMで応答生成
            outputs = self.llm.generate(
                inputs_embeds=inputs_embeds, max_new_tokens=128, do_sample=False,
                pad_token_id=self.llm_tokenizer.pad_token_id
            )
            response_tokens = outputs[0][inputs_embeds.shape[1]:]
            response = self.llm_tokenizer.decode(response_tokens, skip_special_tokens=True)

        print("--- ✅ Inference Forward Pass Complete ---")
        return response

        # model_v1.py の GroundingDINO_AnomalyGPT_v2 クラス内に追加

    def forward_train(self, images: NestedTensor, text_prompts: list[str]):
        """
        学習（Training）用のフォワードパス
        """
        print("\n--- 💡 Starting Training Forward Pass 💡 ---")
        
        # 1. ImageBindで画像特徴を抽出
        #    (注: データローダーから受け取る'images'は既にテンソルなので、パスではなくそのまま使います)
        with torch.no_grad(): # ImageBindの重みは固定
            vision_outputs = self.image_encoder({ModalityType.VISION: images.tensors})[ModalityType.VISION]
            image_features_1024d = vision_outputs[0]

        # 2. Adapterで特徴量をGroundingDINOの形式に変換
        adapted_features_256d = self.adapter(image_features_1024d)

        # 3. GroundingDINOのTransformerに入力するために形式を整える
        hw = int(adapted_features_256d.shape[1] ** 0.5)
        srcs = [adapted_features_256d.permute(0, 2, 1).reshape(adapted_features_256d.shape[0], 256, hw, hw)]
        masks = [torch.zeros(s.shape[0], s.shape[2], s.shape[3], device=self.device, dtype=torch.bool) for s in srcs]
        pos_embeds = [self.pos_embed_generator(NestedTensor(s, m)).to(s.dtype) for s, m in zip(srcs, masks)]

        # 4. テキストプロンプトをエンコード
        tokenizer = self.grounding_dino.tokenizer
        tokenized = tokenizer(text_prompts, padding='longest', return_tensors="pt").to(self.device)
        bert_output = self.grounding_dino.bert(**tokenized)
        text_features_768d = bert_output['last_hidden_state']
        projected_text_features = self.grounding_dino.feat_map(text_features_768d)
        
        text_dict = {
            "encoded_text": projected_text_features,
            "text_token_mask": tokenized["attention_mask"].bool(),
        }

        # 5. GroundingDINOのTransformerで画像とテキストを融合
        #    (この部分が、学習したい核心部分です)
        hs, _, _, _, _ = self.grounding_dino.transformer(srcs, masks, None, pos_embeds, None, text_dict=text_dict)
        
        # 6. 損失計算のために、予測結果を整形して出力
        output_class = self.grounding_dino.class_embed(hs)
        output_coord = self.grounding_dino.bbox_embed(hs).sigmoid()
        
        # 損失関数が期待する辞書形式で返す
        return {'pred_logits': output_class[-1], 'pred_boxes': output_coord[-1]}

# --- コンポーネントのロード関数 ---
def load_components(cfg, device='cuda'):
    """
    事前学習済みの各コンポーネントをロードするヘルパー関数
    """
    print("🔄 Loading components...")
    
    # ImageBind
    image_encoder, _ = imagebind_huge(pretrained=True)
    image_encoder.eval().to(device)
    print("   ✅ ImageBind model loaded.")
    
    # LLM
    llm = AutoModelForCausalLM.from_pretrained(cfg.model.llm.pretrained_path, torch_dtype=torch.float16).eval().to(device)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.llm.pretrained_path, use_fast=False)
    print("   ✅ LLM and Tokenizer loaded.")
    
    # GroundingDINO
    gd_model = load_model(cfg.model.gdino.config_path, cfg.model.gdino.weights_path).eval().to(device)
    print("   ✅ GroundingDINO model loaded.")
    
    return image_encoder, llm, tokenizer, gd_model