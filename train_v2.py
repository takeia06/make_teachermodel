import sys
import os
import torch
from torch.utils.data import DataLoader
from transformers import AdamW, get_scheduler
from yacs.config import CfgNode as CN

# --- 必要なモジュールを全てインポート ---
# （ユーザー指定のカスタムパスを追加するコードもここに含みます）
data_module_path = "/home/takei/data"
if data_module_path not in sys.path:
    sys.path.append(data_module_path)

from model_v1 import load_components
from grounding_anomaly_dataset import GroundingDataset, collate_fn
from lib.losses import SetCriterion
from lib.matcher import build_matcher

from groundingdino.util.slconfig import SLConfig
from groundingdino.models import build_model
from groundingdino.util.misc import clean_state_dict

torch.autograd.set_detect_anomaly(True)

def main():
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"✅ Using device: {DEVICE}")

    gd_config_path = "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
    args = SLConfig.fromfile(gd_config_path)
    args.device = DEVICE

    model = build_model(args)
    model.to(DEVICE)
    model.train()
    print("✅ GroundingDINO model is built and in training mode.")

    matcher = build_matcher(args)
    print("✅ Matcher is built.")
    
    weight_dict = {"loss_ce": 1, "loss_bbox": 5.0, "loss_giou": 2.0}
    criterion = SetCriterion(
        num_classes=256,
        matcher=matcher,
        weight_dict=weight_dict,
        focal_alpha=0.25,
        losses=["labels", "boxes", "cardinality"]
    ).to(DEVICE)
    print("✅ Loss criterion is ready.")

    dataset = GroundingDataset(
        json_path="/home/takei/data/COCO/annotations/training_final.json",
        image_root="/home/takei/data/COCO/train2017"
    )
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=collate_fn)
    print("✅ Dummy dataset loaded successfully.")

    # ★★★ ここに健全性チェックコードを追加 ★★★
    print("\n" + "="*20 + " 🕵️‍ DATA SANITY CHECK 🕵️ " + "="*20)
    check_passed = True
    for i, (images, captions, targets) in enumerate(dataloader):
        if i >= 5: # 最初の5バッチだけチェック
            break
        if images is None:
            print(f"Batch {i}: Skipped an empty batch.")
            continue
            
        print(f"--- Checking Batch {i} ---")
        print(f"  Captions: {captions[0][:50]}...") # キャプションの先頭を表示
        target = targets[0] # バッチサイズ1なので最初の要素を取得
        boxes = target['boxes']
        print(f"  Num boxes: {len(boxes)}")

        # ボックスの値が[0,1]の範囲内か、w,hが正の値かチェック
        if (boxes < 0.0).any() or (boxes > 1.0).any():
            print(f"  ❌ ERROR: Box coordinates are out of [0, 1] range!")
            print(boxes)
            check_passed = False
        
        if (boxes[:, 2] <= 0).any() or (boxes[:, 3] <= 0).any():
            print(f"  ❌ ERROR: Box width or height is not positive!")
            print(boxes)
            check_passed = False
            
    if not check_passed:
        print("="*20 + " 😭 Sanity check failed. Please fix data processing. 😭 " + "="*20)
        return # 問題があればトレーニングを中止
    else:
        print("="*20 + " 👍 Sanity check passed! All good to go. 👍 " + "="*20)
    # ★★★ 健全性チェックここまで ★★★


    # ★★★ 修正点1: 学習率を少し下げる ★★★
    optimizer = AdamW(model.parameters(), lr=1e-5, weight_decay=1e-4) # 1e-5 から 1e-6 へ
    num_epochs = 3
    lr_scheduler = get_scheduler(
        "linear", optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_epochs * len(dataloader),
    )

    print("\n" + "="*20 + " 🚀 STARTING TRAINING 🚀 " + "="*20)
    for epoch in range(num_epochs):
        model.train()
        criterion.train()
        
        for i, (images, captions, targets) in enumerate(dataloader):
            images = images.to(DEVICE)
            targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]
            
            outputs = model(images, captions=captions)

            if torch.isinf(outputs['pred_logits']).any():
                outputs['pred_logits'] = torch.nan_to_num(
                    outputs['pred_logits'], nan=0.0, posinf=1e+4, neginf=-1e+4
                )

            loss_dict = criterion(outputs, targets)
            loss = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

            # ★★★ ここにデバッグ用のprint文を追加 ★★★
            if torch.isnan(loss):
                print("\n" + "="*20 + " 🚨 NaN DETECTED! 🚨 " + "="*20)
                print(f"--- Problematic Batch Info ---")
                print(f"  Captions: {captions}")
                print(f"  Targets: {targets}")
                
                print("\n--- Model Outputs (pred_logits) ---")
                print(f"  Shape: {outputs['pred_logits'].shape}")
                print(f"  Has NaN: {torch.isnan(outputs['pred_logits']).any()}")
                print(f"  Min value: {outputs['pred_logits'].min()}")
                print(f"  Max value: {outputs['pred_logits'].max()}")

                print("\n--- Model Outputs (pred_boxes) ---")
                print(f"  Shape: {outputs['pred_boxes'].shape}")
                print(f"  Has NaN: {torch.isnan(outputs['pred_boxes']).any()}")
                print(f"  Min value: {outputs['pred_boxes'].min()}")
                print(f"  Max value: {outputs['pred_boxes'].max()}")

                print("\n--- Loss Dictionary ---")
                for k, v in loss_dict.items():
                    print(f"  {k}: {v.item()}")
                
                print("="*40 + "\n")
               

            optimizer.zero_grad()
            loss.backward()
            
            # ★★★ 修正点2: 勾配クリッピングを追加（学習の安定化） ★★★
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
            
            optimizer.step()
            lr_scheduler.step()

            # .item()でPythonの数値に変換して表示
            loss_value = loss.item() if not torch.isnan(loss) else "nan"
            print(f"Epoch: {epoch+1}/{num_epochs} | Step: {i+1}/{len(dataloader)} | Loss: {loss_value}")

    print("="*20 + " ✅ TRAINING COMPLETE ✅ " + "="*20 + "\n")

    # ★★★ ご要望の機能: 学習済みモデルの保存 ★★★
    print("💾 Saving trained model...")
    save_dir = "ckpt/GA"
    os.makedirs(save_dir, exist_ok=True) # 保存用ディレクトリを作成
    save_path = os.path.join(save_dir, "groundingdino_finetuned.pth")
    torch.save(model.state_dict(), save_path)
    print(f"✅ Model saved to: {save_path}")


if __name__ == "__main__":
    main()