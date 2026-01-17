import os
import torch
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from model import SignOrientedNetwork
import albumentations as A
from albumentations.pytorch import ToTensorV2

# 1. 設定
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_PATH = 'signnet_best_fold1.pth'  # 你的權重檔名
UKAN_DIR = 'mimi/ukan'                 # UKAN 圖片路徑
LABEL_COLS = ['TonguePale', 'TipSideRed', 'Spot', 'Ecchymosis',
              'Crack', 'Toothmark', 'FurThick', 'FurYellow']

# 2. 預處理函數 (必須與訓練時一致)
def get_three_views(img_np):
    resized = cv2.resize(img_np, (224, 224), interpolation=cv2.INTER_LINEAR)
    gray = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY)
    _, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    kernel_size = int(np.sqrt(224**2 + 224**2) * 0.191)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    mask_eroded = cv2.erode(mask, kernel, iterations=1)
    mask_edge = cv2.subtract(mask, mask_eroded)
    mask_edge[:224 // 4, :] = 0
    mask_body = cv2.subtract(mask, mask_edge)
    body_img = cv2.bitwise_and(resized, resized, mask=mask_body)
    edge_img = cv2.bitwise_and(resized, resized, mask=mask_edge)
    return resized, body_img, edge_img

transform = A.Compose([
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])

# 3. 載入模型
print(f"🔄 正在載入模型: {MODEL_PATH}")
checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
model = SignOrientedNetwork(num_classes=len(LABEL_COLS), backbone='swin_base_patch4_window7_224')
model.load_state_dict(checkpoint['model_state_dict'])
model.to(DEVICE)
model.eval()

# 獲取訓練時的最佳閾值 (如果沒有就用 0.5)
best_thresholds = checkpoint.get('best_thresholds', [0.5] * len(LABEL_COLS))

# 4. 開始預測
results = []
img_files = [f for f in os.listdir(UKAN_DIR) if f.endswith(('.png', '.jpg', '.jpeg'))]

print(f"🚀 開始預測 {len(img_files)} 張圖片...")

with torch.no_grad():
    for f in img_files:
        path = os.path.join(UKAN_DIR, f)
        img_np = np.array(Image.open(path).convert('RGB'))
        
        # 處理成三分支輸入
        w, b, e = get_three_views(img_np)
        w_t = transform(image=w)['image'].unsqueeze(0).to(DEVICE)
        b_t = transform(image=b)['image'].unsqueeze(0).to(DEVICE)
        e_t = transform(image=e)['image'].unsqueeze(0).to(DEVICE)
        
        # 推論
        logits = model(w_t, b_t, e_t)['final']
        probs = torch.sigmoid(logits).cpu().numpy()[0]
        
        # 根據閾值判定結果
        pred_tags = {}
        for i, col in enumerate(LABEL_COLS):
            pred_tags[col] = 1 if probs[i] >= best_thresholds[i] else 0
            # 同時存入機率值方便後續分析
            pred_tags[f"{col}_prob"] = round(float(probs[i]), 4)
            
        pred_tags['filename'] = f
        results.append(pred_tags)

# 5. 輸出結果
df_res = pd.DataFrame(results)
df_res.to_csv('mimi_prediction_results.csv', index=False)
print("✅ 預測完成！結果已儲存至 mimi_prediction_results.csv")
print(df_res[['filename'] + LABEL_COLS].head()) # 顯示前幾筆看看