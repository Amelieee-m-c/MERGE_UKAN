import argparse
import os
import yaml
import torch
import numpy as np
import cv2
from glob import glob
from tqdm import tqdm
from PIL import Image
from albumentations import Resize, Compose
from albumentations.augmentations import transforms
import archs

def main():
    # 1. 載入配置
    config_path = 'outputs/_UKAN/config.yml' 
    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # 2. 模型初始化
    model = archs.__dict__[config['arch']](
        config['num_classes'], 
        config['input_channels'], 
        config['deep_supervision'], 
        embed_dims=config['input_list']
    )
    model = model.cuda()
    model.load_state_dict(torch.load('outputs/_UKAN/model.pth'))
    model.eval()

    # 3. 預處理 - 第一步：Resize 與 Normalize
    # 注意：這裡的 Normalize 必須跟訓練時完全一致
    val_transform = Compose([
        Resize(config['input_h'], config['input_w']),
        transforms.Normalize(),
    ])

    input_dir = 'mimi/images'
    output_dir = 'my_results'
    os.makedirs(output_dir, exist_ok=True)
    img_paths = glob(os.path.join(input_dir, '*.*'))

    with torch.no_grad():
        for img_p in tqdm(img_paths):
            img_id = os.path.splitext(os.path.basename(img_p))[0]
            
            # 讀取 (OpenCV 預設 BGR)
            img = cv2.imread(img_p)
            if img is None: continue
            
            # A. 執行 Albumentations (包含 Normalize)
            augmented = val_transform(image=img)
            img_processed = augmented['image']

            # B. 執行 Dataset 裡的特殊轉換 (這是關鍵！)
            # 訓練時代碼：img = img.astype('float32') / 255
            img_processed = img_processed.astype('float32') / 255
            
            # C. 轉為 Tensor (Transpose 2,0,1)
            img_input = torch.from_numpy(img_processed).permute(2, 0, 1).unsqueeze(0).cuda()

            # 4. 推理
            output = model(img_input)
            output = torch.sigmoid(output).cpu().numpy()
            
            # 5. 後處理 (對應你的 val.py)
            output[output >= 0.5] = 1
            output[output < 0.5] = 0
            pred_np = (output[0, 0] * 255).astype(np.uint8)

            # 6. 儲存
            Image.fromarray(pred_np, 'L').save(os.path.join(output_dir, f'{img_id}.png'))

    print(f"修正完成！請檢查 {output_dir}")

if __name__ == '__main__':
    main()