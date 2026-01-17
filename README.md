## 串接TOLO+UKAN+三分支

### 介紹資料夾
**Seg-UKAN**
-**舌頭偵測**
- `webcam_detect.py` 偵測舌頭的主程式
- `best.pt` 為`webcam_detect.py`所需的模型

### ❓因為實驗室鏡頭沒有裝上去所以目前不能測試，可能會缺少.pth檔案

-**UKAN**
- cvc_UKAN 、inputs、val.py不要理他那是之前拿來訓練用的
    - `mimi`  這個資料夾裡面放的是我們用 `webcam_detect.py`所拍攝的舌頭
    - `my_results`   裡面為 `webcam_detect.py`所拍攝的舌頭，進行UKAN分割後的圖檔
    - 💡`inference.py` 用 `webcam_detect.py`所拍攝的舌頭進行UKAN分割的python檔案
    - `archs.py` 為UKAN原作者所者所寫
    - `kan.py` 為UKAN原作者所者所寫
    - `utils.py` 為UKAN原作者所者所寫
    - `archs.py` 為UKAN原作者所者所寫

**Tongue-AI-V2**
-大部分皆為曹哲維之前的三分支
    - 💡`inference.py` 將圖檔進行八類推論，出來的是csv檔案，裡面是用sigmoid機率去計算，可以替換.pth
---

### 安裝

requirements.txt為UKAN所需套件 UKAN是用WSL22.04的環境下所進行
`webcam_detect.py` 本地端跑即可
**Tongue-AI-V2**這個資料夾甚麼環境下都沒什麼差