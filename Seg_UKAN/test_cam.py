import cv2

print("正在尋找可用的攝影機索引...")

# 測試 0 到 4 號插槽
for index in range(5):
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW) # 先測 DSHOW
    if not cap.isOpened():
        # 如果 DSHOW 失敗，測預設值
        cap = cv2.VideoCapture(index)
        
    if cap.isOpened():
        ret, frame = cap.read()
        if ret:
            print(f"✅ 發現可用鏡頭！索引編號是: {index}")
        else:
            print(f"⚠️  索引 {index} 可以開啟，但讀不到畫面 (可能是紅外線鏡頭)")
        cap.release()
    else:
        print(f"❌ 索引 {index} 無法開啟")

print("測試結束")