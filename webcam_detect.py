import cv2
import time
import os
import threading
import pyttsx3 
from ultralytics import YOLO

# ================= 🔧 參數調整區 (控制台) =================
MODEL_PATH = "best.pt"
SAVE_FOLDER = "tongue_captures"

# 1. 信心度門檻
CONF_THRESHOLD = 0.7 

# 2. 距離/面積門檻 (畫面佔比)
SIZE_MIN = 0.08  # 太遠
SIZE_MAX = 0.70  # 太近/誤判

# 3. 形狀過濾 (長寬比)
AR_MIN = 0.6 
AR_MAX = 1.6

# 4. 倒數秒數
COUNTDOWN_SEC = 3

# 5. 🆕 邊緣保留區 (像素)
# 如果舌頭離邊緣小於這個距離，就視為被裁切
MARGIN = 15 
# ========================================================

# --- 初始化語音 ---
def speak(text):
    def _speak_thread():
        try:
            eng = pyttsx3.init() 
            eng.setProperty('rate', 150)
            eng.say(text)
            eng.runAndWait()
        except:
            pass
    threading.Thread(target=_speak_thread).start()

# --- 建立資料夾 ---
if not os.path.exists(SAVE_FOLDER):
    os.makedirs(SAVE_FOLDER)

# --- 載入模型 ---
print(f"🔍 檢查模型路徑: {os.path.abspath(MODEL_PATH)}")
if os.path.exists(MODEL_PATH):
    print(f"找到模型: {MODEL_PATH}")
    try:
        model = YOLO(MODEL_PATH)
    except Exception as e:
        print(f"載入失敗: {e}, 改用 yolov8n.pt")
        model = YOLO("yolov8n.pt")
else:
    print(f"!!! 找不到 {MODEL_PATH}，系統切換至 yolov8n.pt")
    model = YOLO("yolov8n.pt")

# --- 開啟攝影機 ---

#cap = cv2.VideoCapture(0, cv2.CAP_DSHOW) #看要步要改成這個應該會比較快
cap = cv2.VideoCapture(0, cv2.CAP_MSMF)
FONT = cv2.FONT_HERSHEY_SIMPLEX

# --- 狀態變數 ---
start_time = 0
counting = False
last_spoken_count = COUNTDOWN_SEC + 1
last_instruction_time = 0
current_status = "idle" 

print("🟢 程式啟動，請將舌頭對準框框")
speak("System Ready")

while cap.isOpened():
    success, frame = cap.read()
    if not success: 
        print("無法讀取鏡頭")
        break

    # 1. 鏡像翻轉
    frame = cv2.flip(frame, 1)
    clean_frame = frame.copy()
    
    frame_h, frame_w = frame.shape[:2]
    frame_area = frame_h * frame_w

    # 2. AI 預測
    results = model.predict(frame, verbose=False, conf=CONF_THRESHOLD)
    
    is_good_frame = False 
    
    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            # 確保座標在畫面內
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(frame_w, x2), min(frame_h, y2)

            # =========================================================
            # 🆕 新增邏輯：邊緣觸碰檢查 (Border/Edge Check)
            # =========================================================
            touching_edge = (
                x1 < MARGIN or              # 太靠左
                y1 < MARGIN or              # 太靠上
                x2 > frame_w - MARGIN or    # 太靠右
                y2 > frame_h - MARGIN       # 太靠下
            )

            if touching_edge:
                # 🔴 觸邊處理：顯示紅框、重置倒數、語音提示
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame, "CUT OFF!", (x1, y1 - 35), FONT, 0.8, (0, 0, 255), 2)
                cv2.putText(frame, "CENTER IT", (x1, y1 - 10), FONT, 0.8, (0, 0, 255), 2)
                
                counting = False # 重置倒數
                
                # 語音提示 (每 3 秒一次)
                if time.time() - last_instruction_time > 3:
                     speak("Center your tongue") 
                     last_instruction_time = time.time()
                
                continue # ⛔ 跳過這次迴圈 (不進行後面的合格判斷)
            # =========================================================

            # 計算幾何特徵
            w = x2 - x1
            h = y2 - y1
            box_area = w * h
            ratio = box_area / frame_area
            aspect_ratio = w / h if h > 0 else 0

            # --- 過濾器 ---
            
            # 過濾 1: 形狀不對
            if aspect_ratio < AR_MIN or aspect_ratio > AR_MAX:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (100, 100, 100), 1)
                continue 

            # 過濾 2: 面積太大 (雖然有邊緣檢查，但太滿也可能是誤判)
            if ratio > SIZE_MAX:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame, "Too Close", (x1, y1-10), FONT, 0.8, (0,0,255), 2)
                counting = False
                continue

            # --- 距離判斷與倒數邏輯 ---
            now = time.time()

            if ratio < SIZE_MIN:
                # 太遠
                color = (0, 0, 255)
                label = f"Too Far ({ratio:.1%})"
                instruction = "MOVE CLOSER"
                counting = False 
                
                if now - last_instruction_time > 3:
                    speak("Move Closer")
                    last_instruction_time = now

            else:
                # ✅ 合格 -> 倒數
                color = (0, 255, 0)
                label = f"Good ({ratio:.1%})"
                is_good_frame = True
                
                if not counting:
                    counting = True
                    start_time = now
                    last_spoken_count = COUNTDOWN_SEC + 1
                    instruction = "HOLD STILL"
                    speak("Hold still")
                else:
                    elapsed = now - start_time
                    remaining = COUNTDOWN_SEC - elapsed
                    current_count_int = int(remaining) + 1

                    if remaining > 0:
                        instruction = f"Wait... {current_count_int}"
                        # 中央大字倒數
                        cv2.putText(frame, str(current_count_int), 
                                    (int(frame_w/2)-30, int(frame_h/2)), 
                                    FONT, 4, (0, 255, 255), 5)
                        
                        if current_count_int < last_spoken_count:
                            speak(str(current_count_int))
                            last_spoken_count = current_count_int
                            
                    else:
                        # 📸 拍照
                        instruction = "CAPTURED!"
                        speak("Captured") 
                        
                        filename = f"{SAVE_FOLDER}/tongue_{int(time.time())}.png"
                        roi_img = clean_frame[y1:y2, x1:x2]
                        
                        # if roi_img.size > 0:
                        #     cv2.imwrite(filename, roi_img)
                        if roi_img.size > 0:
                            # 1. 取得原始寬高
                            h, w = roi_img.shape[:2]
                            
                            # 2. 計算需要補白的邊界 (使長寬一致，避免變形)
                            side = max(h, w)
                            pad_img = np.zeros((side, side, 3), dtype=np.uint8) # 建立黑色背景
                            # 如果想用白色背景，可以使用: pad_img.fill(255)
                            
                            # 將原始圖片置中貼上
                            x_offset = (side - w) // 2
                            y_offset = (side - h) // 2
                            pad_img[y_offset:y_offset+h, x_offset:x_offset+w] = roi_img
                            
                            # 3. 縮放到 400x400
                            final_output = cv2.resize(pad_img, (400, 400), interpolation=cv2.INTER_LANCZOS4)
                            
                            # 4. 儲存
                            cv2.imwrite(filename, final_output)
                            print(f"📸 已存檔 (400x400): {filename}")
                            
                            # 閃白效果與重置
                            cv2.rectangle(frame, (0, 0), (frame_w, frame_h), (255, 255, 255), -1)
                            counting = False
                            time.sleep(1)


            # 畫出框框與資訊
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
            cv2.putText(frame, label, (x1, y1 - 35), FONT, 0.6, color, 2)
            cv2.putText(frame, instruction, (x1, y1 - 10), FONT, 0.8, color, 2)

    # 無人/不合格狀態重置
    if not is_good_frame:
        counting = False
        if time.time() - last_instruction_time > 10 and time.time() - start_time > 10:
             last_instruction_time = time.time()

    cv2.imshow("Smart Capture", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()