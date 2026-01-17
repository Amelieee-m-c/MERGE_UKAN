import cv2
import time
import os
import threading
import pyttsx3 
from ultralytics import YOLO

# ================= 🔧 參數調整區 (控制台) =================
MODEL_PATH = "best.pt"
SAVE_FOLDER = "tongue_captures"
# 修改這裡：指向 Windows 的 IP。WSL2 預設可以用這個網址存取主機
STREAM_URL = "http://172.17.192.1:5000/video" 

CONF_THRESHOLD = 0.7 
SIZE_MIN = 0.10  
SIZE_MAX = 0.70  
AR_MIN = 0.6 
AR_MAX = 1.6
COUNTDOWN_SEC = 3
MARGIN = 15 
# ========================================================

# --- 初始化語音 (增加 WSL 相容性處理) ---
def speak(text):
    def _speak_thread():
        try:
            eng = pyttsx3.init() 
            eng.setProperty('rate', 150)
            eng.say(text)
            eng.runAndWait()
        except Exception as e:
            # WSL 如果沒裝音效驅動會報錯，這裡列印出來但不中斷程式
            print(f"🔊 語音提醒: {text} (TTS Error: {e})")
    threading.Thread(target=_speak_thread).start()

if not os.path.exists(SAVE_FOLDER):
    os.makedirs(SAVE_FOLDER)

# --- 載入模型 ---
print(f"🔍 檢查模型路徑: {os.path.abspath(MODEL_PATH)}")
model = YOLO(MODEL_PATH if os.path.exists(MODEL_PATH) else "yolov8n.pt")

# --- 修改處：開啟網路影像串流而非本地鏡頭 ---
print(f"🌐 正在連線至 Windows 影像串流: {STREAM_URL}")
cap = cv2.VideoCapture(STREAM_URL)

# 檢查連線是否成功
if not cap.isOpened():
    print("❌ 無法連線至影像串流！請確保 Windows 端的 host_camera.py 正在執行。")
    exit()

FONT = cv2.FONT_HERSHEY_SIMPLEX
start_time = 0
counting = False
last_spoken_count = COUNTDOWN_SEC + 1
last_instruction_time = 0
current_status = "idle" 

print("🟢 程式啟動，影像來源：網路串流")
speak("System Ready")

while True:
    success, frame = cap.read()
    if not success: 
        print("⚠️ 串流中斷或無法讀取")
        # 嘗試重新連線
        cap = cv2.VideoCapture(STREAM_URL)
        time.sleep(1)
        continue

    # 1. 鏡像翻轉 (如果 Windows 端已經翻轉過，這裡可以註解掉)
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
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(frame_w, x2), min(frame_h, y2)

            touching_edge = (
                x1 < MARGIN or y1 < MARGIN or 
                x2 > frame_w - MARGIN or y2 > frame_h - MARGIN
            )

            if touching_edge:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame, "CENTER IT", (x1, y1 - 10), FONT, 0.8, (0, 0, 255), 2)
                counting = False 
                if time.time() - last_instruction_time > 3:
                     speak("Center your tongue") 
                     last_instruction_time = time.time()
                continue 

            w, h = x2 - x1, y2 - y1
            box_area = w * h
            ratio = box_area / frame_area
            aspect_ratio = w / h if h > 0 else 0

            if aspect_ratio < AR_MIN or aspect_ratio > AR_MAX:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (100, 100, 100), 1)
                continue 

            if ratio > SIZE_MAX:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame, "Too Close", (x1, y1-10), FONT, 0.8, (0,0,255), 2)
                counting = False
                continue

            now = time.time()
            if ratio < SIZE_MIN:
                color, label, instruction = (0, 0, 255), f"Too Far ({ratio:.1%})", "MOVE CLOSER"
                counting = False 
                if now - last_instruction_time > 3:
                    speak("Move Closer")
                    last_instruction_time = now
            else:
                color, label = (0, 255, 0), f"Good ({ratio:.1%})"
                is_good_frame = True
                if not counting:
                    counting, start_time, last_spoken_count = True, now, COUNTDOWN_SEC + 1
                    instruction = "HOLD STILL"
                    speak("Hold still")
                else:
                    remaining = COUNTDOWN_SEC - (now - start_time)
                    current_count_int = int(remaining) + 1
                    if remaining > 0:
                        instruction = f"Wait... {current_count_int}"
                        cv2.putText(frame, str(current_count_int), (int(frame_w/2)-30, int(frame_h/2)), FONT, 4, (0, 255, 255), 5)
                        if current_count_int < last_spoken_count:
                            speak(str(current_count_int))
                            last_spoken_count = current_count_int
                    else:
                        instruction = "CAPTURED!"
                        speak("Captured") 
                        filename = f"{SAVE_FOLDER}/tongue_{int(time.time())}.jpg"
                        roi_img = clean_frame[y1:y2, x1:x2]
                        if roi_img.size > 0:
                            cv2.imwrite(filename, roi_img)
                            print(f"📸 已存檔: {filename}")
                            cv2.rectangle(frame, (0, 0), (frame_w, frame_h), (255, 255, 255), -1)
                        counting = False
                        time.sleep(1) 

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
            cv2.putText(frame, label, (x1, y1 - 35), FONT, 0.6, color, 2)
            cv2.putText(frame, instruction, (x1, y1 - 10), FONT, 0.8, color, 2)

    if not is_good_frame:
        counting = False

    cv2.imshow("WSL Smart Capture", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()