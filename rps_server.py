# rps_server_with_preview.py
import cv2
import time
import mediapipe as mp
import math
import threading
from collections import deque, Counter
from flask import Flask, jsonify
from flask_cors import CORS  # ⬅️ NEW

# ====== CONFIG ======
CAM_INDEX = 0
FRAME_W, FRAME_H = 1280, 720
MIN_DET_CONF = 0.5
MIN_TRK_CONF = 0.5
MODEL_COMPLEXITY = 1
SMOOTH_WINDOW = 7          # กี่เฟรมสำหรับโหวตผล
SHOW_PREVIEW_INIT = True   # เริ่มต้นเปิดพรีวิว

# ====== MediaPipe Hands landmark indices ======
FINGER_IDS = {
    "thumb":  [1,2,3,4],
    "index":  [5,6,7,8],
    "middle": [9,10,11,12],
    "ring":   [13,14,15,16],
    "pinky":  [17,18,19,20],
}

# ====== Gesture history ต่อมือซ้าย/ขวา ======
gesture_histories = {
    "Left": deque(maxlen=SMOOTH_WINDOW),
    "Right": deque(maxlen=SMOOTH_WINDOW),
}
latest_result = {
    "updated_at": 0.0,
    "hands": [
        {"hand": "Left",  "gesture": "Unknown"},
        {"hand": "Right", "gesture": "Unknown"},
    ]
}
lock = threading.Lock()

def angle_between(a, b, c):
    BA = (a[0]-b[0], a[1]-b[1])
    BC = (c[0]-b[0], c[1]-b[1])
    dot = BA[0]*BC[0] + BA[1]*BC[1]
    norm_ba = math.hypot(*BA)
    norm_bc = math.hypot(*BC)
    if norm_ba == 0 or norm_bc == 0:
        return 0.0
    cosang = max(-1.0, min(1.0, dot/(norm_ba*norm_bc)))
    return math.degrees(math.acos(cosang))

def is_finger_extended(lm, ids, angle_thresh=160):
    mcp = (lm[ids[0]].x, lm[ids[0]].y)
    pip = (lm[ids[1]].x, lm[ids[1]].y)
    dip = (lm[ids[2]].x, lm[ids[2]].y)
    tip = (lm[ids[3]].x, lm[ids[3]].y)
    ang_pip = angle_between(mcp, pip, tip)
    ang_dip = angle_between(pip, dip, tip)
    return (ang_pip >= angle_thresh) and (ang_dip >= angle_thresh)

def is_thumb_extended(lm, angle_thresh=155):
    CMC, MCP, IP, TIP = FINGER_IDS["thumb"]
    mcp = (lm[MCP].x, lm[MCP].y)
    ip  = (lm[IP].x,  lm[IP].y)
    tip = (lm[TIP].x, lm[TIP].y)
    ang = angle_between(mcp, ip, tip)
    return ang >= angle_thresh

def count_extended_fingers(lm):
    return {
        "thumb":  is_thumb_extended(lm),
        "index":  is_finger_extended(lm, FINGER_IDS["index"]),
        "middle": is_finger_extended(lm, FINGER_IDS["middle"]),
        "ring":   is_finger_extended(lm, FINGER_IDS["ring"]),
        "pinky":  is_finger_extended(lm, FINGER_IDS["pinky"]),
    }

def classify_rps(lm):
    ext = count_extended_fingers(lm)
    if not any(ext.values()):
        return "Rock"
    if ext["index"] and ext["middle"] and ext["ring"] and ext["pinky"]:
        return "Paper"
    if ext["index"] and ext["middle"] and not ext["ring"] and not ext["pinky"]:
        return "Scissors"
    return "Unknown"

# ====== Flask app ======
app = Flask(__name__)

# ⬇️ ENABLE CORS (allow only your frontend origins)
CORS(app, resources={
    r"/rps": {"origins": ["http://localhost:3000", "http://127.0.0.1:3000"]},
    r"/health": {"origins": ["http://localhost:3000", "http://127.0.0.1:3000"]},
})

@app.get("/health")
def health():
    return jsonify({"status": "ok"})

@app.get("/rps")
def get_rps():
    with lock:
        data = {
            "status": "ok",
            "updated_at": latest_result["updated_at"],
            "window": SMOOTH_WINDOW,
            "hands": latest_result["hands"],
        }
    return jsonify(data)

def vision_worker():
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_styles  = mp.solutions.drawing_styles

    cap = cv2.VideoCapture(CAM_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)

    if not cap.isOpened():
        print(f"[ERROR] เปิดกล้องไม่ได้ (index={CAM_INDEX})")
        return

    print("[INFO] Vision thread started. Press 'v' to toggle preview, 'q' to close preview window.")

    show_preview = SHOW_PREVIEW_INIT
    t_prev = time.time()
    font = cv2.FONT_HERSHEY_SIMPLEX

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        model_complexity=MODEL_COMPLEXITY,
        min_detection_confidence=MIN_DET_CONF,
        min_tracking_confidence=MIN_TRK_CONF
    ) as hands:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame = cv2.flip(frame, 1)
            rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb)

            left_g = "Unknown"
            right_g = "Unknown"

            if result.multi_hand_landmarks and result.multi_handedness:
                for hand_lms, handedness in zip(result.multi_hand_landmarks, result.multi_handedness):
                    hlabel = handedness.classification[0].label
                    score  = handedness.classification[0].score
                    gesture = classify_rps(hand_lms.landmark)
                    gesture_histories[hlabel].append(gesture)

                    if show_preview:
                        mp_drawing.draw_landmarks(
                            frame, hand_lms, mp_hands.HAND_CONNECTIONS,
                            mp_styles.get_default_hand_landmarks_style(),
                            mp_styles.get_default_hand_connections_style()
                        )
                        x0 = int(hand_lms.landmark[0].x * frame.shape[1])
                        y0 = int(hand_lms.landmark[0].y * frame.shape[0])
                        cv2.putText(frame, f"{hlabel} {score:.2f}", (x0+8, y0-8),
                                    font, 0.7, (0,255,0), 2, cv2.LINE_AA)

                if len(gesture_histories["Left"]) > 0:
                    left_g = Counter(gesture_histories["Left"]).most_common(1)[0][0]
                if len(gesture_histories["Right"]) > 0:
                    right_g = Counter(gesture_histories["Right"]).most_common(1)[0][0]

                if show_preview:
                    for hand_lms, handedness in zip(result.multi_hand_landmarks, result.multi_handedness):
                        hlabel = handedness.classification[0].label
                        g = left_g if hlabel == "Left" else right_g
                        x0 = int(hand_lms.landmark[0].x * frame.shape[1])
                        y0 = int(hand_lms.landmark[0].y * frame.shape[0])
                        cv2.putText(frame, f"{g}", (x0+8, y0+24),
                                    font, 0.8, (255,255,255), 2, cv2.LINE_AA)

            with lock:
                latest_result["updated_at"] = time.time()
                latest_result["hands"][0]["gesture"] = left_g
                latest_result["hands"][1]["gesture"] = right_g

            t_now = time.time()
            fps = 1.0 / max(t_now - t_prev, 1e-8)
            t_prev = t_now

            if show_preview:
                cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                            font, 1.0, (0,255,0), 2, cv2.LINE_AA)
                cv2.imshow("RPS Preview", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('v'):
                    show_preview = False
                    cv2.destroyWindow("RPS Preview")
                elif key == ord('q'):
                    show_preview = False
                    cv2.destroyWindow("RPS Preview")
            else:
                key = cv2.waitKey(1) & 0xFF
                if key == ord('v'):
                    show_preview = True

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Vision thread stopped (camera released).")

def main():
    t = threading.Thread(target=vision_worker, daemon=True)
    t.start()
    # เปิดแบบ threaded เพื่อให้บริการ /rps ได้ขณะ vision ทำงาน
    app.run(host="0.0.0.0", port=5000, threaded=True, debug=False)

if __name__ == "__main__":
    main()
