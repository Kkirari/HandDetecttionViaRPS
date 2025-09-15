import cv2
import time
import mediapipe as mp
import math
from collections import deque, Counter

# MediaPipe Hands landmark indices
# Thumb: CMC(1), MCP(2), IP(3), TIP(4)
# Index: MCP(5), PIP(6), DIP(7), TIP(8)
# Middle: MCP(9), PIP(10), DIP(11), TIP(12)
# Ring: MCP(13), PIP(14), DIP(15), TIP(16)
# Pinky: MCP(17), PIP(18), DIP(19), TIP(20)

FINGER_IDS = {
    "thumb":  [1,2,3,4],
    "index":  [5,6,7,8],
    "middle": [9,10,11,12],
    "ring":   [13,14,15,16],
    "pinky":  [17,18,19,20],
}

# buffer สำหรับโหวตผลล่าสุด (ทำให้ผลนิ่งขึ้น)
GESTURE_HISTORY = deque(maxlen=7)  # ลอง 7 เฟรม

def angle_between(a, b, c):
    """มุมที่จุด b (องศา) จากเวกเตอร์ BA และ BC; a,b,c เป็น (x,y) แบบ normalized"""
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
    """
    ตัดสินใจจากมุมที่ข้อ PIP และ DIP:
    - ถ้ามุมทั้งสอง 'ตรง' (>= angle_thresh) ถือว่าเหยียด
    lm: result.multi_hand_landmarks[?].landmark (list of 21)
    ids: [MCP, PIP, DIP, TIP]
    """
    mcp = (lm[ids[0]].x, lm[ids[0]].y)
    pip = (lm[ids[1]].x, lm[ids[1]].y)
    dip = (lm[ids[2]].x, lm[ids[2]].y)
    tip = (lm[ids[3]].x, lm[ids[3]].y)

    # มุมที่ PIP: MCP-PIP-TIP
    ang_pip = angle_between(mcp, pip, tip)
    # มุมที่ DIP: PIP-DIP-TIP
    ang_dip = angle_between(pip, dip, tip)

    return (ang_pip >= angle_thresh) and (ang_dip >= angle_thresh)

def is_thumb_extended(lm, angle_thresh=155):
    """
    โป้ง: ใช้มุม MCP-IP-TIP และ/หรือ MCP-CMC-IP
    ตรงๆพอ: ถ้ามุม MCP-IP-TIP >= thresh ถือว่าเหยียด
    """
    CMC, MCP, IP, TIP = FINGER_IDS["thumb"]
    mcp = (lm[MCP].x, lm[MCP].y)
    ip  = (lm[IP].x,  lm[IP].y)
    tip = (lm[TIP].x, lm[TIP].y)
    ang = angle_between(mcp, ip, tip)
    return ang >= angle_thresh

def count_extended_fingers(lm):
    ext = {}
    ext["thumb"]  = is_thumb_extended(lm)
    ext["index"]  = is_finger_extended(lm, FINGER_IDS["index"])
    ext["middle"] = is_finger_extended(lm, FINGER_IDS["middle"])
    ext["ring"]   = is_finger_extended(lm, FINGER_IDS["ring"])
    ext["pinky"]  = is_finger_extended(lm, FINGER_IDS["pinky"])
    return ext

def classify_rps(lm):
    ext = count_extended_fingers(lm)
    up = [k for k,v in ext.items() if v]
    n_up = len(up)

    # กฎเรียบง่าย แข็งแรงพอสำหรับ demo
    # Rock: ทุกนิ้วพับ (อนุโลมโป้งกางเล็กน้อยได้ถ้าอยาก)
    if (ext["index"]==False and ext["middle"]==False and
        ext["ring"]==False and ext["pinky"]==False and ext["thumb"]==False):
        return "Rock"

    # Paper: 4 นิ้ว index..pinky เหยียดหมด (โป้งจะกางหรือไม่กางก็ขอแค่ไม่พับติดแน่น)
    if (ext["index"] and ext["middle"] and ext["ring"] and ext["pinky"]):
        return "Paper"

    # Scissors: ชี้+กลาง เหยียด, นาง+ก้อย พับ (โป้งไม่บังคับ)
    if (ext["index"] and ext["middle"] and not ext["ring"] and not ext["pinky"]):
        return "Scissors"

    return "Unknown"

# ===== CONFIG =====
CAM_INDEX = 0
FRAME_W, FRAME_H = 1280, 720
MIN_DET_CONF = 0.5
MIN_TRK_CONF = 0.5
SHOW_FPS = True

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_styles  = mp.solutions.drawing_styles

def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)

    if not cap.isOpened():
        raise RuntimeError(f"เปิดกล้องไม่ได้ index={CAM_INDEX}")

    print("[INFO] Press 'q' to quit.")

    # MediaPipe Hands มีโมเดลติดมาแล้ว ไม่ต้องโหลดเพิ่ม
    with mp_hands.Hands(
        static_image_mode=False,         # True = โหมดภาพนิ่ง (ช้ากว่า)
        max_num_hands=2,
        model_complexity=1,              # 0 เร็วขึ้น, 1 คุณภาพดี, 2 ละเอียดกว่า
        min_detection_confidence=MIN_DET_CONF,
        min_tracking_confidence=MIN_TRK_CONF
    ) as hands:

        t_prev = time.time()
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            # แนะนำให้ flip สำหรับกล้องหน้า
            frame = cv2.flip(frame, 1)
            rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Inference
            result = hands.process(rgb)

            # วาดผลลัพธ์
            if result.multi_hand_landmarks:
                for hand_lms, handedness in zip(
                    result.multi_hand_landmarks,
                    result.multi_handedness
                ):
                    # วาดจุด + เส้นเชื่อม
                    mp_drawing.draw_landmarks(
                        frame,
                        hand_lms,
                        mp_hands.HAND_CONNECTIONS,
                        mp_styles.get_default_hand_landmarks_style(),
                        mp_styles.get_default_hand_connections_style(),
                    )

                    # แสดง Left/Right + score
                    label = handedness.classification[0].label   # 'Left' / 'Right'
                    score = handedness.classification[0].score
                    x0 = int(hand_lms.landmark[0].x * frame.shape[1])
                    y0 = int(hand_lms.landmark[0].y * frame.shape[0])
                    cv2.putText(frame, f"{label} {score:.2f}", (x0+8, y0-8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2, cv2.LINE_AA)

                    # ===== RPS Gesture Detection =====
                    gesture = classify_rps(hand_lms.landmark)
                    GESTURE_HISTORY.append(gesture)
                    gesture_smoothed = Counter(GESTURE_HISTORY).most_common(1)[0][0]
                    # วาดป้ายชื่อท่าบริเวณข้อมือ
                    cv2.putText(frame, f"{gesture_smoothed}", (x0+8, y0+24),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

            # FPS
            if SHOW_FPS:
                t_now = time.time()
                fps = 1.0 / max(t_now - t_prev, 1e-8)
                t_prev = t_now
                cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2, cv2.LINE_AA)

            cv2.imshow("MediaPipe Hands (no clone, built-in model)", frame)
            if (cv2.waitKey(1) & 0xFF) == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
