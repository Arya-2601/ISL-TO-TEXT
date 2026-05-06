# -*- coding: utf-8 -*-
import cv2
import mediapipe as mp
import pickle
import numpy as np
from collections import Counter

# ── angle helpers (same as day3) ─────────────────────────────────────────────

def angle_between(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba = a - b
    bc = c - b
    cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))


def get_angles(landmarks):
    lm = [(l.x, l.y, l.z) for l in landmarks]

    triplets = [
        # Thumb
        (0,  1,  2), (1,  2,  3), (2,  3,  4),
        # Index
        (0,  5,  6), (5,  6,  7), (6,  7,  8),
        # Middle
        (0,  9, 10), (9, 10, 11), (10, 11, 12),
        # Ring
        (0, 13, 14), (13, 14, 15), (14, 15, 16),
        # Pinky
        (0, 17, 18), (17, 18, 19), (18, 19, 20),
        # Knuckle spread
        (5,  0,  9), (9,  0, 13), (13, 0, 17), (5,  0, 17),
    ]

    return [angle_between(lm[a], lm[b], lm[c]) for a, b, c in triplets]  # 19 values

# ── load model ────────────────────────────────────────────────────────────────

model_path = r"C:\Users\user\OneDrive\Desktop\projects frontend\ipd\ISL-TO-TEXT\isl_angles_model.pkl"
with open(model_path, 'rb') as f:
    model = pickle.load(f)

# ── mediapipe setup ───────────────────────────────────────────────────────────

mp_hands = mp.solutions.hands
mp_draw  = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.8,
    min_tracking_confidence=0.8
)

# ── webcam ────────────────────────────────────────────────────────────────────

cap = cv2.VideoCapture(0)
print("Webcam started! Press Q to quit.", flush=True)

prediction_buffer = []
BUFFER_SIZE  = 15
stable_letter = ""

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame    = cv2.flip(frame, 1)
    h, w, _  = frame.shape
    rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result   = hands.process(rgb)

    if result.multi_hand_landmarks:

        # Draw skeleton on all detected hands
        for handLms in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)

        # Predict using first (dominant) hand only
        handLms = result.multi_hand_landmarks[0]
        angles  = get_angles(handLms.landmark)          # ← 19 angles, not 63 keypoints
        prediction = model.predict([angles])[0]
        prediction_buffer.append(prediction)

    # Keep buffer size fixed
    if len(prediction_buffer) > BUFFER_SIZE:
        prediction_buffer.pop(0)

    # Show letter only if same prediction appears 10+ times in last 15 frames
    if len(prediction_buffer) == BUFFER_SIZE:
        most_common, count = Counter(prediction_buffer).most_common(1)[0]
        if count >= 10:
            stable_letter = most_common

    # Display
    cv2.rectangle(frame, (0, 0), (w, 90), (0, 0, 0), -1)
    cv2.putText(frame, f'Sign: {stable_letter}',
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX,
                2, (0, 255, 0), 3)
    cv2.putText(frame, 'Press Q to quit',
                (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (255, 255, 255), 1)

    cv2.imshow('ISL to Text - Real Time', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Done!", flush=True)