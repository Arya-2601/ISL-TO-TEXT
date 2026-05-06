# -*- coding: utf-8 -*-
import mediapipe as mp
import cv2
import pandas as pd
import numpy as np
import os

print("Starting...", flush=True)

# ── helpers ──────────────────────────────────────────────────────────────────

def angle_between(a, b, c):
    """Angle (degrees) at point B, given three landmarks A-B-C."""
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba = a - b
    bc = c - b
    cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))


def get_angles(landmarks):
    """
    Returns 19 joint angles from MediaPipe hand landmarks.

    Landmark index reference:
      Wrist: 0
      Thumb:  1-2-3-4
      Index:  5-6-7-8
      Middle: 9-10-11-12
      Ring:   13-14-15-16
      Pinky:  17-18-19-20
    """
    lm = [(l.x, l.y, l.z) for l in landmarks]

    # (A, B, C) → angle measured at joint B
    triplets = [
        # Thumb  (3 angles)
        (0,  1,  2),
        (1,  2,  3),
        (2,  3,  4),
        # Index  (3 angles)
        (0,  5,  6),
        (5,  6,  7),
        (6,  7,  8),
        # Middle (3 angles)
        (0,  9, 10),
        (9, 10, 11),
        (10, 11, 12),
        # Ring   (3 angles)
        (0, 13, 14),
        (13, 14, 15),
        (14, 15, 16),
        # Pinky  (3 angles)
        (0, 17, 18),
        (17, 18, 19),
        (18, 19, 20),
        # Knuckle spread between fingers (4 angles)
        (5,  0,  9),
        (9,  0, 13),
        (13, 0, 17),
        (5,  0, 17),
    ]

    angles = []
    for a, b, c in triplets:
        angles.append(angle_between(lm[a], lm[b], lm[c]))

    return angles  # 19 values


# ── mediapipe setup ───────────────────────────────────────────────────────────

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.3
)

# ── paths ─────────────────────────────────────────────────────────────────────

dataset_path = r"C:\Users\user\OneDrive\Desktop\projects frontend\ipd\ISL-TO-TEXT"
output_path  = r"C:\Users\user\OneDrive\Desktop\ISL_project\isl_angles.csv"

# ── extraction loop ───────────────────────────────────────────────────────────

data    = []
labels  = []
skipped = 0
processed = 0

for folder in sorted(os.listdir(dataset_path)):
    folder_path = os.path.join(dataset_path, folder)

    if not os.path.isdir(folder_path):
        continue
    if folder.lower() not in list('abcdefghijklmnopqrstuvwxyz'):
        continue

    folder_count = 0

    for img_file in os.listdir(folder_path):
        if not img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue

        img_path = os.path.join(folder_path, img_file)
        img = cv2.imread(img_path)
        if img is None:
            skipped += 1
            continue

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result  = hands.process(img_rgb)

        if result.multi_hand_landmarks:
            angles = get_angles(result.multi_hand_landmarks[0].landmark)
            data.append(angles)
            labels.append(folder.upper())
            processed += 1
            folder_count += 1
        else:
            skipped += 1

    print(f"{folder.upper()} done -> {folder_count} extracted", flush=True)

# ── save ──────────────────────────────────────────────────────────────────────

print("\nSaving CSV...", flush=True)

# Column names: angle_00 … angle_18
col_names = [f"angle_{i:02d}" for i in range(19)]

df = pd.DataFrame(data, columns=col_names)
df['label'] = labels

os.makedirs(os.path.dirname(output_path), exist_ok=True)
df.to_csv(output_path, index=False)

print(f"Done!", flush=True)
print(f"Processed : {processed}", flush=True)
print(f"Skipped   : {skipped}", flush=True)
print(f"CSV saved : {output_path}", flush=True)
print(f"Features  : {len(col_names)} angles per sample", flush=True)