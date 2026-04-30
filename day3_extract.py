# -*- coding: utf-8 -*-
import mediapipe as mp
import cv2
import pandas as pd
import os
import sys

print("Starting...", flush=True)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.3
)

dataset_path = r"C:\Users\user\OneDrive\Desktop\ISL_project"

data = []
labels = []
skipped = 0
processed = 0

for folder in sorted(os.listdir(dataset_path)):
    folder_path = os.path.join(dataset_path, folder)

    # skip files, only process a-z folders
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
        result = hands.process(img_rgb)

        if result.multi_hand_landmarks:
            lm_list = []
            for lm in result.multi_hand_landmarks[0].landmark:
                lm_list.extend([lm.x, lm.y, lm.z])
            data.append(lm_list)
            labels.append(folder.upper())
            processed += 1
            folder_count += 1
        else:
            skipped += 1

    print(f"{folder.upper()} done -> {folder_count} extracted", flush=True)

print(f"\nSaving CSV...", flush=True)

df = pd.DataFrame(data)
df['label'] = labels

output_path = r"C:\Users\user\OneDrive\Desktop\ISL_project\isl_keypoints.csv"
df.to_csv(output_path, index=False)

print(f"Done!", flush=True)
print(f"Processed : {processed}", flush=True)
print(f"Skipped   : {skipped}", flush=True)
print(f"CSV saved at: {output_path}", flush=True)