import os
import cv2
import mediapipe as mp
import csv

DATASET_PATH = "asl_dataset"   # folder path
OUTPUT_CSV = "asl_data.csv"

mp_hands = mp.solutions.hands

def extract_features(hand_landmarks):
    xs, ys, zs = [], [], []
    for lm in hand_landmarks.landmark:
        xs.append(lm.x)
        ys.append(lm.y)
        zs.append(lm.z)

    wrist_x, wrist_y, wrist_z = xs[0], ys[0], zs[0]

    rel_x = [x - wrist_x for x in xs]
    rel_y = [y - wrist_y for y in ys]
    rel_z = [z - wrist_z for z in zs]

    max_val = max(
        max(abs(v) for v in rel_x),
        max(abs(v) for v in rel_y),
        max(abs(v) for v in rel_z),
        1e-6
    )

    features = []
    for i in range(21):
        features.extend([
            rel_x[i] / max_val,
            rel_y[i] / max_val,
            rel_z[i] / max_val
        ])

    return features

def main():
    hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1)

    with open(OUTPUT_CSV, mode="w", newline="") as f:
        writer = csv.writer(f)

        # header
        header = []
        for i in range(21):
            header += [f"x{i}", f"y{i}", f"z{i}"]
        header.append("label")
        writer.writerow(header)

        for label in os.listdir(DATASET_PATH):
            folder_path = os.path.join(DATASET_PATH, label)

            if not os.path.isdir(folder_path):
                continue

            print("Processing:", label)

            for img_name in os.listdir(folder_path):
                img_path = os.path.join(folder_path, img_name)

                img = cv2.imread(img_path)
                if img is None:
                    continue

                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                result = hands.process(img_rgb)

                if result.multi_hand_landmarks:
                    for hand_landmarks in result.multi_hand_landmarks:
                        features = extract_features(hand_landmarks)
                        features.append(label)

                        writer.writerow(features)

    print("Dataset converted to CSV!")

if __name__ == "__main__":
    main()