import cv2
import mediapipe as mp
import csv
import time

# ====== CLEAR & DISTINCT GESTURES ======
GESTURES = ["HELLO", "STOP", "YES", "NO"]
SAMPLES_PER_GESTURE = 200
OUTPUT_CSV = "signs_data.csv"
# =======================================

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

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

    norm_x = [v / max_val for v in rel_x]
    norm_y = [v / max_val for v in rel_y]
    norm_z = [v / max_val for v in rel_z]

    features = []
    for i in range(21):
        features.extend([norm_x[i], norm_y[i], norm_z[i]])
    return features

def main():
    cap = cv2.VideoCapture(0)
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    # CSV header
    with open(OUTPUT_CSV, mode="w", newline="") as f:
        writer = csv.writer(f)
        header = []
        for i in range(21):
            header += [f"x{i}", f"y{i}", f"z{i}"]
        header.append("label")
        writer.writerow(header)

    gesture_index = 0
    current_label = GESTURES[gesture_index]
    count_for_current = 0

    print("Gestures:", GESTURES)
    print("Start with:", current_label)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)
        h, w, c = frame.shape

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                if count_for_current < SAMPLES_PER_GESTURE:
                    row = extract_features(hand_landmarks)
                    row.append(current_label)

                    with open(OUTPUT_CSV, mode="a", newline="") as f:
                        writer = csv.writer(f)
                        writer.writerow(row)

                    count_for_current += 1

        cv2.putText(frame, f"Gesture: {current_label} ({count_for_current}/{SAMPLES_PER_GESTURE})",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(frame, "Press 'n' next, 'q' quit",
                    (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        cv2.imshow("Collect Sign Data", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('n'):
            gesture_index += 1
            if gesture_index >= len(GESTURES):
                print("All gestures collected!")
                break
            current_label = GESTURES[gesture_index]
            count_for_current = 0
            time.sleep(1)
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    hands.close()

if __name__ == "__main__":
    main()
