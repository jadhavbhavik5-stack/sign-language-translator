import cv2
import mediapipe as mp
import numpy as np
import os

DATA_PATH = "sequence_data"
GESTURES = ["HELLO", "THANKYOU", "YES", "NO"]
SEQUENCE_LENGTH = 20
SEQUENCES_PER_GESTURE = 30

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

    max_val = max(max(abs(v) for v in rel_x),
                  max(abs(v) for v in rel_y),
                  max(abs(v) for v in rel_z), 1e-6)

    features = []
    for i in range(21):
        features.extend([
            rel_x[i]/max_val,
            rel_y[i]/max_val,
            rel_z[i]/max_val
        ])
    return features

cap = cv2.VideoCapture(0)
hands = mp_hands.Hands(max_num_hands=1)

for label in GESTURES:
    os.makedirs(os.path.join(DATA_PATH, label), exist_ok=True)

    print(f"Collecting for {label}")

    for seq in range(SEQUENCES_PER_GESTURE):
        sequence = []

        while len(sequence) < SEQUENCE_LENGTH:
            ret, frame = cap.read()
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb)

            if result.multi_hand_landmarks:
                for hand_landmarks in result.multi_hand_landmarks:
                    features = extract_features(hand_landmarks)
                    sequence.append(features)

            cv2.putText(frame, f"{label} Seq:{seq}", (10,30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0),2)
            cv2.imshow("Collect", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        np.save(os.path.join(DATA_PATH, label, f"{seq}.npy"), sequence)

cap.release()
cv2.destroyAllWindows()