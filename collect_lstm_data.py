import cv2
import mediapipe as mp
import numpy as np
import os

# ===== CONFIG =====
GESTURES = ["HI", "THANKS", "BYE",
 "I", "YOU", "MY",
 "IS", "AM", "ARE", "HAVE", "WANT", "GO", "COME", "MAKE",
 "NAME", "DAY", "FRIEND", "COLLEGE", "HOME",
 "GOOD", "BAD", "HAPPY", "SAD", "OK",
 "WHAT", "HOW", "WHERE", "WHY",
 "AND", "NOT",
 "STOP", "WAIT", "SHUBHAM"]   # same as static
SEQUENCES = 40
SEQUENCE_LENGTH = 30
SAVE_PATH = "lstm_dataset"
# ==================

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

for gesture in GESTURES:
    os.makedirs(os.path.join(SAVE_PATH, gesture), exist_ok=True)

    print(f"\n👉 Perform gesture: {gesture}")

    for seq in range(SEQUENCES):

        input("Press ENTER to start sequence...")

        sequence = []

        while len(sequence) < SEQUENCE_LENGTH:
            ret, frame = cap.read()
            frame = cv2.flip(frame, 1)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb)

            if result.multi_hand_landmarks:
                for hand_landmarks in result.multi_hand_landmarks:
                    mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    features = extract_features(hand_landmarks)
                    sequence.append(features)

            cv2.putText(frame, f"{gesture} Seq:{seq}", (10,40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0),2)

            cv2.imshow("Collect LSTM Data", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        np.save(os.path.join(SAVE_PATH, gesture, f"{seq}.npy"), sequence)

cap.release()
cv2.destroyAllWindows()