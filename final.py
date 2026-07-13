from operator import le

import cv2
import mediapipe as mp
import numpy as np
import pickle
from keras.models import load_model
from collections import deque, Counter
import pyttsx3

# ===== CONFIG =====
STATIC_MODEL_FILE = "custom_model.pkl"
LSTM_MODEL_FILE = "custom_lstm.h5"

STATIC_MODE = 0
LSTM_MODE = 1

current_mode = STATIC_MODE


SEQUENCE_LENGTH = 30
CONF_THRESHOLD = 0.7
# ==================

# Load models
with open(STATIC_MODEL_FILE, "rb") as f:
    static_model, le = pickle.load(f)

lstm_model = load_model(LSTM_MODEL_FILE)

import pickle
with open("lstm_labels.pkl", "rb") as f:
    lstm_le = pickle.load(f)

# MediaPipe
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(max_num_hands=1)

# TTS
engine = pyttsx3.init()
engine.setProperty('rate', 150)

sequence = deque(maxlen=SEQUENCE_LENGTH)
prediction_buffer = deque(maxlen=10)

current_sentence = ""
last_spoken = ""

last_valid_word = ""
no_hand_counter = 0
frame_count = 0

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
            rel_x[i]/max_val,
            rel_y[i]/max_val,
            rel_z[i]/max_val
        ])
    return features

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)

    # Better size
    frame = cv2.resize(frame, (640, 480))

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    shown_label = last_valid_word if last_valid_word != "" else "Detecting..."

    confidence = 0.0

    # ✅ SAFE BLOCK
    if result.multi_hand_landmarks:

        no_hand_counter = 0

        for hand_landmarks in result.multi_hand_landmarks:

            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            features = extract_features(hand_landmarks)

            # ===== STATIC MODEL =====
            if current_mode == STATIC_MODE:

                row = np.array(features).reshape(1, -1)
                
                pred = static_model.predict(row)[0]
                probs = static_model.predict_proba(row)[0]
                label = le.inverse_transform([pred])[0]
                confidence = np.max(probs)

                if confidence > 0.5:   # 🔥 tune this (0.7–0.85)
                    prediction_buffer.append(label)
                else:
                    prediction_buffer.append(label)   # fallback (important)

                if len(prediction_buffer) >= 7:
                    shown_label = Counter(prediction_buffer).most_common(1)[0][0]
                    last_valid_word = shown_label

            # ===== LSTM MODEL =====
            else:
                sequence.append(features)
                frame_count += 1

                if len(sequence) == SEQUENCE_LENGTH and frame_count % 2 == 0:

                    pred = lstm_model(np.array([sequence]), training=False).numpy()[0]

                    confidence = np.max(pred)
                    index = np.argmax(pred)

                    label = lstm_le.inverse_transform([index])[0]


                    if confidence > CONF_THRESHOLD:
                        prediction_buffer.append(label)

                    if label != "NONE":
                        prediction_buffer.append(label)

                        if len(prediction_buffer) >= 7:
                            shown_label = Counter(prediction_buffer).most_common(1)[0][0]
                            last_valid_word = shown_label
                    
                    print("Pred:", label, "Confidence:", confidence)

    else:
        # No hand handling (NO flicker)
        no_hand_counter += 1

        if no_hand_counter > 10:
            shown_label = "No hand detected"

            

    # ===== UI =====
    cv2.putText(frame, f"Detected: {shown_label}", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    
    cv2.putText(frame, f"Confidence: {round(confidence*100,1)}%", 
            (10, 160),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)

    cv2.putText(frame, f"Sentence: {current_sentence}", (10, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

    mode_text = "STATIC" if current_mode == STATIC_MODE else "LSTM"
    cv2.putText(frame, f"Mode: {mode_text}", (10, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,200,0), 2)

    h, w, _ = frame.shape
    cv2.putText(frame, "a:Add  c:Clear  s:Speak  x:Switch Mode  q:Quit",
                (10, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)

    cv2.imshow("Sign Language Translator", frame)

    key = cv2.waitKey(20) & 0xFF

    if key == ord('q'):
        break

    elif key == ord('x'):
        current_mode = LSTM_MODE if current_mode == STATIC_MODE else STATIC_MODE
        print("Mode:", "STATIC" if current_mode == STATIC_MODE else "LSTM")
        prediction_buffer.clear()
        sequence.clear()

    elif key == ord('a'):
        if shown_label not in ["No hand detected", ""]:
            current_sentence += shown_label + " "

    elif key == ord('c'):
        current_sentence = ""

    elif key == ord('s'):
        if current_sentence.strip() != "":
            engine = pyttsx3.init()   # 🔥 recreate engine
            engine.setProperty('rate', 150)

            engine.say(current_sentence)
            engine.runAndWait()

cap.release()
cv2.destroyAllWindows()