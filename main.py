import cv2
import mediapipe as mp
import numpy as np
import pickle
from collections import Counter
import pyttsx3 # type: ignore

MODEL_FILE = "sign_model.pkl"

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

# Load trained model
with open(MODEL_FILE, "rb") as f:
    model = pickle.load(f)

# Text-to-speech engine
engine = pyttsx3.init()

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


def speak(text):
    engine.say(text)
    engine.runAndWait()

def main():
    cap = cv2.VideoCapture(0)
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    current_sentence = ""
    last_spoken = ""
    prediction_buffer = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)
        h, w, c = frame.shape

        shown_label = "No hand detected"

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            features = extract_features(hand_landmarks)
            row = np.array(features).reshape(1, -1)
            pred = model.predict(row)[0]

            prediction_buffer.append(pred)
            if len(prediction_buffer) > 15:
                    prediction_buffer.pop(0)

            common_pred, count = Counter(prediction_buffer).most_common(1)[0]

            if count > 10:
                    shown_label = common_pred
            else:
                    shown_label = pred

        # Display detected gesture
        cv2.putText(frame, f"Detected: {shown_label}", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Display sentence
        cv2.putText(frame, f"Sentence: {current_sentence}", (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # Instructions
        cv2.putText(frame, "a:Add word  c:Clear  s:Speak  q:Quit",
                    (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        cv2.imshow("Sign Language Translator", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('a'):
            if prediction_buffer:
                common_pred, count = Counter(prediction_buffer).most_common(1)[0]
                current_sentence += common_pred + " "
        elif key == ord('c'):
            current_sentence = ""
        elif key == ord('s'):
            if current_sentence and current_sentence != last_spoken:
                speak(current_sentence)
                last_spoken = current_sentence

    cap.release()
    cv2.destroyAllWindows()
    hands.close()

if __name__ == "__main__":
    main()
