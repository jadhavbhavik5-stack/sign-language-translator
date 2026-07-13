import numpy as np
import os
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical

DATA_PATH = "lstm_dataset"

X, y = [], []

labels = os.listdir(DATA_PATH)

for label in labels:
    folder = os.path.join(DATA_PATH, label)

    for file in os.listdir(folder):
        data = np.load(os.path.join(folder, file))
        X.append(data)
        y.append(label)

X = np.array(X)

le = LabelEncoder()
y = le.fit_transform(y)
y = to_categorical(y)

# 🔥 IMPROVED MODEL
model = Sequential()
model.add(LSTM(64, return_sequences=True, input_shape=(30,63)))
model.add(LSTM(64))
model.add(Dense(64, activation='relu'))
model.add(Dense(len(labels), activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(X, y, epochs=20, validation_split=0.2)

# Save BOTH model + labels
model.save("custom_lstm.h5")

import pickle
with open("lstm_labels.pkl", "wb") as f:
    pickle.dump(le, f)

print("✅ LSTM model trained!")