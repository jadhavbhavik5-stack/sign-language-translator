import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import pickle

DATA_PATH = "custom_dataset"

X, y = [], []

for label in os.listdir(DATA_PATH):
    folder = os.path.join(DATA_PATH, label)

    for file in os.listdir(folder):
        data = np.load(os.path.join(folder, file))
        X.append(data)
        y.append(label)

X = np.array(X)

le = LabelEncoder()
y_encoded = le.fit_transform(y)

model = RandomForestClassifier(n_estimators=200)
model.fit(X, y_encoded)

# Save model + labels
with open("custom_model.pkl", "wb") as f:
    pickle.dump((model, le), f)

print("✅ Model trained and saved!")