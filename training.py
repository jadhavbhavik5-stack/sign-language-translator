import pandas as pd # type: ignore
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from collections import Counter
import pickle

CSV_FILE = "asl_data.csv"
MODEL_FILE = "sign_model.pkl"

def main():
    df = pd.read_csv(CSV_FILE)

    X = df.drop("label", axis=1).values
    y = df["label"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(
        n_estimators=300,
        random_state=42
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    with open(MODEL_FILE, "wb") as f:
        pickle.dump(model, f)

    print(f"Model saved as {MODEL_FILE}")

    print(Counter(y))

if __name__ == "__main__":
    main()
