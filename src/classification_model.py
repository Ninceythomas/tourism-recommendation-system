import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from src.datacleaning import merge_data, encode_features


def train_classification():
    print("Loading and merging data...")
    data = merge_data()

    print("Encoding features...")
    data, le_dict = encode_features(data)

    print("Available columns:")
    print(data.columns)

    features = [
        "continent",
        "region",
        "country",
        "cityname",
        "visityear",
        "visitmonth",
        "attractiontype"
    ]

    X = data[features]
    y = data["visitmode"]

    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("Training RandomForest model...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    print("Accuracy:", accuracy_score(y_test, predictions))
    print(classification_report(y_test, predictions))

    joblib.dump(model, "models/classification.pkl")

    print("Classification model saved successfully!")


if __name__ == "__main__":
    train_classification()