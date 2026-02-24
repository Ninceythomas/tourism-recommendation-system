import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
from src.datacleaning import merge_data, encode_features
import numpy as np


def train_regression():
    data = merge_data()
    data, le_dict = encode_features(data)

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

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    print("R2 Score:", r2_score(y_test, predictions))
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    print("RMSE:", rmse)

    joblib.dump(model, "models/regression.pkl")
    joblib.dump(le_dict, "models/label_encoders.pkl")


if __name__ == "__main__":
    train_regression()