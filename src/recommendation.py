import joblib
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from src.datacleaning import merge_data, encode_features


def train_recommender():
    data = merge_data()

    print("Data Loaded Successfully")
    print("Columns:", data.columns)

    # Create user-item matrix
    user_item = data.pivot_table(
        index="userid",
        columns="attractionid",
        values="rating"
    ).fillna(0)

    print("User-Item Matrix Shape:", user_item.shape)

    # Compute similarity between users
    similarity_matrix = cosine_similarity(user_item)

    print("Similarity Matrix Shape:", similarity_matrix.shape)

    # Save models
    joblib.dump(user_item, "models/user_item.pkl")
    joblib.dump(similarity_matrix, "models/similarity.pkl")

    print("Recommender model saved successfully!")


if __name__ == "__main__":
    train_recommender()