import pandas as pd
from sklearn.preprocessing import LabelEncoder


def clean_columns(df):
    df.columns = df.columns.str.strip() \
                           .str.replace(" ", "") \
                           .str.replace("_", "") \
                           .str.lower()
    return df


def load_data():
    city = clean_columns(pd.read_excel("data/City.xlsx"))
    continent = clean_columns(pd.read_excel("data/Continent.xlsx"))
    country = clean_columns(pd.read_excel("data/Country.xlsx"))
    item = clean_columns(pd.read_excel("data/Item.xlsx"))
    mode = clean_columns(pd.read_excel("data/Mode.xlsx"))
    region = clean_columns(pd.read_excel("data/Region.xlsx"))
    transaction = clean_columns(pd.read_excel("data/Transaction.xlsx"))
    attraction_type = clean_columns(pd.read_excel("data/Type.xlsx"))
    user = clean_columns(pd.read_excel("data/User.xlsx"))

    return city, continent, country, item, mode, region, transaction, attraction_type, user


def merge_data():
    city, continent, country, item, mode, region, transaction, attraction_type, user = load_data()

    # Merge names only (avoid duplicate hierarchy confusion)
    user = user.merge(continent[["continentid", "continent"]], 
                      on="continentid", how="left")

    user = user.merge(region[["regionid", "region"]], 
                      on="regionid", how="left")

    user = user.merge(country[["countryid", "country"]], 
                      on="countryid", how="left")

    user = user.merge(city[["cityid", "cityname"]], 
                      on="cityid", how="left")

    # Merge transaction with user
    data = transaction.merge(user, on="userid", how="left")

    # Merge attraction type
    item = item.merge(attraction_type, on="attractiontypeid", how="left")

    # Merge attraction details
    data = data.merge(item, on="attractionid", how="left")

    data = data.dropna(subset=["rating"])

    return data

def encode_features(data):
    le_dict = {}

    categorical_cols = [
        "continent",
        "region",
        "country",
        "cityname",
        "attractiontype",
        "visitmode"
    ]

    for col in categorical_cols:
        if col in data.columns:
            le = LabelEncoder()
            data[col] = le.fit_transform(data[col].astype(str))
            le_dict[col] = le

    return data, le_dict