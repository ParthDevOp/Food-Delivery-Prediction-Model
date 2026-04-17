import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from scipy import stats

MODEL_PATH = "models/model.pkl"


def load_and_preprocess_data(filepath):
    df = pd.read_csv(filepath)

    num_cols = ['Distance_km', 'Preparation_Time_min',
                'Courier_Experience_yrs', 'Delivery_Time_min']

    # Fill missing values
    df[num_cols] = df[num_cols].apply(lambda x: x.fillna(x.median()))
    df = df.dropna(subset=['Delivery_Time_min'])

    # Map traffic
    traffic_mapping = {'Low': 1, 'Medium': 2, 'High': 3}
    if 'Traffic_Level' in df.columns:
        df['Traffic_Level'] = df['Traffic_Level'].map(traffic_mapping)

    # Remove outliers
    df = df[(np.abs(stats.zscore(df[num_cols])) < 3).all(axis=1)]

    return df


def train_model(data_path):
    df = load_and_preprocess_data(data_path)

    df_model = df.drop(columns=['Order_ID'], errors='ignore')
    df_encoded = pd.get_dummies(df_model, drop_first=True)

    X = df_encoded.drop(columns=['Delivery_Time_min'])
    y = df_encoded['Delivery_Time_min']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=10,
        min_samples_split=10,
        min_samples_leaf=4,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    # Save model + columns
    joblib.dump((model, X.columns), MODEL_PATH)

    return model, X.columns


def load_model():
    return joblib.load(MODEL_PATH)


def predict(input_df):
    model, columns = load_model()

    input_encoded = pd.get_dummies(input_df, drop_first=True)

    # Align columns
    for col in columns:
        if col not in input_encoded.columns:
            input_encoded[col] = 0

    input_encoded = input_encoded[columns]

    prediction = model.predict(input_encoded)[0]
    return prediction