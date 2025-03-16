import os
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

# Ensure the "models" directory exists
os.makedirs("models", exist_ok=True)

#  Load the dataset
df = pd.read_csv("Fish.csv")

#  Extract Features and Labels
X = df.drop(columns=["Species"])  # Features (Fish Measurements)
y = df["Species"]  # Target (Fish Species)

#  Encode Labels (Convert Species Names to Numerical Values)
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

#  Split the Data (80% Train, 20% Test)
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

#  Train the Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

#  Save the Model and Label Encoder
joblib.dump(model, "models/fish_model.pkl")  # Save trained model
joblib.dump(encoder, "models/label_encoder.pkl")  # Save label encoder

print(" Model and Label Encoder saved successfully!")
