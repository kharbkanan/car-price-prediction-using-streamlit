# train_model.py

import pandas as pd
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# Load your dataset (replace with your actual CSV)
df = pd.read_csv("car_data.csv")

# Encode categorical 'fuel_type' into numeric (one-hot encoding)
df = pd.get_dummies(df, columns=["fuel_type"], drop_first=True)

# Define input features (update if needed)
X = df[["year", "engine", "max_power", "mileage", "seats", "fuel_type_Petrol"]]
y = df["price"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train the model
model = RandomForestRegressor()
model.fit(X_train, y_train)

# Save model
pickle.dump(model, open("car_model.pkl", "wb"))

print("✅ Model trained and saved successfully as car_model.pkl")
