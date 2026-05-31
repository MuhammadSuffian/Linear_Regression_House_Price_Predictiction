import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


with open("linear_model.pkl", "rb") as f:
    model = pickle.load(f)


print("=== House Price Predictor ===\n")

area             = float(input("Area (sq ft)               : "))
bedrooms         = int(input("Bedrooms                   : "))
bathrooms        = int(input("Bathrooms                  : "))
stories          = int(input("Stories                    : "))
parking          = int(input("Parking spots              : "))

mainroad         = input("Main road? (yes/no)        : ").strip().lower()
guestroom        = input("Guest room? (yes/no)       : ").strip().lower()
basement         = input("Basement? (yes/no)         : ").strip().lower()
hotwaterheating  = input("Hot water heating? (yes/no): ").strip().lower()
airconditioning  = input("Air conditioning? (yes/no) : ").strip().lower()
prefarea         = input("Preferred area? (yes/no)   : ").strip().lower()
furnishingstatus = input("Furnishing (furnished/semi-furnished/unfurnished): ").strip().lower()

# Build input dataframe
user_input = pd.DataFrame([{
    'area': area, 'bedrooms': bedrooms, 'bathrooms': bathrooms,
    'stories': stories, 'parking': parking,
    'mainroad': mainroad, 'guestroom': guestroom, 'basement': basement,
    'hotwaterheating': hotwaterheating, 'airconditioning': airconditioning,
    'prefarea': prefarea, 'furnishingstatus': furnishingstatus
}])

# Encode
user_encoded = pd.get_dummies(user_input)

# Align with training columns ← X.columns since no train/test split
user_encoded = user_encoded.reindex(columns=X.columns, fill_value=0)

# Predict
predicted_price = model.predict(user_encoded)
print(f"\n🏠 Predicted House Price: {predicted_price[0]:,.2f}")