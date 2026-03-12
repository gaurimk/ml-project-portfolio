import joblib
import pandas as pd

# Load model
model = joblib.load("models/house_price_model.pkl")

# Create sample input with correct feature names
sample = pd.DataFrame({
    "GrLivArea":[2000],
    "BedroomAbvGr":[3],
    "FullBath":[2]
})

prediction = model.predict(sample)

print("Predicted Price:", prediction[0])