import joblib
import numpy as np

model = joblib.load("models/churn_model.pkl")

sample = np.array([[1,0,1,34,0,1,1,0,1,70]])

prediction = model.predict(sample)

print("Prediction:", prediction)