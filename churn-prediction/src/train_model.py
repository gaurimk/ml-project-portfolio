import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from data_preprocessing import preprocess_data

X_train, X_test, y_train, y_test = preprocess_data("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")

model = RandomForestClassifier()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))

joblib.dump(model, "models/churn_model.pkl")

print("Model saved")