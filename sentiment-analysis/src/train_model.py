import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from preprocess import clean_text

# Load dataset
df = pd.read_csv("data/IMDB Dataset.csv")

# Clean text
df["review"] = df["review"].apply(clean_text)

X = df["review"]
y = df["sentiment"]

# Vectorization
vectorizer = TfidfVectorizer(max_features=5000)

X_vec = vectorizer.fit_transform(X)

# Train test split
X_train, X_test, y_train, y_test = train_test_split(
    X_vec, y, test_size=0.2, random_state=42
)

# Model
model = LogisticRegression()

model.fit(X_train, y_train)

# Save model + vectorizer
joblib.dump(model, "models/sentiment_model.pkl")
joblib.dump(vectorizer, "models/vectorizer.pkl")

print("Model saved successfully")