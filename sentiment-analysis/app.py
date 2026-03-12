import streamlit as st
import joblib
import os
from src.preprocess import clean_text

# -----------------------------
# Load model and vectorizer
# -----------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = joblib.load(os.path.join(BASE_DIR, "models", "sentiment_model.pkl"))
vectorizer = joblib.load(os.path.join(BASE_DIR, "models", "vectorizer.pkl"))

# -----------------------------
# Streamlit UI
# -----------------------------

st.set_page_config(page_title="Sentiment Analysis", page_icon="🎬")

st.title("🎬 Movie Review Sentiment Analysis")

st.write("Enter a movie review and the model will predict whether the sentiment is **positive or negative**.")

# -----------------------------
# Input box
# -----------------------------

review = st.text_area("Enter your movie review")

# -----------------------------
# Prediction
# -----------------------------

if st.button("Predict Sentiment"):

    if review.strip() == "":
        st.warning("Please enter a review first.")
    
    else:

        # Clean the input text
        review_clean = clean_text(review)

        # Convert text to vector
        review_vec = vectorizer.transform([review_clean])

        # Predict sentiment
        prediction = model.predict(review_vec)[0]

        # Prediction probability
        proba = model.predict_proba(review_vec)
        confidence = proba.max()

        # Display result
        if prediction == "positive":
            st.success(f"Sentiment: Positive 😊")
        else:
            st.error(f"Sentiment: Negative 😞")

        st.write(f"Confidence: {confidence:.2f}")