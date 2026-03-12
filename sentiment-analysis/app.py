import streamlit as st
import joblib
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# Load model and vectorizer
import os
import joblib

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(BASE_DIR, "models", "sentiment_model.pkl")
vectorizer_path = os.path.join(BASE_DIR, "models", "vectorizer.pkl")

model = joblib.load(model_path)
vectorizer = joblib.load(vectorizer_path)

st.title("🎬 Movie Review Sentiment Analysis")

st.write("Enter a movie review to predict whether the sentiment is positive or negative.")

review = st.text_area("Enter movie review")

if st.button("Predict Sentiment"):

    review_vec = vectorizer.transform([review])

    prediction = model.predict(review_vec)

    # For probability (if Logistic Regression)
    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(review_vec)[0]
        confidence = max(prob)

    if prediction[0] == "positive":
        st.success("😊 Positive Review")
    else:
        st.error("😡 Negative Review")

    # Confidence score
    if hasattr(model, "predict_proba"):
        st.write(f"Confidence Score: {confidence:.2f}")

        st.progress(float(confidence))

    # WordCloud
    st.subheader("Word Cloud of Review")

    wordcloud = WordCloud(
        width=800,
        height=400,
        background_color="black"
    ).generate(review)

    fig, ax = plt.subplots()
    ax.imshow(wordcloud)
    ax.axis("off")

    st.pyplot(fig)