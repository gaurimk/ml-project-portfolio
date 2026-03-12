import os
import pickle

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

movies_path = os.path.join(BASE_DIR, "models", "movies.pkl")
vectors_path = os.path.join(BASE_DIR, "models", "vectors.pkl")

movies = pickle.load(open(movies_path, "rb"))
similarity = pickle.load(open(vectors_path, "rb"))

import streamlit as st
import pickle
import requests
from sklearn.metrics.pairwise import cosine_similarity

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="Movie Recommendation System",
    page_icon="🎬",
    layout="wide"
)

# -----------------------------
# LOAD MODELS (CACHED)
# -----------------------------
@st.cache_resource
def load_models():
    movies = pickle.load(open("models/movies.pkl", "rb"))
    vectors = pickle.load(open("models/vectors.pkl", "rb"))
    return movies, vectors

movies, vectors = load_models()

# -----------------------------
# TMDB API KEY
# -----------------------------
API_KEY = st.secrets["tmdb_api_key"]

# -----------------------------
# FETCH POSTER (CACHED)
# -----------------------------
@st.cache_data
def fetch_poster(movie_title):

    url = f"https://api.themoviedb.org/3/search/movie?api_key={API_KEY}&query={movie_title}"

    response = requests.get(url)
    data = response.json()

    if data.get("results"):

        poster_path = data["results"][0].get("poster_path")

        if poster_path:
            return "https://image.tmdb.org/t/p/w500/" + poster_path

    return "https://via.placeholder.com/500x750?text=No+Poster"


# -----------------------------
# RECOMMEND FUNCTION
# -----------------------------
def recommend(movie):

    index = movies[movies["title"] == movie].index[0]

    similarity_scores = cosine_similarity([vectors[index]], vectors)[0]

    movie_list = sorted(
        list(enumerate(similarity_scores)),
        reverse=True,
        key=lambda x: x[1]
    )[1:6]

    recommended_movies = []
    recommended_posters = []

    for i in movie_list:

        movie_title = movies.iloc[i[0]].title

        recommended_movies.append(movie_title)
        recommended_posters.append(fetch_poster(movie_title))

    return recommended_movies, recommended_posters


# -----------------------------
# UI
# -----------------------------
st.title("🎬 Movie Recommendation System")

st.write("Select a movie and get similar movie recommendations.")

movie_list = movies["title"].values

selected_movie = st.selectbox(
    "Select a movie",
    movie_list
)

col1, col2, col3 = st.columns([1,2,1])

with col2:
    recommend_btn = st.button("Recommend")


# -----------------------------
# SHOW RESULTS
# -----------------------------
if recommend_btn:

    st.divider()
    st.subheader("🎥 Recommended Movies")

    names, posters = recommend(selected_movie)

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.image(posters[0], use_container_width=True)
        st.caption(names[0])

    with col2:
        st.image(posters[1], use_container_width=True)
        st.caption(names[1])

    with col3:
        st.image(posters[2], use_container_width=True)
        st.caption(names[2])

    with col4:
        st.image(posters[3], use_container_width=True)
        st.caption(names[3])

    with col5:
        st.image(posters[4], use_container_width=True)
        st.caption(names[4])


# -----------------------------
# FOOTER
# -----------------------------
st.markdown("---")
st.caption("Built with Python, Scikit-learn, TMDB API and Streamlit")