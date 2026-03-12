import streamlit as st
import pandas as pd
import requests
from sklearn.feature_extraction.text import CountVectorizer
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
# API KEY
# -----------------------------
API_KEY = st.secrets["tmdb_api_key"]

# -----------------------------
# LOAD DATA
# -----------------------------
@st.cache_data
def load_data():
    movies = pd.read_csv("data/tmdb_5000_movies.csv")
    credits = pd.read_csv("data/tmdb_5000_credits.csv")

    movies = movies.merge(credits, on="title")

    movies = movies[['movie_id','title','overview','genres','keywords','cast','crew']]
    movies.dropna(inplace=True)

    # simple tag creation
    movies['tags'] = movies['overview']

    return movies

movies = load_data()

# -----------------------------
# VECTORIZE + SIMILARITY
# -----------------------------
@st.cache_resource
def create_similarity(movies):

    cv = CountVectorizer(max_features=5000, stop_words='english')
    vectors = cv.fit_transform(movies['tags']).toarray()

    similarity = cosine_similarity(vectors)

    return similarity

similarity = create_similarity(movies)

# -----------------------------
# FETCH POSTER
# -----------------------------
def fetch_poster(movie_title):

    search_url = f"https://api.themoviedb.org/3/search/movie?api_key={API_KEY}&query={movie_title}"

    data = requests.get(search_url)
    data = data.json()

    if len(data["results"]) > 0:

        poster_path = data["results"][0]["poster_path"]

        if poster_path:
            return "https://image.tmdb.org/t/p/w500/" + poster_path

    return "https://via.placeholder.com/500x750?text=No+Poster"


# -----------------------------
# RECOMMEND FUNCTION
# -----------------------------
def recommend(movie):

    index = movies[movies['title'] == movie].index[0]

    distances = similarity[index]

    movies_list = sorted(
        list(enumerate(distances)),
        reverse=True,
        key=lambda x: x[1]
    )[1:6]

    recommended_movies = []
    recommended_posters = []

    for i in movies_list:

        movie_title = movies.iloc[i[0]]["title"]

        recommended_movies.append(movie_title)
        recommended_posters.append(fetch_poster(movie_title))

    return recommended_movies, recommended_posters


# -----------------------------
# UI
# -----------------------------
st.title("🎬 Movie Recommendation System")

st.write("Select a movie and get similar movie recommendations.")

movie_list = movies['title'].values

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
        st.image(posters[0])
        st.caption(names[0])

    with col2:
        st.image(posters[1])
        st.caption(names[1])

    with col3:
        st.image(posters[2])
        st.caption(names[2])

    with col4:
        st.image(posters[3])
        st.caption(names[3])

    with col5:
        st.image(posters[4])
        st.caption(names[4])


# -----------------------------
# FOOTER
# -----------------------------
st.markdown("---")
st.caption("Built with Python, Scikit-learn, TMDB API and Streamlit")