# 🎬 Movie Recommendation System

A **Machine Learning powered Movie Recommendation System** that suggests similar movies based on the movie you select. The system uses **Content-Based Filtering with Cosine Similarity** to find movies with similar metadata and features.

The application is built using **Python, Scikit-learn, and Streamlit** and is deployed online for interactive use.

---

# 🚀 Live Application

👉 **Try the App Here**

https://ml-project-portfolio-dujh6rmynfal3mzss3rcwt.streamlit.app/

---

# 📌 Features

- 🎥 Select any movie from the dropdown
- 🤖 Get **Top 5 similar movie recommendations**
- 🖼️ Movie posters fetched using **TMDB API**
- ⚡ Fast recommendations using **Cosine Similarity**
- 🌐 Deployed using **Streamlit Cloud**

---

# 🧠 How It Works

1️⃣ Movie metadata is processed and converted into numerical vectors.

2️⃣ A **similarity matrix** is created using **Cosine Similarity**.

3️⃣ When a user selects a movie:
- The system finds the most similar movies
- Returns **Top 5 recommendations**

Content-based filtering works by recommending items that are similar to the item a user already likes, based on features like genres, tags, or descriptions.

---

# 🛠️ Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- Streamlit
- TMDB API
- Pickle

---

# 📂 Project Structure
movie-recommendation-system
│
├── data
│ ├── tmdb_5000_movies.csv
│ └── tmdb_5000_credits.csv
│
├── models
│ ├── movies.pkl
│ └── vectors.pkl
│
├── notebooks
│ └── movie_recommender.ipynb
│
├── app.py
├── requirements.txt
└── README.md

---

# ⚙️ Installation (Run Locally)

Clone the repository:

```bash
git clone https://github.com/gaurimk/ml-project-portfolio.git
📊 Example

Input Movie:

Iron Man

Recommended Movies:

The Avengers
Captain America
Thor
Doctor Strange
Guardians of the Galaxy
