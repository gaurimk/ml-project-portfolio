# ✈ Flight Price Prediction App

A Machine Learning web application that predicts **flight ticket prices** based on airline, source, destination, number of stops, and other flight details.

The model is trained using historical flight data and deployed using **Streamlit Cloud** to provide an interactive prediction interface.

---

## 🚀 Live Demo

🔗 **Try the App:**
https://ml-project-portfolio-f9rfesn3j8appppoydyfhsr.streamlit.app/

---

## 📊 Project Overview

This project demonstrates an **end-to-end Machine Learning workflow**:

1. Data preprocessing
2. Feature engineering
3. Model training
4. Model serialization
5. Web app development
6. Cloud deployment

Users can input flight details and instantly get an estimated flight ticket price.

---

## 🧠 Machine Learning Pipeline

The pipeline includes:

* Data Cleaning
* Feature Engineering
* One-Hot Encoding
* Model Training
* Model Serialization using `pickle`
* Prediction pipeline for inference

---

## 🛠 Tech Stack

**Programming Language**

* Python

**Libraries**

* Pandas
* NumPy
* Scikit-Learn
* XGBoost
* Streamlit

**Deployment**

* Streamlit Cloud

---

## 📂 Project Structure

```
ml-project-portfolio
│
├── flight-price-prediction
│   │
│   ├── app.py
│   ├── requirements.txt
│   │
│   ├── artifacts
│   │   ├── model.pkl
│   │   ├── features.pkl
│   │
│   └── src
│       └── predict_pipeline.py
│
└── README.md
```

---

## ⚙️ Installation (Run Locally)

Clone the repository:

```
git clone https://github.com/your-username/ml-project-portfolio.git
```

Move into the project directory:

```
cd flight-price-prediction
```

Install dependencies:

```
pip install -r requirements.txt
```

Run the Streamlit app:

```
streamlit run app.py
```

---

## 🖥 App Interface

The app allows users to:

* Select Airline
* Select Source City
* Select Destination
* Choose Number of Stops
* Predict Flight Price

The model then estimates the **ticket price in Indian Rupees (₹).**

---

## 📈 Future Improvements

* Add more airlines and routes
* Improve feature engineering
* Add model comparison
* Deploy multiple ML apps in a single dashboard
* Add data visualization

---

## 👩‍💻 Author

**Gauri Mahadev**

Machine Learning & Data Science Enthusiast

---

⭐ If you found this project useful, consider giving it a **star on GitHub**!
