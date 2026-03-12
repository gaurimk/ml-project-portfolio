import pickle
import pandas as pd
from src.feature_engineering import FeatureEngineering

class PredictPipeline:

    def __init__(self):

        import os
        import pickle

        model_path = os.path.join("flight-price-prediction", "artifacts", "model.pkl")
        self.model = pickle.load(open(model_path, "rb"))
        os.path.join("flight-price-prediction","artifacts","features.pkl")

        self.fe = FeatureEngineering()

    def predict(self, data):

        # apply same feature engineering
        data = self.fe.transform(data)

        # match training columns
        data = data.reindex(columns=self.features, fill_value=0)

        prediction = self.model.predict(data)

        return prediction