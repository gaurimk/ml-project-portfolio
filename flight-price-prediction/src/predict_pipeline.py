import pickle
import pandas as pd
from src.feature_engineering import FeatureEngineering

class PredictPipeline:

    def __init__(self):

        self.model = pickle.load(open("artifacts/model.pkl", "rb"))
        self.features = pickle.load(open("artifacts/features.pkl", "rb"))

        self.fe = FeatureEngineering()

    def predict(self, data):

        # apply same feature engineering
        data = self.fe.transform(data)

        # match training columns
        data = data.reindex(columns=self.features, fill_value=0)

        prediction = self.model.predict(data)

        return prediction