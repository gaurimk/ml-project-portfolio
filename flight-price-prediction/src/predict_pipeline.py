import os
import pickle
import pandas as pd


class PredictPipeline:

    def __init__(self):

        model_path = os.path.join("flight-price-prediction","artifacts","model.pkl")
        features_path = os.path.join("flight-price-prediction","artifacts","features.pkl")

        self.model = pickle.load(open(model_path,"rb"))
        self.features = pickle.load(open(features_path,"rb"))


    def predict(self,data:pd.DataFrame):

        data = pd.get_dummies(data)

        data = data.reindex(columns=self.features, fill_value=0)

        preds = self.model.predict(data)

        return preds