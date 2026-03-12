import pandas as pd

class DataPreprocessing:

    def preprocess(self, df):

        df.dropna(inplace=True)

        return df