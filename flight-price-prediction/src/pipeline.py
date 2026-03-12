from data_ingestion import DataIngestion
from data_preprocessing import DataPreprocessing
from feature_engineering import FeatureEngineering
from model_trainer import ModelTrainer

class TrainingPipeline:

    def start(self):

        ingestion = DataIngestion()
        df = ingestion.initiate_data_ingestion()

        preprocessing = DataPreprocessing()
        df = preprocessing.preprocess(df)

        feature = FeatureEngineering()
        df = feature.transform(df)

        trainer = ModelTrainer()
        trainer.train(df)


if __name__ == "__main__":

    pipeline = TrainingPipeline()
    pipeline.start()