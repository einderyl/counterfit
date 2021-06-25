import pickle
import pandas as pd
from counterfit.core.targets import TextTarget

class Xgb(TextTarget):
    data_paths = ['drive/MyDrive/May2Dec2020_Defaced.csv']
    model_name = "xgb"
    model_data_type = "text"
    model_endpoint = "drive/MyDrive/xgbmodel.pickle"
    feature_extractor_endpoint = "drive/MyDrive/tfidf.pickle"
    model_input_shape = (1,)
    model_output_classes = [0,1]
    X = []

    def __init__(self):
        self.X = self._get_data(self.data_paths)
        self.feature_extractor = self._load(self.feature_extractor_endpoint)
        self.model = self._load(self.model_endpoint) 

    @staticmethod
    def _load(pickle_path):
        with open(pickle_path, 'rb') as f:
            return pickle.load(f)

    @staticmethod
    def _get_data(links):
        dfs = [pd.read_csv(source_link, lineterminator='\n').dropna() for source_link in links]
        content = []

        for df in dfs:
            content += list(df['Content'])

        return content

    def __call__(self, x):
        features = self.feature_extractor.transform(x)
        probabilities = self.model.predict_proba(features)
        return probabilities
