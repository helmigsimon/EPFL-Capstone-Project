from data.util.db import ExtractedDataClient, HighLevelFeatureClient, APIDataClient
import pandas as pd
from tqdm import tqdm
import numpy as np
import pickle
import os  
from data.util.paths import DATA_PATH

class DataLoader:
    def __init__(self):
        self.api_data_client = APIDataClient()
        self.high_level_feature_client = HighLevelFeatureClient()
        self.extracted_data_client = ExtractedDataClient()

    def _load_data_as_df(self,client,**kwargs):
        return pd.read_sql(client.name,client.engine,**kwargs)

    def load_api_data(self,client=None):
        return self._load_data_as_df(self.api_data_client)
    
    def load_extracted_data(self,):
        return self._load_data_as_df(self.extracted_data_client)

    def load_high_level_features(self,):
        high_level_feature_df = self._load_data_as_df(self.high_level_feature_client,chunksize=10000)

        

        return high_level_feature_df

