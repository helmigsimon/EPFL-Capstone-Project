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

    def _load_data_as_df(self,client):
        return pd.read_sql(client.name,client.engine)

    def load_api_data(self,client=None):
        if not client:
            return self._load_data_as_df(self.api_data_client)
        
        assert type(client) == type(self.api_data_client)
        return self._load_data_as_df(client)
    
    def load_extracted_data(self,client=None):
        if not client:
            return self._load_data_as_df(self.extracted_data_client)
        
        assert type(client) == type(self.extracted_data_client)
        return self._load_data_as_df(client)

    def load_high_level_features(self,client=None):
        if not client:
            high_level_feature_df = self._load_data_as_df(self.high_level_feature_client)
        else:
            assert type(client) == type(self.high_level_feature_client)
            high_level_feature_df = self._load_data_as_df(client)
        
        high_level_feature_df.reset_index(drop=True,inplace=True)
        high_level_feature_df = high_level_feature_df.astype({'release_id':np.int32,'bitmap':np.int32})

        return high_level_feature_df

