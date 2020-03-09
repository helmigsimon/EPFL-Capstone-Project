from data.util.db import ExtractedDataClient, HighLevelFeatureClient, APIDataClient
import pandas as pd
from tqdm import tqdm
import numpy as np
import pickle
import os  
from data.util.paths import DATA_PATH

def load_data():
    extracted_data_client, high_level_feature_client, api_data_client = ExtractedDataClient(), HighLevelFeatureClient(), APIDataClient()
    
    extracted_data_df = load_extracted_data(extracted_data_client)

    extracted_release_ids = {id_:1 for id_ in extracted_data_df['release_id']}

    high_level_feature_df = load_high_level_features(high_level_feature_client, extracted_release_ids)

    api_data_df = load_api_data(api_data_client, extracted_release_ids)

    combined_df = pd.merge(extracted_data_df,api_data_df,how='inner',on='release_id')

    proposal_df = pd.merge(combined_df,high_level_feature_df,how='inner',on='release_id').drop_duplicates('release_id')

    proposal_df.drop(['have','want','thumb_url','release_url'],inplace=True,axis=1)

    return proposal_df

def load_api_data(api_data_client, extracted_release_ids):
    api_data_columns = [column.name for column in api_data_client.columns if column.name != 'id']
    api_data = api_data_client.get_entries()

    api_data_dict = {column: [] for column in api_data_columns}

    convert_list_to_string = lambda x: str(pickle.loads(x))[1:-1]
    for data in tqdm(api_data):
        if data['release_id'] in extracted_release_ids:
            for column in api_data_columns:
                if type(data[column]) == bytes:
                    api_data_dict[column].append(convert_list_to_string(data[column]))     
                    continue
                api_data_dict[column].append(data[column])            

    api_data_df = pd.DataFrame(api_data_dict).drop_duplicates('release_id')

    return api_data_df
    


def load_high_level_features(high_level_feature_client=None):
    if not high_level_feature_client:
        high_level_feature_client = HighLevelFeatureClient()
    high_level_feature_df = pd.read_sql('high_level_features',high_level_feature_client.engine)

    high_level_feature_df.reset_index(drop=True,inplace=True)
    high_level_feature_df = high_level_feature_df.astype({'release_id':np.int32,'bitmap':np.int32})

    return high_level_feature_df

def load_extracted_data(extracted_data_client):
    extracted_data_columns = [column.name for column in extracted_data_client.columns if column.name != 'id']
    extracted_data = extracted_data_client.get_entries()


    extracted_data_dict = {column: [] for column in extracted_data_columns}

    convert_list_to_string = lambda x: str(pickle.loads(x))[1:-1]
    for data in tqdm(extracted_data):
        for column in extracted_data_columns:
            if column == 'track_titles':
                extracted_data_dict[column].append(convert_list_to_string(data[column])) 
                continue
            extracted_data_dict[column].append(data[column])         

    return pd.DataFrame(extracted_data_dict) 


def load_standards():
    with open(os.path.join(DATA_PATH,'standards.pkl'),'rb') as f:
        standards = pickle.load(f)

    
    standards_ = {standard.lower(): 1 for standard in standards}

    return standards_