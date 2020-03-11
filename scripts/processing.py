import pandas as pd
import requests
from bs4 import BeautifulSoup
import os
import numpy as np
from copy import deepcopy
import pdpipe as pdp
import re
from tqdm import tqdm
from functools import lru_cache

from data.util.environment_variables import GEOSCHEME_CODES, COUNTRY_CODES, EXCEPTIONS, REDIRECTIONS, SUPERREGIONS, REGIONS, COUNTRIES
from data.scripts.project_data import DataLoader

punctuation = re.compile(r"[^\s\w]")
entity = re.compile(r"[\w\s]+")

def evaluate_geoscheme_name(name, geoscheme_df):
    """
    Evaluates the country, region and superregion breakdown of a string, given that it has been matched to the GEOSCHEME_CODES Hash Map
    """
    code = GEOSCHEME_CODES[name]

    inverse_geoscheme_codes = {v:i for i,v in GEOSCHEME_CODES.items()}

    grouping = {'superregion': None, 'region': None, 'country': None}

    neg_index = -1

    while (neg_index > -6) or (None in grouping.values()):
        if code not in geoscheme_df.iloc[:,neg_index].values:
            neg_index -= 1
            continue
        
        first_row = geoscheme_df.iloc[:,neg_index][code==geoscheme_df.iloc[:,neg_index].values].index[0]

        if neg_index == -1 or geoscheme_df.iloc[first_row,neg_index+1] is None:
            assert inverse_geoscheme_codes[code] in SUPERREGIONS
            grouping['superregion'] = inverse_geoscheme_codes[code]
        else:
            grouping['region'] = inverse_geoscheme_codes[code]
            superregion_index = neg_index
            while grouping['superregion'] not in SUPERREGIONS:
                superregion_index += 1
                grouping['superregion'] = inverse_geoscheme_codes[geoscheme_df.iloc[first_row,superregion_index]]
        break

    try:
        assert grouping['region'] in REGIONS + [None]
        assert grouping['superregion'] in SUPERREGIONS + [None]
    except AssertionError as e:
        print(e)
        print(grouping)
        raise

    return grouping

def evaluate_country_name(name,geoscheme_df):
    """
    Evaluates the country, region and superregion breakdown of a string, given that it has been matched to the COUNTRY_CODES Hash Map
    """
    code = COUNTRY_CODES[name]

    inverse_geoscheme_codes = {v:i for i,v in GEOSCHEME_CODES.items()}

    row_index = geoscheme_df[geoscheme_df['numeric'] == code].index

    grouping = {'superregion': None, 'region': None, 'country': name}

    row = geoscheme_df.iloc[row_index,:].values[0]
    neg_index = -1

    while (neg_index > -6) or (None in grouping.values()):
        if row[neg_index] is None:
            neg_index -= 1
            continue

        if neg_index != -1 and row[neg_index] and row[neg_index+1]:
            grouping['region'] = inverse_geoscheme_codes[row[neg_index]]
            superregion_index = neg_index
            while grouping['superregion'] not in SUPERREGIONS:
                superregion_index += 1
                grouping['superregion'] = inverse_geoscheme_codes[geoscheme_df.iloc[row,superregion_index]]
            break
        
        if (row[neg_index]) and (row[neg_index+1] is None):
            grouping['superregion'] = inverse_geoscheme_codes[row[neg_index]]
            neg_index -= 1
            continue

        break
    
    try:
        assert grouping['region'] in REGIONS + [None]
        assert grouping['superregion'] in SUPERREGIONS + [None]
    except AssertionError as e:
        print(e)
        print(grouping)
        raise

    return grouping

def get_country_region_superregion(geoscheme_df,name):
    """
    Returns the country, region, superregion hash map for a given string
    """
    lower_name = name.lower()

    if lower_name in GEOSCHEME_CODES:
        return evaluate_geoscheme_name(lower_name,geoscheme_df)
        
    if lower_name in COUNTRY_CODES:
        return evaluate_country_name(lower_name,geoscheme_df)

    if lower_name in EXCEPTIONS:
        return EXCEPTIONS[lower_name]

    if lower_name in REDIRECTIONS:
        return get_country_region_superregion(geoscheme_df,REDIRECTIONS[lower_name])
    
    if punctuation.findall(lower_name):
        list_ = []
        for region in entity.findall(lower_name):
            region = region.strip()
            list_.append(get_country_region_superregion(geoscheme_df,region))
        combined_dict = {
            'country':[],
            'region': [],
            'superregion': []
        } 
        for dict_ in list_:
            for key,value in dict_.items():
                if value and value not in combined_dict[key]:
                    combined_dict[key].append(value)
        for key,value in combined_dict.copy().items():
            if type(value) == list and len(value) == 1:
                combined_dict[key] = value[0]

        return combined_dict

    return {
        'country': lower_name,
        'region': 'unknown',
        'superregion': 'unknown'
    }

def load_geoscheme_df():
    """
    Loads the UN Geoscheme DataFrame mapping countries to regions and superregions
    """
    geoscheme_table = get_geoscheme_table()

    geoscheme_df = pd.DataFrame(geoscheme_table)
    
    geoscheme_df = pd.concat([
        geoscheme_df.loc[:,['country/region','numeric']],
        geoscheme_df['m49'].str.split('<',expand=True)
    ],axis=1)

    conversion_pipe = make_conversion_pipe()

    return conversion_pipe.apply(geoscheme_df)

def get_geoscheme_table():
    """
    Downloads the UN Geoscheme Table from Wikipedia and scrapes the table
    """
    GEOSCHEME_WIKI_URL = 'https://en.wikipedia.org/wiki/List_of_countries_by_United_Nations_geoscheme'
    
    response = requests.get(GEOSCHEME_WIKI_URL)
    soup = BeautifulSoup(response.text,features="html.parser")

    rows = []
    headers = ['country/region','capital','alpha-2','alpha-3','numeric','m49']

    for tr in soup.find_all('tr')[2:]:
        tds = tr.find_all('td')
        rows.append({
            header: entry.text for header, entry in zip(headers,tds)
        })

    return rows

def clean_out_world(x):
    """
    Removes the instances of the World m49 Code '001' from the UN Geoscheme DataFrame
    """
    if x is None:
        return x
    if type(x) == str:
        if x.replace(' ','') == '001':
            return None
        return x.replace(' ','')

def replace_new_lines(x):
    """
    Replaces instances of the \n escape character 
    """
    if type(x) != str:
        return x
    return x.replace('\n','')

def make_conversion_pipe():
    """
    Creates the Pandas Pipeline for the transformation of the UN Geoscheme DataFrame
    """
    pipeline = pdp.ColRename({i:str(i) for i in range(0,5)})
    pipeline += pdp.ApplyByCols(['country/region','numeric','0','1','2','3','4'],func=replace_new_lines)
    pipeline += pdp.ApplyByCols(['numeric','0','1','2','3','4'],func=clean_out_world)
    pipeline += pdp.DropNa(axis=1,how='all')
    return pipeline


def get_country_to_dict_mapping(api_data=None):
    if not api_data:
        data_loader = DataLoader()
        api_data = data_loader.load_api_data()

    unique_countries = api_data['country'].unique()

    geoscheme_df = load_geoscheme_df()

    country_to_dict_mapping = {i: get_country_region_superregion(geoscheme_df, i) for i in unique_countries}


    return country_to_dict_mapping

def encode_country_column(country_column: pd.Series):
    country_column = country_column.copy()
    country_one_hot = pd.get_dummies(pd.DataFrame(country_column.to_dict()).T.applymap(lambda x: str(x) if type(x) == list else x))

    mistakes = list(filter(lambda x: "_[" in x if x else x,country_one_hot.columns))

    def correct_mistaken_one_hot_encodings(df,mistakes):
        correct_names = set(COUNTRIES + REGIONS + SUPERREGIONS)
        df = df.copy()
        entity = re.compile(r"\'(.+?)\'")
        for mistake in tqdm(mistakes):
            mistaken_indices = df[df[mistake]==1].index
            category, mistaken_regions = mistake.split('_')
            regions = entity.findall(mistaken_regions)
            for region in regions:
                if region not in correct_names:
                    region = entity.findall(region)
                sub_column = ''.join([category,'_',region])
                df.loc[mistaken_indices,sub_column] = 1
        df.fillna(int(0),inplace=True)
        df.drop(mistakes,axis=1,inplace=True)
        return df

    
    country_one_hot = correct_mistaken_one_hot_encodings(country_one_hot,mistakes)

    #Dropping one entry to avoid full dummy variable coverage
    country_one_hot.drop(['superregion_unknown','region_unknown','country_unknown'],inplace=True,axis=1)

    #Converting all indicator feature values to integers
    country_one_hot.astype(np.int32)
    
    return country_one_hot

