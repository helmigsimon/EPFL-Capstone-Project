import os
import re
import requests
import pandas as pd
import numpy as np

from bs4 import BeautifulSoup
from tqdm import tqdm

from data.util.environment_variables import SUPERREGIONS, REGIONS, COUNTRIES
from data.scripts.project_data import DataLoader
from lib.util.processing import entity, get_country_region_superregion, get_geoscheme_table, make_conversion_pipe


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
    country_one_hot = country_one_hot.astype(np.int32)
    
    return country_one_hot

def encode_genre_column(genre_column: pd.Series):
    genre_column = genre_column.copy()
    df = pd.get_dummies(genre_column.apply(lambda x: str(x).strip('[]').replace("'","") if type(x) == list else x))

    unique_genres = get_genres(genre_column)

    for genre in unique_genres:
        df[genre] = 0

    entity = re.compile(r"\'(.+?)\'")

    agg_columns = set(df.columns)-unique_genres
    for genre in tqdm(agg_columns):
        genre_indices = df[df[genre]==1].index
        sub_genres = entity.findall(genre)
        for sub_genre in sub_genres:
            if sub_genre not in unique_genres:
                raise Exception
            df.loc[genre_indices,sub_genre] = 1
    
    df.fillna(int(0),inplace=True)
    df.drop(agg_columns,axis=1,inplace=True)

    return df

def encode_style_column(style_column: pd.Series):
    style_column = style_column.copy()
    df = pd.get_dummies(style_column.apply(lambda x: str(x).strip('[]').replace("'","") if type(x) == list else x))

    unique_styles = get_styles(style_column)

    for style in unique_styles:
        df[style] = 0

    entity = re.compile(r"\'(.+?)\'")

    # Can this be vectorized?

    agg_columns = set(df.columns)-unique_styles
    for style in tqdm(agg_columns):
        style_indices = df[df[style]==1].index
        sub_styles = entity.findall(style)
        for sub_style in sub_styles:
            if sub_style not in unique_styles:
                raise Exception
            df.loc[style_indices,sub_style] = 1
    df.fillna(0,inplace=True)
    df.drop(agg_columns,axis=1,inplace=True)

    df = df.astype(np.int32)

    return df

def get_genres(genre_column: pd.Series):
    genres = get_unique_values_from_column_with_list_dtype(genre_column)

    genres.remove("Children's")
    genres.add("Childrens")

    return genres

def get_unique_values_from_column_with_list_dtype(column):
    unique = set()

    for entry in tqdm(column):
        for element in entry:
            unique.add(element)

    return unique

def get_styles(style_column: pd.Series):
    return get_unique_values_from_column_with_list_dtype(style_column)

