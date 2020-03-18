import os
import re
import requests
import pandas as pd
from string import punctuation
import numpy as np
from nltk.corpus import stopwords
import nltk

from bs4 import BeautifulSoup
from tqdm import tqdm

from data.util.environment_variables import SUPERREGIONS, REGIONS, COUNTRIES
from data.scripts.project_data import DataLoader
from lib.util.processing import entity, get_country_region_superregion, get_geoscheme_table, make_conversion_pipe

paren_num = re.compile(r"[(\d+)]+")
spaces = re.compile(r"\s+")

ARTIST_SPLIT_PUNCTUATION = ['/', '|', '','&', ',','-','.','*']
ARTIST_REMOVAL_PUNCTUATION_SET = set(punctuation) - set(ARTIST_SPLIT_PUNCTUATION)
SPLITWORDS = set(['feat', 'and', 'with','featuring','presents','vs','starring'])
STOPWORDS = set(stopwords.words('english'))
STOPWORDS.update(set(["'s"]))

UNNECESSARY_ARTIST_WORDS = set(['special','guest','orchestra','quartet','ensemble','band','trio','quintet'])
ARTIST_REMOVAL_WORDS = STOPWORDS.union(UNNECESSARY_ARTIST_WORDS,SPLITWORDS)

UNNECESSARY_LABEL_WORDS = set([
    'com',
    'records',
    'recordings',
    'compositions',
    'ltd',
    'mastering',
    'studio',
    'studios',
    'productions',
    'music',
    'pub',
    'co',
    'soundtracks',
    'series',
    'ag',
    'ab',
    'inc',
    'ltda',
    'llc',
    'company',
    'com',
    'corp',
    'enterprises',
    'label',
    'recording',
    'communications',
    'produktion',
    'not',
    'on',
    'self',
    'released',
    'audio',
    'production',
    'distribution',
    'plc',
    'pressings',
    'collection',
    'entertainment'
])
LABEL_REMOVAL_WORDS = STOPWORDS.union(UNNECESSARY_LABEL_WORDS)

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
    unique_genres = get_genres(genre_column)
    df = pd.get_dummies(pd.DataFrame({genre: np.zeros(len(genre_column)) for genre in unique_genres}))

    def encode_styles(row):
        idx = row.name
        if idx % 30000 == 0:
            print(idx)
        for list_ in row:
            try:
                df[list_][idx] = 1
            except KeyError:
                df[[element.replace("'","") for element in list_]][idx] = 1

    pd.DataFrame(genre_column).progress_apply(encode_styles,axis=1)
    

    return df


def encode_style_column(style_column: pd.Series):
    style_column = style_column.copy()

    unique_styles = get_styles(style_column)

    df = pd.DataFrame({style: np.zeros(len(style_column)) for style in unique_styles})

    entity = re.compile(r"\'(.+?)\'")

    def encode_styles(row):
        idx = row.name
        if idx % 30000 == 0:
            print(idx)
        for element in row:
            try:
                df[element][idx] = 1
            except KeyError as e:
                print(row)
                print(element)
                print(idx)
                print(e)
                raise
    style_column = pd.DataFrame(style_column).progress_apply(encode_styles,axis=1)

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

def clean_label_column(label_entries):
    label_entry = label_entries[0]

    label_entry = label_entry.lower()

    label_entry = remove_words(label_entry,LABEL_REMOVAL_WORDS)

    label_entry = remove_paren_num(label_entry)

    label_entry = remove_punctuation(label_entry)
    
    label_entry = remove_excess_spaces(label_entry)

    return label_entry




def clean_artist_column(artist_entry):
    artist_entry = artist_entry.lower()

    artist_entry = remove_words(artist_entry,ARTIST_REMOVAL_WORDS)

    artist_entry = remove_paren_num(artist_entry)

    artist_entry = remove_punctuation(artist_entry)

    artist_entry = remove_excess_spaces(artist_entry)

    #artist_entry = remove_leading_comma_space(artist_entry)
    
    return artist_entry

def remove_punctuation(artist_entry):
    return ''.join(char for char in artist_entry if char not in punctuation)

def remove_words(entry,words):
    return ' '.join(word for word in nltk.word_tokenize(entry) if word not in words)

def remove_unnecessary_words(artist_entry):
    return artist_entry.replace('& orchestra','').replace('and orchestra','').replace('special guest','')

def remove_paren_num(artist_entry):
    return paren_num.sub('',artist_entry)

def remove_excess_spaces(artist_entry):
    return spaces.sub(' ',artist_entry)

def remove_leading_comma_space(artist_entry):
    return artist_entry.replace(' ,',',')


def splitting_artist_column():
    split_encoded_artist = encoded_artist.str.split(r"/|,|\*|vs| and |&|featuring|presents|feat|starring|with| -|\|| . |•",expand=True) 
    unique_artists = set()
    for column in split_encoded_artist.columns:
        unique_artists = unique_artists.union(set(split_encoded_artist[column].apply(lambda x: x.strip(' ') if x else x)))