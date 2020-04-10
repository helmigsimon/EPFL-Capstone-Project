import os
import re
import requests
import random
import pickle
import pandas as pd
from string import punctuation
import numpy as np
from nltk.corpus import stopwords
import nltk
from functools import lru_cache
from scipy.sparse import csr_matrix
import sparse_dot_topn.sparse_dot_topn as ct
from collections.abc import Iterable


from sklearn.feature_extraction.text import TfidfVectorizer
from bs4 import BeautifulSoup
from tqdm.notebook import tqdm

from data.util.environment_variables import SUPERREGIONS, REGIONS, COUNTRIES
from data.util.paths import DATA_PATH
from data.scripts.project_data import DataLoader
from lib.util.processing import entity, get_country_region_superregion, get_geoscheme_table, make_conversion_pipe

paren_num = re.compile(r"[(\d+)]+")
spaces = re.compile(r"\s+")

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

def make_market_value_col(median_col,market_price_col):
    median_col, market_price_col = median_col.copy(), market_price_col.copy()
    
    market_value_col = median_col
    
    market_value_null_idx = market_value_col[market_value_col.isnull()].index
    
    market_value_col[market_value_null_idx] = market_price_col[market_value_null_idx]
    
    return market_value_col


def get_unique_values_from_column_with_list_dtype(column):
    unique = set()

    for entry in tqdm(column):
        for element in entry:
            unique.add(element)

    return unique

def remove_non_ascii(string_):
    return string_.encode("ascii",errors="ignore").decode()

def remove_punctuation(string_):
    return string_.translate(str.maketrans('','',punctuation))

def remove_words(entry,words):
    return ' '.join(word for word in nltk.word_tokenize(entry) if word not in words)

def remove_unnecessary_words(string_):
    return string_.replace('& orchestra','').replace('and orchestra','').replace('special guest','')

def remove_paren_num(string_):
    return paren_num.sub('',string_)

def remove_plural(string_):
    try:
        if string_[-1] == 's':
            string_ = string_[:-1]
        return string_
    except IndexError:
        return string_

def remove_excess_spaces(string_):
    return spaces.sub(' ',string_).strip(' ')

def save_to_pkl(obj, name, path=DATA_PATH):
    file_name = '{}.pkl'.format(name) 
    with open(os.path.join(path,file_name),'wb') as f:
        pickle.dump(obj,f)

def load_from_pkl(name,path=DATA_PATH):
    file_name = '{}.pkl'.format(name) 
    with open(os.path.join(path,file_name),'rb') as f:
        df = pickle.load(f)
    return df

def make_year_range_dict(columns,df):
    return {
        column: range(df[df[column]==1]['year'].min(),df[df[column]==1]['year'].max()+1) for column in columns
    }    
