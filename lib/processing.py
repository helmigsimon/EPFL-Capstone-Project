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

def encode_genre_column(genre_column: pd.Series):
    genre_column = genre_column.copy()
    unique_genres = get_genres(genre_column)
    df = pd.get_dummies(pd.DataFrame({genre: np.zeros(len(genre_column)) for genre in unique_genres}))

    def encode_styles(row):
        idx = row.name
        for list_ in row:
            try:
                df.loc[idx,list_] = 1
            except KeyError:
                df.loc[idx,[element.replace("'","") for element in list_]] = 1

    pd.DataFrame(genre_column).progress_apply(encode_styles,axis=1)
    

    df = df.astype(np.uint8)
    df.drop('Jazz',axis=1,inplace=True)
    df.rename(columns={column:'genre_{}'.format(column) for column in df.columns},inplace=True)
    return df


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


def encode_style_column(style_column: pd.Series):
    style_column = style_column.copy()

    unique_styles = get_styles(style_column)

    df = pd.DataFrame({style: np.zeros(len(style_column)) for style in unique_styles})

    def encode_styles(row):
        idx = row.name
        for element in row:
            try:
                df.loc[idx,element] = 1
            except KeyError as e:
                print(row)
                print(element)
                print(idx)
                print(e)
                raise
    style_column = pd.DataFrame(style_column).progress_apply(encode_styles,axis=1)

    df = df.astype(np.uint8)
    df.rename(columns={column:'style_{}'.format(column) for column in df.columns},inplace=True)
    
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

def clean_label_column(label_entry):
    label_entry = label_entry.lower()

    label_entry = remove_words(label_entry,LABEL_REMOVAL_WORDS)

    label_entry = remove_paren_num(label_entry)

    label_entry = remove_punctuation(label_entry)
    
    label_entry = remove_excess_spaces(label_entry)

    label_entry = remove_plural(label_entry)

    return label_entry




def clean_artist_column(artist_entry):
    artist_entry = artist_entry.lower()

    artist_entry = remove_words(artist_entry,ARTIST_REMOVAL_WORDS)

    artist_entry = remove_paren_num(artist_entry)

    artist_entry = remove_punctuation(artist_entry)

    artist_entry = remove_excess_spaces(artist_entry)
    
    return artist_entry

def clean_format_text(format_text_entry):
    if not format_text_entry:
        return str(format_text_entry)

    format_text_entry = format_text_entry.lower()

    format_text_entry = remove_punctuation(format_text_entry)

    format_text_entry = remove_excess_spaces(format_text_entry)

    return format_text_entry




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

def get_ngrams(string_, n=3):
    ngrams = zip(*[string_[i:] for i in range(n)])
    return [''.join(ngram) for ngram in ngrams]

def artist_ngrams(string_,n=3):
    string_ = clean_artist_column(string_)

    return get_ngrams(string_,n)

def label_ngrams(string_,n=3):
    string_ = clean_label_column(string_)

    return get_ngrams(string_,n)

def cossine_similarity(A,B, ntop, lower_bound=0):
    A = A.tocsr()
    B = B.tocsr()
    M, _ = A.shape
    _, N = B.shape
 
    idx_dtype = np.int32
 
    nnz_max = M*ntop
 
    indptr = np.zeros(M+1, dtype=idx_dtype)
    indices = np.zeros(nnz_max, dtype=idx_dtype)
    data = np.zeros(nnz_max, dtype=A.dtype)

    ct.sparse_dot_topn(
        M, N, np.asarray(A.indptr, dtype=idx_dtype),
        np.asarray(A.indices, dtype=idx_dtype),
        A.data,
        np.asarray(B.indptr, dtype=idx_dtype),
        np.asarray(B.indices, dtype=idx_dtype),
        B.data,
        ntop,
        lower_bound,
        indptr, indices, data)

    return csr_matrix((data,indices,indptr),shape=(M,N)) 

def get_matches_df(sparse_matrix, name_vector, top=100):
    non_zeros = sparse_matrix.nonzero()
    
    sparserows = non_zeros[0]
    sparsecols = non_zeros[1]
    
    if top:
        nr_matches = top
    else:
        nr_matches = sparsecols.size
    
    left_side = np.empty([nr_matches], dtype=object)
    right_side = np.empty([nr_matches], dtype=object)
    similarity = np.zeros(nr_matches)

    
    for index in range(0, nr_matches):
        try:
            left_side[index] = name_vector[sparserows[index]]
            right_side[index] = name_vector[sparsecols[index]]
            similarity[index] = sparse_matrix.data[index]
        except IndexError:
            print(index)
            print(name_vector[sparserows[index]])
            raise


    return pd.DataFrame({'left_side': left_side,
                          'right_side': right_side,
                           'similarity': similarity})

def create_match_lookup(match_df):

    #Create match lookup dictionary
    match_lookup = {}
    for index,row in match_df.iterrows():
        if not match_lookup.get(row['right_side']):
            match_lookup[row['right_side']] = row['left_side']
        else:
            if len(row['left_side']) < len(match_lookup[row['right_side']]):
                match_lookup[row['right_side']] = row['left_side']

    #Make one-way references circular
    for key, value in match_lookup.copy().items():
        if not match_lookup.get(value):
            match_lookup[value] = key

    #Resolve circular references by favoring shorter names
    for key,value in match_lookup.items():
        if key == match_lookup[value]:
            if len(key) < len(value):
                match_lookup[key] = key
            elif len(key) > len(value):
                match_lookup[value] = value
            else:
                coin_flip = random.random()
                if coin_flip < 0.5:
                    match_lookup[key] = key
                else:
                    match_lookup[value] = value

    #Resolve multi-node reference chains to direct mappings of best name value
    for key, value in match_lookup.items():
        if key != match_lookup[value]:
            lookup_key = key
            lookup_value = value

            while lookup_key != lookup_value:
                lookup_key = lookup_value
                lookup_value = match_lookup[lookup_value]

            match_lookup[key] = lookup_value
            
    return {key: value for key,value in match_lookup.items() if key != value}
        


def splitting_artist_column():
    split_encoded_artist = encoded_artist.str.split(r"/|,|\*|vs| and |&|featuring|presents|feat|starring|with| -|\|| . |•",expand=True) 
    unique_artists = set()
    for column in split_encoded_artist.columns:
        unique_artists = unique_artists.union(set(split_encoded_artist[column].apply(lambda x: x.strip(' ') if x else x)))

def encode_formats(df):
    df = df.copy()

    for index, format_list in tqdm(enumerate(df.loc[:,'formats'])):
        format_dictionary = format_list[0]
        for key, value in format_dictionary.items():
            try:
                df.loc[index,key] = value
            except ValueError:
                df.loc[index,key] = 'üü'.join(value)
    return df

def make_format_description_column(format_list):
    return format_list[0].get('descriptions')

def make_format_name_column(format_list):
    return format_list[0].get('name')

def make_format_quantity_column(format_list):
    return int(format_list[0].get('qty'))

def make_format_text_column(format_list):
    return format_list[0].get('text')


def expand_format_description_column(df):
    df = df.copy()
    df_old_columns = set(df.columns)
    for index, description_list in tqdm(enumerate(df['format_description'])):
        if not description_list:
            continue
        for element in description_list:
            df.loc[index, element] = 1

    df_new_columns = set(df.columns) - df_old_columns

    format_df = df[df_new_columns].fillna(value=0).astype(np.uint8)
    #All entries are albums by default
    format_df.drop('Album',axis=1,inplace=True)
    format_df.rename(columns={column: 'format_description_{}'.format(column) for column in format_df.columns},inplace=True)

    return format_df



def convert_last_sold_encoding(last_sold):
    pass

def save_to_pkl(obj, name):
    file_name = '{}.pkl'.format(name) 
    with open(os.path.join(DATA_PATH,file_name),'wb') as f:
        pickle.dump(obj,f)

def load_from_pkl(name):
    file_name = '{}.pkl'.format(name) 
    with open(os.path.join(DATA_PATH,file_name),'rb') as f:
        df = pickle.load(f)
    return df

def get_cosine_similarity_matches(column,analyzer):
    unique_entries = column.unique()
    
    vectorizer = TfidfVectorizer(min_df=1, analyzer=analyzer)
    tf_idf_matrix = vectorizer.fit_transform(unique_entries)
    
    matches = cossine_similarity(tf_idf_matrix, tf_idf_matrix.transpose(), 10,0.9)
    
    matches_df = get_matches_df(matches, unique_entries, min(10000,len(unique_entries)))
    matches_df = matches_df[matches_df['similarity'] < 0.99999] #Remove identical entries
    
    return matches_df

from sklearn.neighbors import NearestNeighbors

def match_track_titles_to_standards(standards, track_titles):
    lowercase_no_punctuation = lambda x: x.lower().translate(str.maketrans('','',punctuation))
    standards_series = pd.Series(standards).apply(lowercase_no_punctuation)

    vectorizer = TfidfVectorizer(min_df=1,analyzer=get_ngrams)
    tfidf = vectorizer.fit_transform(standards_series)

    nbrs = NearestNeighbors(n_neighbors=1,n_jobs=-1).fit(tfidf)

    track_titles_expanded = []
    track_titles.progress_apply(lambda title_list: track_titles_expanded.extend([lowercase_no_punctuation(title) for title in pickle.loads(title_list)]))
    
    unique_track_titles = set(track_titles_expanded)

    def getNearestN(query):
        queryTFIDF_ = vectorizer.transform(query)
        distances, indices = nbrs.kneighbors(queryTFIDF_)
        return distances,indices

    distances, indices = getNearestN(unique_track_titles)

    unique_track_titles = list(unique_track_titles)
    matches = [[unique_track_titles[index], standards_series.values[indices_idx][0],round(distances[index][0],2)] for index, indices_idx in tqdm(enumerate(indices))]

    return pd.DataFrame(matches, columns=['Original Name','Matched Name','Match Confidence'])


    
