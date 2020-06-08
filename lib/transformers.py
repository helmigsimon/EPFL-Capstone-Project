import pandas as pd
import re
import pickle
import string
from collections.abc import Iterable, Callable, Collection
from data.util.environment_variables import COUNTRIES, REGIONS, SUPERREGIONS
from lib.processing import *

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.random import sample_without_replacement

class RowRemover(BaseEstimator, TransformerMixin):
    """
    Abstract Base Class for Removing Rows in a DataFrame
    """
    def __init__(self, features):
        self.features = features

    def fit(self,X,y=None):
        return self

    def remove(self,X):
        raise NotImplementedError

    def transform(self, X, y = None):
        X = X.copy()

        return self.remove(X)

class ConditionalRowRemover(RowRemover):
    """
    Removes Rows according to a Condition
    """
    def __init__(self, features, condition: Callable,**apply_args):
        super().__init__(features)
        self.condition = condition
        self.apply_args = apply_args

    def remove(self, X):
        return X[X[self.features].apply(self.condition,**self.apply_args)]


class NullRemover(RowRemover):
    """
    Removes Nulls from a DataFrame for specified features
    """
    def __init__(self,features):
        super().__init__(features)

    def remove(self,X):
        if (not isinstance(self.features,Collection)) or isinstance(self.features,str):
            self.features = [self.features]
        return X.dropna(subset=self.features)

class DuplicateRemover(RowRemover):
    """
    Removes Duplicates from a DataFrame for specified features
    """
    def __init__(self,features):
        super().__init__(features)

    def remove(self,X):
        return X.drop_duplicates(self.features)

class FeatureSplitter(BaseEstimator,TransformerMixin):
    """
    Splits a string feature according to a pre-defined delimiter into a List or DataFrame
    """
    def __init__(self,feature,delimiter,n,expand):
        self.feature = feature
        self.delimiter = delimiter
        self.n = n
        self.expand = expand

    def fit(self,X,y=None):
        return self

    def split_feature(self,X):
        return pd.concat([X,X[self.feature].str.split(self.delimiter,n=self.n,expand=self.expand)],axis=1)
        return X[self.feature].str.split(self.delimiter,n=self.n,expand=self.expand)

    def transform(self,X,y=None):
        X = X.copy()

        return self.split_feature(X)

class TitleSplitter(FeatureSplitter):
    """
    Capstone-Specific
    -----------------
    Implementation of the FeatureSplitter Base Class for handling the Title Feature
    """
    def __init__(self):
        super().__init__('title',' - ',1,True)

    def split_feature(self,X):
        X = super().split_feature(X)
        X.drop(self.feature,axis=1,inplace=True)
        return X.rename(columns={0:'artist',1:'title'})

class ColumnCombiner(BaseEstimator, TransformerMixin):
    """
    Fills the nulls of a base column in a dataframe with the values of another column for the same indices
    """
    def __init__(self,base_column,merge_column, new_column=None):
        self.base_column = base_column
        self.merge_column = merge_column
        self.new_column = new_column if new_column else '_'.join((base_column,merge_column))

    def fit(self,X,y=None):
        return self

    def transform(self, X, y=None):
        X = X.copy()
    
        X.loc[:,self.new_column] = X.loc[:,self.base_column]
        
        base_column_null_idx = X.loc[:,self.base_column][X.loc[:,self.base_column].isnull()].index
        
        X.loc[base_column_null_idx,self.new_column] = X.loc[base_column_null_idx,self.merge_column]

        return X
        

class RunningTimeImputer(BaseEstimator, TransformerMixin):
    """
    Capstone-Specific
    -----------------
    Implementation of a Transformer which imputes the running_time feature according to the number of tracks listed for each album
    """
    def __init__(self,running_time, number_of_tracks):
        self.running_time = running_time
        self.number_of_tracks = number_of_tracks
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        X = X.copy()
        
        if not hasattr(self,'average_time_per_track'):
            self.average_time_per_track = X.loc[:,self.running_time].mean() / X.loc[:,self.number_of_tracks].mean()
            
        null_indices = X[X.loc[:,self.running_time].isna()].index
        
        X.loc[null_indices,self.running_time] = X.loc[null_indices,self.number_of_tracks] * self.average_time_per_track
        
        return X
        
class ColumnRemover(BaseEstimator,TransformerMixin):
    """
    Removes unwanted columns from a DataFrame
    """
    def __init__(self,remove_columns):
        self.remove_columns = remove_columns
    
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = X.copy()
        
        if type(self.remove_columns) == tuple:
            self.remove_columns = list(self.remove_columns)
            
        return X.drop(self.remove_columns,axis=1)

class ConditionalColumnConsolidator(ColumnRemover):
    """
    Removes columns which do not meet a certain condition from a DataFrame
    """
    def __init__(self, condition,**kwargs):
        super().__init__(remove_columns=None)
        self.condition = condition

    def fit(self, X, y=None):
        return self

    def set_remove_columns(self,X,y=None):
        self.remove_columns = []
        for column in tqdm(X.columns):
            try:
                if not self.condition(X[column]):
                    self.remove_columns.append(column)
            except:
                pass

    def transform(self, X, y=None):
        self.set_remove_columns(X,y)
        return super().transform(X,y)

class RemoveZeroIndicators(ConditionalColumnConsolidator):
    """
    Removes indicator features from DataFrame which sum to zero
    """
    def __init__(self, **kwargs):
        super().__init__(self.remove_zero_indicators,**kwargs)

    def remove_zero_indicators(self,x):
        if not isinstance(x,np.object):
            try:
                return x.sum()>0
            except TypeError:
                return True
        return True

class DummyGenerator(BaseEstimator,TransformerMixin):
    def __init__(self,feature):
        self.feature = feature

    def fit(self,X,y=None):
        return self

    def transform(self, X, y=None):
        X = X.copy()

        dummies = pd.get_dummies(X.loc[:,self.feature],prefix=self.feature,dtype=np.uint8)

        X = pd.concat([X,dummies],axis=1)

        return X


class MultiValueCategoricalEncoder(BaseEstimator,TransformerMixin):
    """
    Dummy encodes features with array-like datatypes
    """
    def __init__(self, feature):
        self.feature = feature

    def fit(self,X,y=None):
        return self

    def stack_column(self,X):
        return X.loc[:,self.feature]\
                .apply(lambda x: 'üü'.join(x) if type(x)==list else str(x))\
                .str.split('üü',expand=True)\
                .stack()

    def get_dummies(self,stacked_column):
        return pd.get_dummies(stacked_column,prefix=self.feature)\
                .groupby(level=0).sum()
    
    def transform(self, X, y=None):
        X = X.copy()

        stacked_column = self.stack_column(X)

        dummies = self.get_dummies(stacked_column)

        return pd.concat([X,dummies],axis=1)

class Unpickler(BaseEstimator, TransformerMixin):
    """
    Unpickles features with pickle datatype
    """
    def __init__(self,columns):
        self.columns = [columns] if isinstance(columns,str) else columns 

    def fit(self,X,y=None):
        return self

    def transform(self,X,y=None):
        X = X.copy()

        X.loc[:,self.columns] = X[self.columns].astype(bytes)

        for column in self.columns:
            X.loc[:,column] = X.loc[:,column].apply(pickle.loads)

        return X


class GenreEncoder(MultiValueCategoricalEncoder):
    """
    Capstone-Specific
    -----------------
    Implementation of the MultiValueCategoricalEncoder for the genre feature
    """
    def __init__(self,column='genre'):
        super().__init__(column)
   
    def stack_column(self,X): 
        return super().stack_column(X).apply(lambda x: 'Childrens' if x == "Children's" else x)
       
    def get_dummies(self, stacked_column):
        return super().get_dummies(stacked_column).drop('_'.join([self.feature,'Jazz']),axis=1)

class CountryEncoder(BaseEstimator, TransformerMixin):
    """
    Capstone-Specific
    -----------------
    Implementation of an encoder for the country feature into country_, region_ and superregion_ dummy variables
    """
    def __init__(self,column='country', geoscheme_df=None,country=True,region=True,superregion=True):
        self.column = column
        self.geoscheme_df = geoscheme_df if geoscheme_df else load_geoscheme_df()
        self.country= country
        self.region = region
        self.superregion = superregion
        self.to_drop = zip((country,region,superregion),('country_','region_','superregion_'))

    def fit(self,X,y=None):
        return self

    def get_country_mapping(self,X):
        unique_countries = X.loc[:,self.column].unique()
        return {country: get_country_region_superregion(self.geoscheme_df,country) for country in unique_countries}

    def encode_column(self,mapping):
        return pd.get_dummies(pd.DataFrame(pd.Series(mapping).to_dict()).T.applymap(lambda x: str(x) if type(x) == list else x))

    def correct_mistaken_encodings(self,encoded_df):
        mistakes = list(filter(lambda x: "_[" in x,encoded_df.columns))
        correct_names = set(COUNTRIES + REGIONS + SUPERREGIONS)
        entity = re.compile(r"\'(.+?)\'")
        for mistake in mistakes:
            mistaken_indices = encoded_df[encoded_df[mistake]==1].index
            category, mistaken_regions = mistake.split('_')
            regions = entity.findall(mistaken_regions)
            for region in regions:
                if region not in correct_names:
                    region = entity.findall(region)
                sub_column = ''.join([category,'_',region])
                encoded_df.loc[mistaken_indices,sub_column] = 1
        encoded_df.fillna(int(0),inplace=True)
        encoded_df.drop(mistakes,axis=1,inplace=True)
        return encoded_df.astype(np.uint8)

    def transform(self,X,y=None):
        X = X.copy()

        country_mapping = self.get_country_mapping(X)

        country_encoding = self.encode_column(X.loc[:,self.column].map(country_mapping))

        for bool_, prefix in self.to_drop:
            if not bool_:
                drop_columns = list(filter(lambda x: prefix in x, country_encoding.columns))
                country_encoding.drop(drop_columns,axis=1,inplace=True)

        return pd.concat([X,self.correct_mistaken_encodings(country_encoding)],axis=1)

class StringMatcher:
    """
    Inspired by the following blogpost: https://bergvca.github.io/2017/10/14/super-fast-string-matching.html
    --------------
    Implements TF-IDF N-Gram String Matching
    """
    def __init__(self,similarity_threshold=0):
        self.similarity_threshold = similarity_threshold

    def get_match_lookup(self,series):
        matches_df = self.get_cosine_similarity_matches(series,self.get_ngrams)

        match_lookup = self.create_match_lookup(matches_df)

        return series.apply(lambda x: match_lookup[x] if match_lookup.get(x) else x)


    def get_cosine_similarity_matches(self,column,analyzer):
        unique_entries = column.unique()
        
        vectorizer = TfidfVectorizer(min_df=1, analyzer=analyzer)
        tf_idf_matrix = vectorizer.fit_transform(unique_entries)
        
        matches = cossine_similarity(tf_idf_matrix, tf_idf_matrix.transpose(), 10,0.9)
        
        matches_df = get_matches_df(matches, unique_entries, min(10000,len(unique_entries)))
        matches_df = matches_df[matches_df['similarity'] < 0.99999] #Remove identical entries
        
        return matches_df

    def get_ngrams(self,string_, n=3):
        ngrams = zip(*[string_[i:] for i in range(n)])
        return [''.join(ngram) for ngram in ngrams]

    def create_match_lookup(self,match_df):
        #Create match lookup dictionary
        match_lookup = {}
        for index,row in match_df.iterrows():
            if not match_lookup.get(row['right_side']):
                match_lookup[row['right_side']] = row['left_side']
            else:
                if len(row['left_side']) < len(match_lookup[row['right_side']]):
                    match_lookup[row['right_side']] = row['left_side']

        #Make dead-end references circular
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
        
        return {
            key: value 
            for key,value in match_lookup.items() 
            if self.filter_match_lookup(key,value)
        }
        
    def filter_match_lookup(self,key,value):
        if key == value:
            return False
        if np.abs(len(key) - len(value)) <= self.similarity_threshold:
            return False
        return True

class FeatureCleanReduce(BaseEstimator, TransformerMixin):
    """
    Abstract Base Class for the application of TF-IDF N-Grams String Matching 
    """
    def __init__(self,feature,matcher_threshold=0):
        self.feature = feature
        self.string_matcher = StringMatcher(matcher_threshold)

    def fit(self,X,y=None):
        return self

    def clean(self,entry):
        raise NotImplementedError

    def transform(self,X,y=None):
        X = X.copy()
        clean_feature = X.loc[:,self.feature].apply(self.clean)
        match_lookup = self.string_matcher.get_match_lookup(clean_feature)
        X.loc[:,self.feature] = clean_feature.apply(lambda x: match_lookup[x] if match_lookup.get(x) else x)
        return X  

class LabelCleanReduce(FeatureCleanReduce):
    """
    Capstone-Specific
    -----------------
    Cleans and reduces the label feature for downstream label encoding
    """
    def __init__(self,feature='label'):
        super().__init__(feature,2)
        self.multi_value_encoder = MultiValueCategoricalEncoder(feature)

    def fit(self,X,y=None):
        return self

    def clean(self,entry):
        entry = entry.lower()
        entry = remove_words(entry,LABEL_REMOVAL_WORDS)
        entry = remove_paren_num(entry)
        entry = remove_punctuation(entry)
        entry = remove_excess_spaces(entry)
        entry = remove_plural(entry)
        return entry

    def transform(self,X,y=None):
        X = X.copy()
        stacked_column = self.multi_value_encoder\
                                                .stack_column(X)\
                                                .apply(self.clean)
                                                
        match_lookup = self.string_matcher.get_match_lookup(stacked_column)
        
        X.loc[:,self.feature] =  stacked_column\
                                               .apply(lambda x: match_lookup[x] if match_lookup.get(x) else x)\
                                               .unstack()\
                                               .loc[:,0]
        return X



class ArtistCleanReduce(FeatureCleanReduce):
    """
    Capstone-Specific
    -----------------
    Cleans and reduces the artist feature for downstream label encoding
    """
    def __init__(self,feature='artist'):
        super().__init__(feature)

    def clean(self,entry):
        entry = entry.lower()
        entry = remove_words(entry,ARTIST_REMOVAL_WORDS)
        entry = remove_paren_num(entry)
        entry = remove_punctuation(entry)
        entry = remove_excess_spaces(entry)
        return entry

    def clean_main_artist(self,row):
        if set(row.values) == set([None]):
            return row
        if isinstance(row[0],str) and len(row[0].strip(' ')) > 3:
            return row
        return self.clean_main_artist(pd.Series(list(row[1:].values)+[None],index=row.index))
    
    def clean_column(self, column):
        return column\
                .str.lower()\
                .str.replace(r"[-,\\|]",'üü')\
                .str.replace(r"[^a-zA-Z\d\s']",'')\
                .str.replace('and','üü')\
                .str.replace(r"[ü]+$","")\
                .str.split('üü',expand=True)
    
        
    def transform(self,X,y=None):
        X = X.copy()
        stacked_column = self.clean_column(X.loc[:,self.feature])\
                            .apply(self.clean_main_artist,axis=1)\
                            .stack()\
                            .apply(self.clean)

        match_lookup = self.string_matcher.get_match_lookup(stacked_column)

        X.loc[:,self.feature] = stacked_column\
                                    .apply(lambda x: match_lookup[x] if match_lookup.get(x) else x)\
                                    .unstack()\
                                    .loc[:,0]
        return X

class FormatEncoder(BaseEstimator,TransformerMixin):
    """
    Capstone-Specific
    -----------------
    Creates the columns for the keys present in the format dictionary feature
    """
    def __init__(self,feature='formats'):
        self.feature = feature

    def fit(self,X,y=None):
        return self

    def make_format_name(self,format_list):
        return format_list[0].get('name')

    def make_format_quantity(self,format_list):
        return int(format_list[0].get('qty'))

    def make_format_text(self,format_list):
        return format_list[0].get('text')

    def make_format_description(self,format_list):
        return format_list[0].get('descriptions')

    def transform(self,X,y=None):
        X = X.copy()
        feature_function_mapping = zip(('format_name','format_quantity','format_text','format_description'), (self.make_format_name,self.make_format_quantity,self.make_format_text,self.make_format_description))

        for feature, function in feature_function_mapping:
            X.loc[:,feature] = X.loc[:,self.feature].apply(function)

        return X


class FormatDescriptionEncoder(MultiValueCategoricalEncoder):
    """
    Capstone-Specific
    -----------------
    Implementation of the MultiValueCategoricalEncoder for the format_description feature
    """
    def __init__(self,feature='format_description'):
        super().__init__(feature)

    def transform(self,X,y=None):
        X.loc[:,self.feature] = X.loc[:,'format_description'].apply(lambda x: x[0].get('descriptions'))
        return super().transform(X)


class FormatTextCleanReduce(FeatureCleanReduce):
    """
    Capstone-Specific
    -----------------
    Implementation of FeatureCleanReduce for the format_text feature
    """
    def __init__(self,feature='format_text'):
        super().__init__(feature)

    def fit(self,X,y=None):
        return super().fit(X,y)

    def transform(self,X,y=None):
        return super().transform(X,y)

    def clean(self,entry):
        if not entry:
            return str(entry)
        entry = entry.lower()
        entry = remove_punctuation(entry)
        entry = remove_excess_spaces(entry)
        return entry


class TimePeriodEncoder(BaseEstimator, TransformerMixin):
    """
    Capstone-Specific
    -----------------
    Dummy Encodes the Jazz Eras and Periods associated with the time-range considered in this project
    """
    time_periods = {
        'era': {
            'swing': (1925,1945),
            'modern': (1940,1970),
            'contemporary': (1970,2020)
        },
        'period': {
            'big_band': (1930,1950),
            'bebop': (1940,1955),
            'cool': (1950,1970),
            'fusion': (1970,2020)
        }
    }
    
    def __init__(self,feature='year'):
        self.feature = feature

    def fit(self,X,y=None):
        return self

    def _make_time_period_column(self,year,start,end):
        if start <= year <= end:
            return 1
        return 0
    
    def transform(self,X,y=None):
        X = X.copy()
        
        for category, time_periods in self.time_periods.items():
            for time_period, year_tuple in time_periods.items():
                start,end = year_tuple
                X.loc[:,'_'.join([category,time_period])] = X.loc[:,self.feature].apply(self._make_time_period_column,start=start,end=end)

        return X


class StandardCountEncoder(BaseEstimator, TransformerMixin):
    """
    Capstone-Specific
    -----------------
    Counts the occurrence of Jazz Standards in the track listings of each album
    """
    def __init__(self,feature='track_titles',standards=None,**kwargs):
        self.feature = feature
        if not standards:
            standards = load_from_pkl('standards',path=kwargs.get('path'))
        self.standards = standards
        self._standards_lookup = self.get_standards_lookup()

    def fit(self,X,y=None):
        matched_track_titles = self.match_track_titles_to_standards(X.loc[:,self.feature])

        matched_track_titles = matched_track_titles[matched_track_titles['Match Confidence'] < 0.7]

        tfidf_lookup = {row['Original Name']:row['Matched Name'] for _, row in matched_track_titles.iterrows() if row['Original Name'] not in self._standards_lookup}

        self._standards_lookup = dict(**self._standards_lookup,**tfidf_lookup)
        
        return self

    def get_standards_lookup(self):
        lowercase_no_punctuation = lambda x: x.lower().translate(str.maketrans('','',string.punctuation))
        return {lowercase_no_punctuation(standard):0 for standard in self.standards}

    def count_jazz_standards(self,title_list):
        standards_counter = 0
        for title in title_list:
            title = str(title).lower().translate(str.maketrans('', '', string.punctuation))
            if title in self._standards_lookup:
                standards_counter += 1               
        return standards_counter

    def match_track_titles_to_standards(self, track_titles):
        lowercase_no_punctuation = lambda x: x.lower().translate(str.maketrans('','',string.punctuation))
        
        standards_series = pd.Series(self.standards).apply(lowercase_no_punctuation)

        vectorizer = TfidfVectorizer(min_df=1,analyzer=get_ngrams)
        tfidf = vectorizer.fit_transform(standards_series)

        nbrs = NearestNeighbors(n_neighbors=1,n_jobs=-1).fit(tfidf)

        track_titles_expanded = []
        track_titles.apply(lambda title_list: track_titles_expanded.extend([lowercase_no_punctuation(title) for title in title_list]))
        
        unique_track_titles = set(track_titles_expanded)

        def getNearestN(query):
            queryTFIDF_ = vectorizer.transform(query)
            distances, indices = nbrs.kneighbors(queryTFIDF_)
            return distances,indices

        distances, indices = getNearestN(unique_track_titles)

        unique_track_titles = list(unique_track_titles)
        matches = [[unique_track_titles[index], standards_series.values[indices_idx][0],round(distances[index][0],2)] for index, indices_idx in tqdm(enumerate(indices))]

        return pd.DataFrame(matches, columns=['Original Name','Matched Name','Match Confidence'])   

    def transform(self, X, y=None):
        X = X.copy()

        X.loc[:,'_'.join(['standards_count'])] = X.loc[:,self.feature].apply(self.count_jazz_standards)

        return X


class ColumnStore(BaseEstimator, TransformerMixin):
    """
    Stores the Columns of a DataFrame according to a pre-defined Column Set
    """
    def __init__(self):
        pass
    
    def _get_column_set(self,str_):
        return set(column for column in self._all if str_ in column)

    def _get_null_columns(self, X,threshold=None):
        if not threshold:
            threshold = 0
        return set(filter(lambda x: X[x].isna().sum() > threshold,self._all))

    def _filter_by_function(self, func, X):
        columns = []
        for column in self._all:
            try:
                if func(X.loc[:,column]):
                    columns.append(column)
            except TypeError:
                pass
        
        return set(columns)

    def fit(self,X,column_sets,**kwargs):
        X = X.copy()
        self._all = set(X.columns)
        self._rest = self._all.copy()

        assert type(column_sets) == dict

        for name, columns in tqdm(column_sets.items()):
            set_name = f'_{name}'
            if columns is None:
                setattr(self,set_name,self._get_null_columns(X,kwargs.get('threshold')))
                continue

            if isinstance(columns,Callable):
                setattr(self,set_name,self._filter_by_function(columns,X))
                continue
            
            if not isinstance(columns,Iterable):
                print(name)
                raise TypeError('Value of column_sets attribute must be iterable')
       
            setattr(self,set_name,set())
            if isinstance(columns,dict):
            
                for column_group, columns_ in columns.items():
                    if type(columns_) == str:
                        columns_ = self._get_column_set(columns_)
                    column_group_set = set(columns_)
                    column_group_name = '_'.join([set_name,column_group])
                    setattr(self,column_group_name,column_group_set)
                    setattr(self,set_name,set((
                        *getattr(self,set_name),
                        *column_group_set
                    )))
                self._rest -= getattr(self,set_name)
                continue

            if isinstance(columns,str):
                columns = self._get_column_set(columns)

            columns = set(columns)
            setattr(self,set_name,columns)
            self._rest -= columns

class OutlierRemover(BaseEstimator, TransformerMixin):
    """
    Removes Rows from a DataFrame which exceed the three standard deviation range from the mean for the passed features
    """
    def __init__(self,features,sigma=3):
        self.features = [features] if isinstance(features,str) else features
        self.sigma = 3

    def fit(self,X,y=None):
        X_ = pd.DataFrame([X.loc[:,self.features].mean(),X.loc[:,self.features].std()],columns=self.features,index=('mean','sigma'))
        X_.loc['upper_bound',:] = X_.loc['mean',:] + 3*X_.loc['sigma',:]
        X_.loc['lower_bound',:] = X_.loc['mean',:] - 3*X_.loc['sigma',:]

        self._bounds = {feature: (X_.loc['lower_bound',feature],X_.loc['upper_bound',feature]) for feature in self.features}        

        return self

    def transform(self,X,y=None):
        X = X.copy()

        for feature in self.features:
            if not X[feature].dtype in (int,float):
                continue
            X = X[X[feature] > self._bounds[feature][0]]
            X = X[X[feature] < self._bounds[feature][1]]

        return X

class IndicatorCounter(BaseEstimator,TransformerMixin):
    """
    Counts the number of total categories of a specific indicator group for each row
    """
    def __init__(self,columns, counter_name):
        self.columns = columns
        self.counter_name = counter_name

    def fit(self,X,y=None):
        X = X.copy()

        self.indicator_counter = X.loc[:,self.columns].sum(axis=1)

        return self

    def transform(self,X,y=None):
        X = X.copy()
        X.loc[:,self.counter_name] = self.indicator_counter
        return X    


class IndicatorConsolidator(IndicatorCounter):
    """
    Consolidates a given set of indicators into the most important indicators and records the rest in an indicator_counter column
    """
    def __init__(self, output_column,columns=None, threshold=None, counter_name=None):
        super().__init__(columns,counter_name)
        self.columns = columns
        self.threshold = threshold
        self.output_column = output_column

    def fit(self, X, y=None):
        if (self.counter_name) and (self.counter_name not in self.columns):
            super().fit(X)

        if not self.columns:
            self.columns = X.columns        

        if not self.threshold:
            self.threshold = X[self.columns].sum().median()

        self.consolidation_columns = list(filter(lambda x: X[x].sum() < self.threshold,self.columns))        
        return self

    def transform(self, X, y=None):
        X = X.copy()
        if hasattr(self,'indicator_counter'):
            X = super().transform(X)
        X.loc[:,self.output_column] = X.loc[:,self.consolidation_columns].max(axis=1)

        assert X.loc[:,self.output_column].sum() == X.loc[:,self.consolidation_columns].max(axis=1).sum()

        return X.drop(self.consolidation_columns,axis=1)    

class LastSoldEncoder(BaseEstimator,TransformerMixin):
    """
    Capstone-Specific
    -----------------
    Encodes the last_sold feature of the data
    """
    def __init__(self,feature,new_feature=None,end_date=None):
        self.feature = feature
        self.end_date = end_date
        self.new_feature = new_feature

    def fit(self,X,y=None):
        if not self.end_date:
            self.end_date = X[self.feature].max()
        if not self.new_feature:
            self.new_feature = '_'.join['days_since',self.feature]
        return self

    def transform(self,X,y=None):
        X = X.copy()
        X.loc[:,self.new_feature] = X.loc[:,self.feature].apply(lambda x: (self.end_date-x).days)
        assert X.loc[:,self.counter_name].sum() == X.loc[:,self.columns].sum(axis=1).sum()
        return X
        
class IndicatorReducer(BaseEstimator,TransformerMixin):
    """
    Reduces a set of indicators using a dimensionality reduction algorithm
    """
    def __init__(self, indicators, algorithm,components,reduced_column_prefix= 'indicator_reduced'):
        self.indicators = indicators
        self.algorithm = algorithm
        self.components = components
        self.reduced_column_prefix = reduced_column_prefix
  
    def fit(self,X,y=None):
        return self

    def transform(self,X,y=None):
        X = X.copy()

        algorithm = self.algorithm(n_components=self.components)

        reduced_indicators = pd.DataFrame(
            algorithm.fit_transform(X.loc[:,self.indicators].values),
            index=X.index,
            columns=['_'.join([self.reduced_column_prefix,str(i)]) for i in range(self.components)]
            )

        return pd.concat([X, reduced_indicators],axis=1)


