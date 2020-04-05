import pandas as pd
import re
import pickle
import string
from lib.processing import *
from lib.util.processing import *
from data.util.environment_variables import COUNTRIES, REGIONS, SUPERREGIONS

from sklearn.base import BaseEstimator, TransformerMixin

class RowRemover(BaseEstimator, TransformerMixin):
    def __init__(self,features):
        self.features = features

    def fit(self,X,y=None):
        return self

    def remove(self,X):
        raise NotImplementedError

    def transform(self, X, y = None):
        X = X.copy()

        return self.remove(X)

class NullRemover(RowRemover):
    def __init__(self,features):
        super().__init__(features)

    def remove(self,X):
        return X.dropna(self.features)

class DuplicateRemover(RowRemover):
    def __init__(self,features):
        super().__init__(features)

    def remove(self,X):
        return X.drop_duplicates(self.features)

class FeatureSplitter(BaseEstimator,TransformerMixin):
    def __init__(self,feature,delimiter,n,expand):
        self.feature = feature
        self.delimiter = delimiter
        self.n = n
        self.expand = expand

    def fit(self,X,y=None):
        return self

    def split_feature(self,X):
        return pd.concat([X,X[self.feature].str.split(self.delimiter,n=self.n,expand=self.expand)],axis=1)
        #return X[self.feature].str.split(self.delimiter,n=self.n,expand=self.expand)

    def transform(self,X,y=None):
        X = X.copy()

        return self.split_feature(X)

class TitleSplitter(FeatureSplitter):
    def __init__(self):
        super().__init__('title',' - ',1,True)

    def split_feature(self,X):
        X = super().split_feature(X)
        X.drop(self.feature,axis=1,inplace=True)
        return X.rename(columns={0:'artist',1:'title'})

class ColumnCombiner(BaseEstimator, TransformerMixin):
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
    def __init__(self,cols_to_remove):
        self.cols_to_remove = cols_to_remove
    
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = X.copy()
        
        if type(self.cols_to_remove) == tuple:
            self.cols_to_remove = list(self.cols_to_remove)
        elif type(self.cols_to_remove) != list:
            raise TypeError
            
        return X.drop(list(self.cols_to_remove),axis=1)

class MultiValueCategoricalEncoder(BaseEstimator,TransformerMixin):
    def __init__(self, feature):
        self.feature = feature

    def fit(self,X,y=None):
        return self

    def stack_column(self,X):
        return X.loc[:,self.feature].apply(lambda x: 'üü'.join(x) if type(x)==list else str(x)).str.split('üü',expand=True).stack()

    def get_dummies(self,stacked_column):
        return pd.get_dummies(stacked_column,prefix=self.feature).groupby(level=0).sum()
    
    def transform(self, X, y=None):
        X = X.copy()

        stacked_column = self.stack_column(X)

        dummies = self.get_dummies(stacked_column)

        return pd.concat([X,dummies],axis=1)

class Unpickler(BaseEstimator, TransformerMixin):
    def __init__(self,columns):
        self.columns = columns

    def fit(self,X,y=None):
        return self

    def transform(self,X,y=None):
        X = X.copy()

        X.loc[:,self.columns] = X[self.columns].astype(bytes)

        for column in self.columns:
            X.loc[:,column] = X.loc[:,column].apply(pickle.loads)

        return X


class GenreEncoder(MultiValueCategoricalEncoder):
    def __init__(self,column='genre'):
        super().__init__(column)

    def stack_column(self,X):
        return super().stack_column(X).apply(lambda x: 'Childrens' if x == "Children's" else x)

    def get_dummies(self, stacked_column):
        return super().get_dummies(stacked_column).drop('_'.join([self.feature,'Jazz']),axis=1)




class CountryEncoder(BaseEstimator, TransformerMixin):
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
        return encoded_df

    def transform(self,X,y=None):
        X = X.copy()

        country_mapping = self.get_country_mapping(X)

        country_encoding = self.encode_column(X.loc[:,self.column].map(country_mapping))

        for bool_, prefix in self.to_drop:
            if not bool_:
                drop_columns = list(filter(lambda x: prefix in x, country_encoding.columns))
                country_encoding.drop(drop_columns,axis=1)

        return pd.concat([X,self.correct_mistaken_encodings(country_encoding)],axis=1)

class StringMatcher:
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
        
        return {key: value for key,value in match_lookup.items() if self.filter_match_lookup(key,value)}
        
    def filter_match_lookup(self,key,value):
        if key == value:
            return False
        if np.abs(len(key) - len(value)) <= self.similarity_threshold:
            return False
        return True

class FeatureCleaner(BaseEstimator, TransformerMixin):
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

class LabelCleaner(FeatureCleaner):
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
        stacked_column = self.multi_value_encoder.stack_column(X).apply(self.clean)
        match_lookup = self.string_matcher.get_match_lookup(stacked_column)
        X.loc[:,self.feature] =  stacked_column.apply(lambda x: match_lookup[x] if match_lookup.get(x) else x).unstack().loc[:,0]
        return X



class ArtistCleaner(FeatureCleaner):
    def __init__(self,feature='artist'):
        super().__init__(feature)

    def clean(self,entry):
        entry = entry.lower()
        entry = remove_words(entry,ARTIST_REMOVAL_WORDS)
        entry = remove_paren_num(entry)
        entry = remove_punctuation(entry)
        entry = remove_excess_spaces(entry)
        return entry

class FormatEncoder(BaseEstimator,TransformerMixin):
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
    def __init__(self,feature='format_description'):
        super().__init__(feature)

    def transform(self,X,y=None):
        X.loc[:,self.feature] = X.loc[:,'format_description'].apply(lambda x: x[0].get('descriptions'))
        return super().transform(X)


class FormatTextCleaner(FeatureCleaner):
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
    def __init__(self,feature='track_titles',standards=None):
        self.feature = feature
        if not standards:
            standards = load_from_pkl('standards')
        self.standards = standards
        self.standards_lookup = self.get_standards_lookup()

    def fit(self,X,y=None):
        return self

    def get_standards_lookup(self):
        lowercase_no_punctuation = lambda x: x.lower().translate(str.maketrans('','',string.punctuation))
        return {lowercase_no_punctuation(standard):0 for standard in self.standards}

    def count_jazz_standards(self,title_list):
        standards_counter = 0
        for title in title_list:
            title = title.lower().translate(str.maketrans('', '', string.punctuation))
            if title in self.standards_lookup:
                standards_counter += 1               
        return standards_counter

    def match_track_titles_to_standards(self, track_titles):
        lowercase_no_punctuation = lambda x: x.lower().translate(str.maketrans('','',string.punctuation))
        
        standards_series = pd.Series(self.standards).apply(lowercase_no_punctuation)

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

    def transform(self, X, y=None):
        X = X.copy()
        matched_track_titles = self.match_track_titles_to_standards(X.loc[:,self.feature])

        matched_track_titles = match_track_titles[match_track_titles['Match Confidence'] < 0.7]

        tfidf_lookup = {row['Original Name']:row['Matched Name'] for _, row in match_title_to_standards_df.iterrows() if row['Original Name'] not in self.standards_lookup}

        self.standards_lookup = dict(**self.standards_lookup,**tfidf_lookup)

        X.loc[:,'_'.join([self.feature,'count'])] = X.loc[:,self.feature].apply(self.count_jazz_standards)

        return X


