import pandas as pd
import re
import pickle
from lib.processing import get_country_region_superregion, load_geoscheme_df
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

    def transform(self,X,y=None):
        X = X.copy()

        return self.split_feature(X)

class TitleSplitter(FeatureSplitter):
    def __init__(self):
        super().__init__('title',' - ',1,True)

    def split_feature(self,X):
        X = super().split_feature(X)

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
        self.pickle = pickle

    def fit(self,X,y=None):
        return self

    def stack_column(self,X):
        return X.loc[:,self.feature].apply(lambda x: 'üü'.join(x) if x else x).str.split('üü',expand=True).stack()

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
    def __init__(self,column='country', geoscheme_df=None):
        self.column = column
        self.geoscheme_df = geoscheme_df if geoscheme_df else load_geoscheme_df()

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

        return pd.concat([X,self.correct_mistaken_encodings(country_encoding)],axis=1)