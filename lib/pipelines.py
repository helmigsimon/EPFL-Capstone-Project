from sklearn.pipeline import Pipeline
from data.util.paths import DATA_PATH
from lib.transformers import (
    ColumnRemover,
    Unpickler,
    ColumnCombiner,
    DuplicateRemover,
    NullRemover,
    StandardCountEncoder,
    LastSoldEncoder,
    LabelCleanReduce,
    ArtistCleanReduce,
    CountryEncoder,
    GenreEncoder,
    MultiValueCategoricalEncoder,
    FormatEncoder,
    FormatTextCleanReduce,
    TitleSplitter,
    TimePeriodEncoder,
    OutlierRemover,
    ColumnStore,
    IndicatorConsolidator,
    ConditionalColumnConsolidator,
    ConditionalRowRemover,
    DummyGenerator
)

#Extracted Data Pipes
market_value_pipe = Pipeline([
    ('make_market_value', ColumnCombiner('median','market_price','market_value')),
    ('remove_nulls',NullRemover('market_value')),
    ('remove_outliers', OutlierRemover('market_value')) 
])

extracted_pipe = Pipeline([
    ('remove_id', ColumnRemover('id')),
    ('unpickle', Unpickler(['track_titles'])),
    ('remove_duplicates', DuplicateRemover('release_id')),
    ('count_standards',StandardCountEncoder('track_titles',DATA_PATH)),
    ('market_value_pipe', market_value_pipe)
])

#API Pipes
clean_text_pipe = Pipeline([
    ('label', LabelCleanReduce()),
    ('artist', ArtistCleanReduce()),
])

column_encoding_pipe = Pipeline([
    ('country',CountryEncoder()),
    ('genre',GenreEncoder()),
    ('style', MultiValueCategoricalEncoder(feature='style'))
])

format_pipe = Pipeline([
    ('make_columns', FormatEncoder()),
    ('encode_format_name', DummyGenerator('format_name')),
    ('remove_quantity_outliers', OutlierRemover('format_quantity')),
    ('encode_descriptions',MultiValueCategoricalEncoder('format_description')),
    ('clean_format_text',FormatTextCleanReduce())
])

api_pipe = Pipeline([
    ('remove_columns', ColumnRemover('id')),
    ('split_title', TitleSplitter()),
    ('unpickle', Unpickler(['genre','style','label','formats'])),
    ('clean_text',clean_text_pipe),
    ('remove_duplicates', DuplicateRemover('release_id')),
    ('encode_columns',column_encoding_pipe),
    ('format_columns', format_pipe),
    ('encode_time_periods', TimePeriodEncoder())
])

genres = ['Pop','Rock','Funk / Soul','Latin','Classical','Blues','Electronic','Brass & Military','Hip Hop','Non-Music','Stage & Screen','Childrens','Reggae','Folk, World, & Country']

def make_data_consolidation_pipe(df, column_store):
    return Pipeline([
        ('focus_pure_jazz', ConditionalRowRemover(column_store._genre,lambda x: x.sum()==0,axis=1)),
        ('focus_post_1950', ConditionalRowRemover('year',lambda x: x>1950)),
        ('drop_null_columns', ConditionalColumnConsolidator(condition=lambda x: x.sum()>0 if x.dtypes==np.uint8 else True)),
        ('style_consolidator', IndicatorConsolidator(
            output_column='style_Other',
            columns=column_store._style,
            threshold=int(0.1*len(df)), 
        )),
        ('format_description_consolidator', IndicatorConsolidator(
            columns=column_store._format_description,
            output_column='format_description_Other',
            threshold=int(0.1*len(df)),
        )),
        ('format_name_consolidator', IndicatorConsolidator(
            output_column='format_name_other',
            columns=column_store._format_name,
            threshold=5000
        )),
        ()
    ])

