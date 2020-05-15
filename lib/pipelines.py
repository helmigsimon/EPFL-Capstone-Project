from sklearn.pipeline import Pipeline
from data.util.paths import DATA_PATH
from .transformers import (
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
    OutlierRemover
)

#Extracted Data Pipe
extracted_pipe = Pipeline([
    ('remove_id', ColumnRemover('id')),
    ('unpickle', Unpickler(['track_titles'])),
    ('remove_duplicates', DuplicateRemover('release_id')),
    ('count_standards',StandardCountEncoder('track_titles',DATA_PATH)),
    ('count_days_since_last_sale',LastSoldEncoder(feature='last_sold',new_feature='days_since_last_sale'))
])

market_value_pipe = Pipeline([
    ('make_market_value', ColumnCombiner('median','market_price','market_value')),
    ('remove_nulls',NullRemover('market_value')),
    ('remove_outliers', OutlierRemover('market_value')) 
])

#API Pipes
clean_text_pipe = Pipeline([
    ('label', LabelCleanReduce()),
    ('artist', ArtistCleanReduce())
])

column_encoding_pipe = Pipeline([
    ('country',CountryEncoder()),
    ('genre',GenreEncoder()),
    ('style', MultiValueCategoricalEncoder(feature='style'))
])

format_pipe = Pipeline([
    ('make_columns', FormatEncoder()),
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

