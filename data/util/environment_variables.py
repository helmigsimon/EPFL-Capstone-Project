import os

MOBILENET_V2_URL ='https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/feature_vector/2'

WIKIPEDIA_STANDARDS_URL = 'https://en.wikipedia.org/wiki/List_of_jazz_standards'

USD_EXCHANGE_120220 = {
    'CA$': 1.324470,
    '$': 1,
    '€': 0.918149,
    '£': 0.770693,
    'CHF': 0.976852,
    '¥': 110.004475,
    'SEK': 9.626057 	
}

MONTH_INT_CONVERSION = {
    'jan': 1,
    'feb': 2,
    'mar': 3,
    'apr': 4,
    'may': 5,
    'jun': 6,
    'jul': 7,
    'aug': 8,
    'sep': 9,
    'oct': 10,
    'nov': 11,
    'dec': 12
}

PROXY_AUTH = {
    'username': os.environ.get('PROXY_USR'),
    'password': os.environ.get('PROXY_PWD')
}
