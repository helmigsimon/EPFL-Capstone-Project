import pandas as pd
import requests
from bs4 import BeautifulSoup
import os


def get_geoscheme_table():
    GEOSCHEME_WIKI_URL = 'https://en.wikipedia.org/wiki/List_of_countries_by_United_Nations_geoscheme'
    
    response = requests.get(GEOSCHEME_WIKI_URL)
    soup = BeautifulSoup(response.text)

    rows = []
    headers = ['country/region','capital','alpha-2','alpha-3','numeric','m49']

    for tr in soup.find_all('tr')[2:]:
        tds = tr.find_all('td')
        rows.append({
            header: entry.text for header, entry in zip(headers,tds)
        })

    geoscheme_df = pd.DataFrame(rows).applymap(lambda x: x.replace('\n','')).drop(['capital','alpha-2','alpha-3'],axis=1)

    return geoscheme_df




def get_countries_and_regions(country_list):
    pass

def classify_country(country):
    if country.lower() == 'australia':
        return {
            'country': 'australia',
            'geoscheme': 'australasia',
            'region': 'australia'
        }

    if country.lower() in REGIONS:
        return {
            'country': 'unknown',
            'geoscheme': 'unknown',
            'region': country
        }

    if country.lower() in GEOSCHEMES_CONTINENTS:
        return {
            'country': 'unknown',
            'geoscheme': country,
            'region': GEOSCHEMES_CONTINENTS[country.lower()].title()
        }

    if country.lower() in COUNTRIES:
        pass
