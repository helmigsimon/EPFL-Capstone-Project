import discogs_client
import sqlite3
from sqlalchemy.sql.expression import bindparam
from bs4 import BeautifulSoup
import sqlalchemy as db 
import pickle
import time
import os
import sqlalchemy.exc
from typing import Dict
import requests
import pandas as pd
import datetime
from itertools import cycle

from data.util.paths import DATA_PATH
from data.util.db import APIDataClient

API_KEY = os.environ.get('DISCOGS_KEY')
YEAR_BEGIN = 1934
YEAR_END = 2020

class DiscogsClient(discogs_client.Client):
    def __init__(self,api_key):
        super().__init__('discogs_client',user_token=api_key)

    def get_album_releases_per_year(self,year,**kwargs):
        return self.search(type='release',format='album',year=year,**kwargs)

    def get_album_releases_for_year_range(self,year_begin,year_end,**kwargs):
        releases = {year: None for year in range(year_begin,year_end+1)}

        for year in releases.keys():
            try:
                releases[year] = list(self.get_album_releases_per_year(year,**kwargs))
               
            except discogs_client.exceptions.HTTPError as e:
                print(e)
                time.sleep(70)
                print('---------Resuming requests---------')
                
                releases[year] = list(self.get_album_releases_per_year(year,**kwargs))
            
            print('Got year %s' % year)

        return releases


def get_relevant_release_information(self, release, transform=lambda x:x):
    return {
        'release_id': release.id,
        'title': release.data['title'],
        'year': release.year,
        'country': release.country,
        'genre': transform(release.data['genre']),
        'style': transform(release.data['style']),
        'label': transform(release.data['label']),
        'community_have': release.data['community']['have'],
        'community_want': release.data['community']['want'],
        'formats': transform(release.formats),
        'master_id': release.data['master_id'],
        'thumb_url': release.data['thumb'],
        'release_url': release.data['resource_url']
    }


def save_jazz_album_releases_to_pkl(discogs_client: DiscogsClient, year_begin: int,year_end: int) -> None:
    jazz_album_releases = discogs_client.get_album_releases_for_year_range(year_begin,year_end,genre='jazz')

    with open(os.path.join(DATA_PATH,'jazz_album.pkl'),'wb') as f:
        pickle.dump(jazz_album_releases,f)


def save_jazz_album_releases_to_api_table() -> None:
    api_client = APIDataClient()

    with open(os.path.join(DATA_PATH,'jazz_album.pkl'),'rb') as f:
        jazz_album_releases = pickle.load(f)

    for year,releases in jazz_album_releases.items():
        release_data = [get_relevant_release_information(release,transform=pickle.dumps) for release in release_list]
        
        api_client.insert_multiple_releases(release_data)



if __name__ == '__main__':
    discogs_client = DiscogsClient(API_KEY)
    api_data_client = APIDataClient()
    
    if 'jazz_album.pkl' not in os.listdir(DATA_PATH):
        save_jazz_album_releases_to_pkl(discogs_client,YEAR_BEGIN,YEAR_END)

    try:
        api_data = api_data_client.get_table()
    except sqlalchemy.exc.NoSuchTableError:
        api_data = api_data_client.create_table(api_data_client.columns)
        save_jazz_album_releases_to_api_data_table()
