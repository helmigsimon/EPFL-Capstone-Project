import discogs_client
import sqlite3
from bs4 import BeautifulSoup
import sqlalchemy as db 
import sqlalchemy.exc
import requests

API_KEY = 'evKJvYsWMnyRgRMCunMrgBJFsZeUfyBSkcGvzAfb'

def main():
    client = get_discogs_client()

    releases = get_all_jazz_releases(client)

    connection = connect_to_database()
    assert connection

    try:
        api_data = get_api_data_table(engine)
    except sqlalchemy.exc.NoSuchTableError:
        api_data = create_api_table(engine)






#Step 1 - Get all release IDs 
def get_discogs_client():
    return discogs_client.Client('discogs_price_prediction',user_token=API_KEY)

def get_all_jazz_releases(client):
    return client.search(type='release',genre='jazz')

def connect_to_database():
    engine = db.create_engine('sqlite:///jazz.sqlite')
    engine.connect()
    return engine


def create_api_data_table(engine):
    metadata = db.MetaData()
    api_data = db.Table('api_data', metadata,
        db.Column('id',db.Integer(), primary_key=True,nullable=False),
        db.Column('year',db.Integer()),
        db.Column('country',db.String(255)),
        db.Column('genre',db.Binary()),
        db.Column('style', db.Binary()),
        db.Column('label', db.Binary()),
        db.Column('community_have',db.Integer()),
        db.Column('community_want',db.Integer()),
        db.Column('formats',db.Binary())
    )
    
def get_api_data_table(engine): 
    metadata = db.MetaData()
    return db.Table('api_data', metadata, autoload=True, autoload_with=engine)

def extract_relevant_information(release):
    return (
        release.id,
        release.year,
        release.country,
        release.data['genre'],
        release.data['style'],
        release.data['label'],
        release.data['community']['have'],
        release.data['community']['want'],
        release.formats,
        release.data['master_id'],
    )


if __name__ == '__main__':
    main()