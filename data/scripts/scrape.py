import os
import pickle
import sqlalchemy
from pprint import pprint
from tqdm import tqdm
from traceback import print_exc

from data.util.api import DiscogsClient, save_jazz_album_releases_to_pkl, save_jazz_album_releases_to_api_table
from data.util.db import ScrapedDataClient, APIDataClient, ExtractedDataClient
from data.util.paths import DATA_PATH, IMAGE_PATH, IMAGE_PATH_HD
from data.util.scrape import DiscogsScraper, ScrapedDataExtractor, ImageScraper, get_standards, save_standards, extract_standards, FeatureExtractor, ScrapedDataExtractor

YEAR_BEGIN = 1934
YEAR_END = 2020

def scrape_api():
    discogs_client = DiscogsClient(os.environ.get('DISCOGS_KEY'))
    api_data_client = APIDataClient()
    
    if 'jazz_album.pkl' not in os.listdir(DATA_PATH):
        save_jazz_album_releases_to_pkl(discogs_client,YEAR_BEGIN,YEAR_END)

    try:
        api_data = api_data_client.get_table()
    except sqlalchemy.exc.NoSuchTableError:
        api_data = api_data_client.create_table(api_data_client.columns)
        save_jazz_album_releases_to_api_table()

def scrape_html():
    discogs_scraper = DiscogsScraper()
    discogs_scraper.scrape_unscraped_releases()


def scrape_images():
    image_scraper = ImageScraper(image_path=IMAGE_PATH_HD)
    image_scraper.scrape_unscraped_images()

def scrape_standards():
    standard_html = get_standards()
    standards = extract_standards(standard_html)
    save_standards(standards)
    return standards

def extract_features_from_images():
    feature_extractor = FeatureExtractor()
    feature_extractor.save_to_high_level_feature_table()

def extract_features_from_scraped_html():
    scraped_data_client = ScrapedDataClient()
    scraped_data = tuple(scraped_data_client.get_scraped_releases())

    extracted_data_client = ExtractedDataClient()
    extracted_ids = extracted_data_client.get_entry_release_ids()
    extracted_data_ids = {release_id: 1 for release_id in extracted_ids}
    del extracted_ids
    try:
        for release in tqdm(scraped_data):
            release_id, scraped_html = release['release_id'], release['scraped_html']
            if extracted_data_ids.get(release_id):
                print(f'Skipped {release_id}')
                continue
            scraped_data_extractor = ScrapedDataExtractor(pickle.loads(scraped_html))
            
            try:
                extracted_html = scraped_data_extractor.extract_data()
                extracted_html.update({'release_id':release_id})
                extracted_data_client.insert_release(extracted_html)
            except Exception as e:
                print(release_id)
                raise
    except Exception as e:
        print_exc(e)
