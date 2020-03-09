import datetime
import numpy as np
import os
import pickle
import pandas as pd
import random
import requests
import shutil
import string
import tensorflow as tf
import tensorflow_hub as hub
import time

from bs4 import BeautifulSoup
from PIL import Image
from typing import Dict, List
from fake_useragent import UserAgent, FakeUserAgentError
from itertools import cycle
from requests.exceptions import ConnectTimeout, HTTPError, ProxyError
from urllib3.exceptions import MaxRetryError
import sqlalchemy as db
from tqdm import tqdm
from copy import deepcopy

from data.util.paths import DATA_PATH, IMAGE_PATH_HD
from data.util.exceptions import ScrapingError, TooManyRequests, RequestFailed
from data.util.environment_variables import WIKIPEDIA_STANDARDS_URL, USD_EXCHANGE_120220, MONTH_INT_CONVERSION, PROXY_AUTH, MONTH_INT_CONVERSION, MOBILENET_V2_URL
from data.util.db import APIDataClient, ScrapedDataClient, ExtractedDataClient, HighLevelFeatureClient


def send_requests(links,scrape=True):
    proxy_pool, header_pool = create_pools(scrape)

    results = []

    for link in links:
        proxy, headers = iterate_proxy_and_header_pools(proxy_pool,header_pool)
        try:
            page = send_request(link,proxy,headers)
        except (ConnectTimeout, HTTPError) as e:
            print(''.join(['---------Request Error---------\n','Link: %s\n' % link, 'Exception: %s\n' % e]))
            proxy, headers = iterate_proxy_and_header_pools(proxy_pool,header_pool)
            page = send_request(link,proxy,headers)

        results.append(page)
    
    return results

def iterate_proxy_and_header_pools(proxy_pool,header_pool):
    return next(proxy_pool),next(header_pool)


def send_request(link,proxy,headers,error=False):
    with requests.Session() as req:
        page = req.get(link, proxies={"http": 'http://' + proxy, "https": 'https://' + proxy},
                    headers=headers, timeout=10)
    
    return page

class Scraper():
    def __init__(self,proxy:str,authentication: Dict[str,str]):
        self.proxy = proxy
        self.authentication = authentication
        self.session = requests.Session()
        self.proxies = {
            'http': 'http://%s:%s@%s' % (authentication['username'],authentication['password'],proxy),
            'https': 'http://%s:%s@%s' % (authentication['username'],authentication['password'],proxy)
        }

    def get(self,url:str,**kwargs):
        return self.session.get(url,proxies=self.proxies,**kwargs)
    
    def _random_header(self):
        accepts = {"Firefox": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                    "Safari, Chrome": "application/xml,application/xhtml+xml,text/html;q=0.9, text/plain;q=0.8,image/png,*/*;q=0.5"}
        
        try: 
            ua = UserAgent()
            if random.random() > 0.5:
                random_user_agent = ua.chrome
            else:
                random_user_agent = ua.firefox

        except FakeUserAgentError:
            user_agents = [
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/60.0.3112.113 Safari/537.36",
                "Mozilla/5.0 (Windows NT 5.1; rv:7.0.1) Gecko/20100101 Firefox/7.0.1"]  # Just for case user agents are not extracted from fake-useragent package
            random_user_agent = random.choice(user_agents)
        
        finally:
            valid_accept = accepts['Firefox'] if random_user_agent.find('Firefox') > 0 else accepts['Safari, Chrome']
            headers = {"User-Agent": random_user_agent,
                    "Accept": valid_accept}
        return headers



class DiscogsScraper(Scraper):
    def __init__(self):
        super().__init__(proxy='gate.dc.smartproxy.com:20000',authentication=PROXY_AUTH)
        self.scraped_data_client = ScrapedDataClient()
        self.api_data_client = APIDataClient()
        self.release_url = 'http://www.discogs.com/release/'

    def get_unscraped_release_ids(self):
        release_ids = set(self.api_data_client.get_entry_release_ids())
        scraped_release_ids = set(self.scraped_data_client.get_scraped_release_ids())
        
        return release_ids - scraped_release_ids

    def scrape_unscraped_releases(self):
        unscraped_release_ids = self.get_unscraped_release_ids()

        print(f'Number of unscraped release ids is: {len(unscraped_release_ids)}')
        
        unscraped_release_ids_urls = [(release_id,''.join([self.release_url,str(release_id)])) for release_id in unscraped_release_ids]
        headers = [self._random_header() for i in range(100)]
        header_pool = cycle(headers)
        
        for release_tuple in tqdm(unscraped_release_ids_urls):
            release_id, release_url = release_tuple
            headers = next(header_pool)

            proxy = True

            try:
                response = self.get(release_url,headers=headers)
            except requests.exceptions.ProxyError:
                proxy=False

            while response.status_code == 429 or not proxy:
                print(f'\n Status Code: {response.status_code}')
                print(f'\n Retrying {release_id}')
                time.sleep(5)
                try:
                    response = self.get(release_url,headers=headers)
                    proxy=True
                except requests.exceptions.ProxyError:
                    proxy=False

            if response.status_code not in (200,429):
                print(f'\n Status Code: {response.status_code}')
                print(f'\n Skipping {release_id}')
                continue
                
            response_pickle = pickle.dumps(response.text)

            self.scraped_data_client.insert_release({'release_id': release_id,'scraped_html':response_pickle})



class ImageScraper(Scraper):
    def __init__(self,image_path):
        super().__init__(proxy='gate.dc.smartproxy.com:20000',authentication=PROXY_AUTH)
        self.image_path = image_path
        self.scraped_images = {image_file_name: 1 for image_file_name in os.listdir(self.image_path)}
        self.scraped_data_client = ScrapedDataClient()
        self.scraped_data_extractor = ScrapedDataExtractor
    

    def download_image(self, image_id: int, image_url: str):
        image_file_name = f'{image_id}.png' if image_url else f'bitmap_{image_id}.png'
        image_file_path = os.path.join(self.image_path,image_file_name)
        
        if not image_url:
            random_bitmap_original = os.path.join(DATA_PATH,'randbitmap-rdo.png')
            shutil.copyfile(random_bitmap_original,image_file_path)
            self.scraped_images.update({image_file_name:1})
            return True
        try:
            response = self.get(image_url,stream=True)

            while response.status_code == 429:
                print(f'\n Status Code: {response.status_code}')
                print(f'\n Retrying {image_id}')
                time.sleep(5)
                response = self.get(image_url,stream=True)

            if response.status_code not in (200,429):
                raise RequestFailed(response.status_code)
        except Exception:
            raise RequestFailed

        with open(image_file_path, 'wb') as out_file:
            shutil.copyfileobj(response.raw,out_file)
        self.scraped_images.update({image_file_name:1})
        return True

    def get_unscraped_images(self):
        name_extractor = lambda x: int(x.split('.')[0]) if 'bitmap' not in x else int(x.split('.')[0].split('_')[1])
        scraped_image_ids = {name_extractor(image_file_name):value for image_file_name, value in self.scraped_images.items()}
        unscraped_image_entries = self.scraped_data_client.get_entries()
        id_scraped_html_tuples = ((unscraped_image_entry['release_id'],unscraped_image_entry['scraped_html']) for unscraped_image_entry in unscraped_image_entries if unscraped_image_entry['release_id'] not in self.scraped_images)
        return id_scraped_html_tuples

    def scrape_unscraped_images(self):
        id_scraped_html_tuples = self.get_unscraped_images()
        for id_, scraped_html in tqdm(id_scraped_html_tuples):
            image_file_name = f'{id_}.png' 
            bitmap_file_name = f'bitmap_{id_}.png'
            
            if (image_file_name in self.scraped_images) or (bitmap_file_name in self.scraped_images):
                print(f'Download Aborted - Image for {id_} already exists')
                continue

            cover_url = self.scraped_data_extractor(pickle.loads(scraped_html)).get_cover_image_url()
            
            try:
                self.download_image(id_,cover_url)
            except RequestFailed as e:
                print(f'\n Status Code: {e}')
                print(f'\n Skipping {id_}')
            


def get_standards():
    return BeautifulSoup(requests.get(WIKIPEDIA_STANDARDS_URL).text,'lxml')

def extract_standards(standards_soup):
    bullet_point_lists = standards_soup.findAll('ul')
    bullet_points = [link.text for link in bullet_point_lists]
    begin = 3
    end  = 28

    standards_groups = [
        bullet_points[i].split('\n') for i in range(begin,end+1)
    ]
    
    standards = []

    for group in standards_groups:
        for standard in group:
            alias_ = False
            for alias in ['(see ','(a.k.a. ','(orig. ','(short for']:
                if alias in standard.lower():
                        standard_names = standard.split(alias)
                        standard_names = (standard_names[0],standard_names[1].replace(')',''))
                        standards.extend(standard_names) 
                        alias_ = True
                
            if alias_ == True:
                break
            standards.append(standard)
            alias_=True


    return standards


def save_standards(standards):
    if 'standards.pkl' not in os.listdir(DATA_PATH):
        with open(os.path.join(DATA_PATH,'standards.pkl'),'wb') as standards_pkl:
            pickle.dump(standards,standards_pkl)
        print('Pickled standards list to DATA_PATH')
        return True
    print('Standards list already pickled in DATA_PATH')
    return False


class ScrapedDataExtractor:
    def __init__(self,scraped_html,scraped_data_client=None,extracted_data_client=None):
        self.scraped_html = scraped_html
        self.soup = BeautifulSoup(self.scraped_html,'lxml')

    def extract_data(self):
        extracted_data = {
            'market_price': self.get_market_price(),
            'units_for_sale': self.get_units_for_sale(),
            'have': self.get_have(),
            'want': self.get_want(),
            'average_rating': self.get_average_rating(),
            'rating_count': self.get_rating_count(),
            'last_sold': self.get_date_last_sold(),
            'number_of_tracks': self.get_number_of_tracks(),
            'running_time': self.get_album_running_time(),
            'track_titles': self.get_track_titles(),
        }
        extracted_data.update(self.get_last_ul())
        return extracted_data
    
    def convert_price_to_usd(self,price,symbol):
        return price / USD_EXCHANGE_120220[symbol]

    def convert_price_string_to_float(self, price):
        if price == '--':
            return None

        if ',' in price:
            price = price.replace(',','')
        
        for symbol in USD_EXCHANGE_120220.keys():
            if symbol in price:
                    float_price = float(price.replace(symbol,''))
                    return self.convert_price_to_usd(float_price,symbol)

        return price
    
    def get_market_price(self):
        try:
            market_price = self.soup.find('span',class_='price').text
        except AttributeError:
            return None
        return self.convert_price_string_to_float(market_price)
        
    
    def get_units_for_sale(self):
        try:
            units_for_sale = self.soup.find('span',class_='marketplace_for_sale_count').text
            return int(units_for_sale.replace('\n','').split(' ')[0])
        except (ValueError, AttributeError):
            return None
        

    def parse_stat(self,class_, type_):
        try:
            stat = self.soup.find('a',class_=class_).text
        except AttributeError:
            try:
                stat = self.soup.find('span',class_=class_).text
            except AttributeError:
                return None
        try:
            return type_(stat)
        except ValueError:
            return None    

    def get_have(self):
        return self.parse_stat('coll_num',int)

    def get_want(self):
        return self.parse_stat('want_num',int)

    def get_average_rating(self):
        return self.parse_stat('rating_value',float)

    def get_rating_count(self):
        return self.parse_stat('rating_count',int)

    def get_date_last_sold(self):
        try:
            date_last_sold = self.soup.find('li','last_sold').text.replace('Last Sold:','').replace(' ','').replace('\n','')
        except AttributeError:
            return None
        
        if date_last_sold == 'Never':
            return None
        
        day_str = date_last_sold[:2]
        year_str = date_last_sold[-2:]

        day_int = int(day_str[1]) if day_str[0] == '0' else int(day_str[:2])
        year_int = int(f'19{year_str}') if int(year_str) > 20 else int(f'20{year_str}')
        month_int = MONTH_INT_CONVERSION[date_last_sold.replace(day_str,'').replace(year_str,'').lower()]
        datetime_last_sold = datetime.datetime(year_int,month_int,day_int)
        return datetime_last_sold

    def get_last_ul(self):
        last_ul = self.soup.find('ul','last')
        parse = lambda x: x.text.replace(' ','').replace('\n','').split(':')
        try:
            prices = last_ul.findAll('li')
        except AttributeError:
            return {
                'lowest': None,
                'median': None,
                'highest': None
            }
        parsed_ul =  {parse(li)[0].lower(): self.convert_price_string_to_float(parse(li)[-1]) for li in prices} 
        try:
            del parsed_ul['lastsold']
        except:
            pass
        return parsed_ul

    def get_track_titles(self):
        try:
            tracklist = self.soup.findAll('span',class_='tracklist_track_title')
            track_titles = pickle.dumps([track.text for track in tracklist])

        except AttributeError as e:
            print(e)
            raise

        return track_titles

    #def _test_if_title_is_a_standard(self,title):
    #    if


    def get_number_of_tracks(self):
        try:
            tracklist = self.soup.find('table',class_='playlist').findAll('tr')
        except AttributeError:
            return None

        counter = 0
        for track in tracklist:
            #Don't want to count an overarching track twice
            if 'subtrack' not in track.attrs['class']:
                counter += 1
        return counter

    def get_album_running_time(self): 
        try:  
            track_durations_raw = self.soup.find('table',class_='playlist').findAll('td',{'class': 'tracklist_track_duration'})
        except AttributeError:
            return None
        minutes = 0
        for track in track_durations_raw:
            track_clean = track.text.replace('\n','').split(':')
            try:
                track_minutes = int(track_clean[0])
                track_seconds = int(track_clean[1])
            except ValueError:
                return None
            except IndexError:
                try:
                    track_minutes = int(track_clean[0][:-2])
                    track_seconds = int(track_clean[0][-2:])
                except ValueError:
                    return None


            minutes += (track_minutes + track_seconds/60)
        return minutes
        

    def get_cover_image_url(self):
        try:
            return self.soup.find('span',class_='thumbnail_center').img['src']
        except AttributeError:
            if (self.soup.find('i', class_='icon icon-vinyl') is None) and (self.soup.find('i',class_='icon icon-cd') is None) and (self.soup.find('i',class_='icon icon-cassette') is None) and (self.soup.find('i',class_='icon icon-digital') is None):
                raise
            return None



class FeatureExtractor:
    def __init__(self, module_url=MOBILENET_V2_URL, image_path = IMAGE_PATH_HD):
        self.module_url = module_url
        self.image_path = image_path
        self.HighLevelFeatureClient = HighLevelFeatureClient()

    def save_to_high_level_feature_table(self):
        high_level_feature_df = self.load_high_level_features_as_df()

        high_level_feature_df.to_sql('high_level_features',self.HighLevelFeatureClient.engine,if_exists='error')

    def load_high_level_features_as_df(self):
        high_level_features, labels = self.get_high_level_features()

        high_level_feature_cols = [f'feature_{i}' for i in range (1,high_level_features.shape[2]+1)]
        high_level_feature_df = pd.DataFrame(high_level_features.reshape(high_level_features.shape[0],high_level_features.shape[2]), columns=high_level_feature_cols)

        bitmap = [1 if 'bitmap_' in label else 0 for label in labels]
        
        convert_label_to_release_id = lambda x: int(x.split('.')[0].replace('bitmap_',''))
        release_ids = [convert_label_to_release_id(label) for label in labels]
        
        high_level_feature_df['bitmap'] = bitmap
        high_level_feature_df['release_id'] = release_ids

        high_level_feature_df = high_level_feature_df[['release_id','bitmap'] + high_level_feature_cols]

        return high_level_feature_df

    def get_high_level_features(self):
        if 'high_level_features.npz' in os.listdir(DATA_PATH):
            with np.load(os.path.join(DATA_PATH, 'high_level_features.npz')) as npz_file:
                X = npz_file['data']
                y = npz_file['label']
            
        else:
            self._extract_high_level_features()
            return self.get_high_level_features()

        return X,y
    
    def _load_extraction_module_and_parameters(self):
        return self._load_extraction_module()._get_feature_extraction_module_height_and_width()

    def _load_extraction_module(self):
        self.extraction_module = hub.Module(self.module_url)
        return self

    def _get_feature_extraction_module_height_and_width(self):
        self.module_height, self.module_width = hub.get_expected_image_size(self.extraction_module)
        return self

    def _create_feature_extraction_graph(self):
        self.feature_extraction_graph = tf.Graph()

        with self.feature_extraction_graph.as_default():
            if not hasattr(self,"extraction_module"):
                self._load_extraction_module_and_parameters()

            self.feature_node_placeholder = tf.placeholder(dtype=tf.float32,shape=[None,self.module_height,self.module_width,3])

            self.feature_node = self.extraction_module(self.feature_node_placeholder)

            self.tf_initializations = tf.group([
                tf.global_variables_initializer(),tf.tables_initializer()
            ])
        
        self.feature_extraction_graph.finalize()



    def _extract_high_level_features(self):
        if 'high_level_features.npz' in os.listdir(DATA_PATH):
            response = input('high_level_features.npz already exists. Do you want to overwrite this file? (y/n)')
            print('\n')
            if response.lower() not in ['y','n']:
                raise ValueError
            if response.lower() == 'n':
                return
        
        if not hasattr(self,"feature_extraction_graph"):
            self._create_feature_extraction_graph()

        image_arrays = self.load_image_arrays()

        session = tf.Session(graph=self.feature_extraction_graph)
        
        session.run(self.tf_initializations)

        high_level_features = (session.run(
            self.feature_node,
            feed_dict={self.feature_node_placeholder:image_array})
        for image_array in image_arrays)

        np.savez(os.path.join(DATA_PATH, f'high_level_features'),data=tuple(high_level_features),label=self.get_image_names())

        return high_level_features

    def get_image_names(self):
        image_file_names = os.listdir(self.image_path)
        return [image_file for image_file in image_file_names if self._check_file_type(image_file)]

    def convert_image_to_array(self,path):
        image = Image.open(path)
        
        image_resized = image.resize(
            [self.module_height,self.module_width],
            resample=Image.BILINEAR
        )

        image = np.array(image_resized,dtype=np.float32)[np.newaxis,:,:,:]/255

        return image

    def load_image_arrays(self):
        image_file_names = self.get_image_names()

        image_arrays = (self.convert_image_to_array(os.path.join(self.image_path,file_name)) for file_name in image_file_names)

        return image_arrays

    def _check_file_type(self, file):
        for ext in ['png','jpg','jpeg']:
            if ext in file:
                return True
        return False