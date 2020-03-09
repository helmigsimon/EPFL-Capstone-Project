import os

from scripts.scrape import scrape_html, scrape_images, scrape_api, scrape_standards, extract_features_from_images, extract_features_from_scraped_html
from data.util.paths import DATA_PATH 

def main(scrape_api=False,scrape_html=False,scrape_images=False,extract_images=False,extract_html=False):
    if scrape_api:
        print('---API Data---\n')
        if 'jazz_album.pkl' not in os.listdir(DATA_PATH):
            print('Scraping API Data')
            scrape_api()
        
        print('API Data has been scraped')

    if scrape_html:
        print('---HTML Data---\n')
        scrape_html()

    if scrape_images:
        print('---Image Data---\n')
        try:
            scrape_images()
        except FileNotFoundError:
            print('Path to Image Data is unavailable\n')

    if extract_html:
        print('---Extract HTML Data---\n')
        extract_features_from_scraped_html()

    if extract_images:
        print('---Extract Image Features---\n')
        extract_features_from_images()

if __name__ == '__main__':
    main()