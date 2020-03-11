from scripts.processing import load_geoscheme_df, get_country_region_continent
from data.scripts.project_data import DataLoader
import pandas as pd
from unittest import TestCase

class TestGetGeoschemeTable(TestCase):
    def test_load_geoscheme_df(self):
        geoscheme_df = load_geoscheme_df()

        self.assertEqual(type(geoscheme_df),pd.DataFrame)
        self.assertEqual(list(geoscheme_df.columns),['country/region','numeric','0','1','2','3'])
        self.assertEqual(len(geoscheme_df),249)

class TestInterpretUniqueCountries(TestCase):
    loader = DataLoader()
    geoscheme_df = load_geoscheme_df()
    def test_apply_to_unique_countries(self):
        api_data = self.loader.load_api_data()
        unique_countries = api_data['country'].unique()

        converted_unique_countries = {name: get_country_region_continent(name,self.geoscheme_df) for name in unique_countries}

        unsuccessful = {name: name_dict for name, name_dict in converted_unique_countries.items() if 'unknown' in name_dict.values()}

        print(unsuccessful)
        self.assertEqual(len(unsuccessful),1)
