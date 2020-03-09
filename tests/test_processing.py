from scripts.processing import get_geoscheme_table
import pandas as pd
from unittest import TestCase

class TestGetGeoschemeTable(TestCase):
    def test_get_geoscheme_table(self):
        geoscheme_df = get_geoscheme_table()

        self.assertEqual(type(geoscheme_df),pd.DataFrame)
        self.assertEqual(list(geoscheme_df.columns),['country/region','numeric','m49'])
        self.assertEqual(len(geoscheme_df),249)
