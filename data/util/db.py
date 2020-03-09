from typing import List, Tuple, Dict
import sqlalchemy as db
from data.util.paths import DATA_PATH
import pandas as pd

class SQLClient():
    def __init__(self,name: str,path: str,dialect: str='sqlite'):
        self.name = name
        self.path = path
        self.dialect = dialect
        self.connect()

    def connect(self):
        self.engine = db.create_engine(''.join([self.dialect,':///',self.path,'/',self.name,'.',self.dialect]),connect_args={'check_same_thread':False})
        self.connection = self.engine.connect()
        self.metadata = db.MetaData()

class TableClient(SQLClient):
    columns = NotImplementedError

    def __init__(self,name: str,db_path: str,db_name: str,db_dialect: str):
        super().__init__(db_name,db_path,db_dialect)
        self.name = name
        self.create_table(self.columns)

    def create_table(self,columns: Tuple = None):
        try:
            self.table = self.get_table()
            return self.table
        except db.exc.NoSuchTableError:
            self.table = db.Table(self.name,self.metadata,*columns)
            self.metadata.create_all(self.engine)
            return self.table
    
    def get_table(self):
        self.table = db.Table(self.name, self.metadata, autoload=True, autoload_with=self.engine)
        return self.table
        
    def insert_release(self,entry: Dict):
        if not hasattr(self,'table'):
            raise AttributeError("TableClient object has no attribute 'table', execute the TableClient.create_table() method first")

        query = self.table.insert().values(**entry)
        return self.connection.execute(query)

    def insert_multiple_releases(self,releases: List[Dict]):
        query = self.table.insert()
        
        return self.connection.execute(query,releases)


    def get_table_as_dataframe(self):
        query = db.select([self.table])
        result = self.connection.execute(query)
        table_columns = [column.key for column in self.table.columns]
        return pd.DataFrame(result,columns=table_columns)

    def get_entry_by_release_id(self,id: int):
        query = db.select([self.table]).where(self.table.columns.release_id.is_(id))
        return self.connection.execute(query).fetchall()[0]

    def get_entry_release_ids(self):
        query = db.select([self.table.columns.release_id])
        return (release['release_id'] for release in self.connection.execute(query))

    def get_entry_release_ids_except_by_ids(self, ids: List[int]):
        query = db.select([self.table.columns.release_id]).where(self.table.columns.release_id not in ids)
        return [release['release_id'] for release in self.connection.execute(query)]

    def get_entries_by_release_ids(self,ids: List[int]):
        query = db.select([self.table]).where(self.table.columns.release_id.in_(ids))
        return [release for release in self.connection.execute(query)]

    def get_entries_except_by_ids(self, ids: List[int]):
        query = db.select([self.table]).where(self.table.columns.release_id not in ids)
        return [release for release in self.connection.execute(query)]

    def get_entries(self):
        query = db.select([self.table])
        return (release for release in self.connection.execute(query))



    
class APIDataClient(TableClient):
    columns: Tuple = (
        db.Column('id',db.Integer(), primary_key=True, autoincrement=True),
        db.Column('release_id',db.Integer()),
        db.Column('title',db.String()),
        db.Column('year',db.Integer()),
        db.Column('country',db.String(255)),
        db.Column('genre',db.PickleType()),
        db.Column('style', db.PickleType()),
        db.Column('label', db.PickleType()),
        db.Column('community_have',db.Integer()),
        db.Column('community_want',db.Integer()),
        db.Column('formats',db.PickleType()),
        db.Column('master_id',db.Integer()),
        db.Column('thumb_url',db.String()),
        db.Column('release_url',db.String())
    )

    def __init__(self):
        super().__init__('api_data',DATA_PATH,'jazz_album','sqlite')

    
    def get_entries_by_years(self,years: List[int]):
        query = db.select([self.table]).where(self.table.columns.year.in_(years))
        return self.connection.execute(query)

    def get_release_ids_by_years(self,years: List[int]) -> List[int]:
        return [release['release_id'] for release in self.get_entries_by_years(years)]

    def get_release_urls_by_years(self,years: List[int]) -> List[str]:
        return [release['release_url'] for release in self.get_entries_by_years(years)]

    def get_thumb_urls_by_ids(self,ids: List[int]) -> List[str]:
        return [release['thumb_url'] for release in self.get_entries_by_release_ids(ids)]

    def get_thumb_urls_except_by_ids(self,ids: List[int]) -> List[str]:
        return [release['thumb_url'] for release in self.get_entries_except_by_ids(ids)]

    def get_thumb_urls_by_years(self,years: List[int]) -> List[str]:
        return [release['thumb_url'] for release in self.get_entries_by_years(years)]


class ScrapedDataClient(TableClient):
    columns = (
        db.Column('id',db.Integer(), primary_key=True,autoincrement=True),
        db.Column('release_id',db.Integer()),
        db.Column('scraped_html',db.PickleType())
    )

    def __init__(self):
        super().__init__('scraped_data',DATA_PATH,'jazz_album','sqlite')

    def get_unscraped_releases(self):
        query = db.select([self.table]).where(self.table.columns.scraped_html.is_(None))
        return self.connection.execute(query).fetchall()

    def get_scraped_releases(self):
        query = db.select([self.table]).where(self.table.columns.scraped_html != None)
        return self.connection.execute(query).fetchall()

    def get_scraped_release_ids(self):
        return [release['release_id'] for release in self.get_scraped_releases()]

    def get_unscraped_release_ids(self):
        return [release['release_id'] for release in self.get_unscraped_releases()]

    def get_scraped_html_except_by_ids(self,ids):
        return [release['scraped_html'] for release in self.get_entries_except_by_ids(ids)]
    

class ExtractedDataClient(TableClient):
    columns = (
        db.Column('id',db.Integer(), primary_key=True, autoincrement=True),
        db.Column('release_id',db.Integer()),
        db.Column('market_price',db.Integer()),
        db.Column('units_for_sale',db.Integer()),
        db.Column('have', db.Integer()),
        db.Column('want', db.Integer()),
        db.Column('average_rating',db.Float()),
        db.Column('rating_count',db.Integer()),
        db.Column('last_sold',db.DateTime()),
        db.Column('number_of_tracks',db.Integer()),
        db.Column('running_time',db.Float()),
        db.Column('lowest',db.Float()),
        db.Column('median',db.Float()),
        db.Column('highest',db.Float()),
        db.Column('track_titles',db.PickleType())
    )

    def __init__(self):
        super().__init__('extracted_data',DATA_PATH,'jazz_album','sqlite')



class HighLevelFeatureClient(TableClient):
    columns = tuple(
        [
            db.Column('release_id',db.Integer(), primary_key=True),
            db.Column('bitmap',db.Integer(), nullable=False)
        ]+
        [
            db.Column(f'feature_{i}',db.String(), nullable=False) for i in range(1,1281)
        ]
    )

    def __init__(self):
        super().__init__('high_level_features',DATA_PATH,'jazz_album','sqlite')