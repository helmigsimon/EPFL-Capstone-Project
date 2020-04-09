import os
from pathlib import Path

FILE_PATH = os.path.dirname(os.path.abspath(__file__))
ROOT_PATH = Path(FILE_PATH).parent
DATA_PATH = os.path.join(ROOT_PATH,'data')
API_PATH = os.path.join(ROOT_PATH,'api')
IMAGE_PATH = os.path.join(DATA_PATH,'images')
try:
    IMAGE_PATH_HD = Path('/Volumes/Simon Helmig Harddrive/EPFL/DiscogsImages')
except:
    IMAGE_PATH_HD = None

