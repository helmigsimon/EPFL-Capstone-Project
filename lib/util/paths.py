import os
from pathlib import Path

LIB_PATH = Path(os.path.dirname(os.path.abspath(__file__))).parent
PIPELINE_PATH = os.path.join(LIB_PATH,'pipeline')
