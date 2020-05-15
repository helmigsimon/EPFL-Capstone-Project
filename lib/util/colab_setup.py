import sys,os,subprocess

class ColabConfiguration:
    def __init__(self,algorithms):
        self.setup_drive()
        self.algorithms = algorithms

    def main():
        self.upgrade_runtime_ram()

        #Setting up PyPi Packages
        !pip install geopandas sparse-dot-topn pdpipe category-encoders catboost
        global gdp, ct, pdp, category_encoders
        import geopandas as gpd
        import sparse_dot_topn.sparse_dot_topn as ct
        import pdpipe as pdp
        import category_encoders

        #Setting up Conda Packages
        self.setup_conda()
        
        #Initializing NLTK
        global nltk
        import nltk
        nltk.download('stopwords')
        nltk.download('punkt')
        
        #Setting up RAPIDS AI
        self.setup_rapids()
        
        #Importing CUML algorithms
        self.import_algorithms()
 
    def setup_drive(self):
        global drive
        from google.colab import drive
        drive.mount('/content/drive')
        self._drive_mounted = True

    def upgrade_runtime_ram():
        meminfo = subprocess.getoutput('cat /proc/meminfo').split('\n')
        memory_info = {entry.split(':')[0]: int(entry.split(':')[1].replace(' kB','').strip()) for entry in meminfo}
        if memory_info['MemTotal'] > 17000000:
            return
        a = []
        while(1):
            a.append('1')

    def _restart_runtime():
        os.kill(os.getpid(), 9)

    def setup_conda(self):
        if not 'Miniconda3-4.5.4-Linux-x86_64.sh' in os.listdir():
            !wget https://repo.continuum.io/miniconda/Miniconda3-4.5.4-Linux-x86_64.sh && bash Miniconda3-4.5.4-Linux-x86_64.sh -bfp /usr/local

        if not ('EPFL-Capstone-Project' in os.listdir()) and (os.getcwd().split('/')[-1] != 'EPFL-Capstone-Project'):
            !git clone https://github.com/helmigsimon/EPFL-Capstone-Project  
        if 'EPFL-Capstone-Project' in os.listdir():
            os.chdir('EPFL-Capstone-Project')

        !conda env create -f environment.yml
        !conda activate exts-ml

        self._conda_setup = True
        
    def setup_rapids(self):
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        device_name = pynvml.nvmlDeviceGetName(handle)
        if (device_name != b'Tesla T4') and (device_name != b'Tesla P4') and (device_name != b'Tesla P100-PCIE-16GB'):
            print("Wrong GPU - Restarting Runtime")
            restart_runtime()

        # clone RAPIDS AI rapidsai-csp-utils scripts repo
        !git clone https://github.com/rapidsai/rapidsai-csp-utils.git

        # install RAPIDS
        !bash rapidsai-csp-utils/colab/rapids-colab.sh 0.13

        # set necessary environment variables 
        dist_package_index = sys.path.index('/usr/local/lib/python3.6/dist-packages')
        sys.path = sys.path[:dist_package_index] + ['/usr/local/lib/python3.6/site-packages'] + sys.path[dist_package_index:]
        sys.path

        # update pyarrow & modules 
        exec(open('rapidsai-csp-utils/colab/update_modules.py').read(), globals())

    def import_algorithms(self):
        for algorithm in self.algorithms:
            global algorithm
            from cuml import algorithm
