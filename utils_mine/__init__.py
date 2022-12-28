from utils_mine.log import *
from utils_mine.plot import *
from utils_mine.gpu import *

def make_output_folders(root_dir, folder_names, makedirs=True):
    dirs = {}
    for name in folder_names:
        odir = os.path.join(root_dir, name)
        if makedirs:
            os.makedirs(odir, exist_ok=True)
        dirs[name] = odir
    return dirs