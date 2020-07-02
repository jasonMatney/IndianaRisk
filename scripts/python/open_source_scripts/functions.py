import geopandas as gpd
import rasterio
from skimage import io
import fiona

def load_shp_file(arg):
    output = gpd.read_file(arg[1])
    return arg[0], output


def load_gdb_shp_file(arg):
    output = gpd.read_file(arg[1], driver=arg[2], layer=arg[3])
    return arg[0], output

def load_tif_file(arg):
    output = io.imread(arg[1])
    return arg[0], output

def records(shapefile):
    with fiona.open(shapefile, ignore_fields = ['VINTAGE']) as source:
        for feature in source:
                yield feature
         
        
def shp_file_with_null(arg): 
    output = gpd.GeoDataFrame.from_features(records(arg[1]))
    return arg[0], output
    
