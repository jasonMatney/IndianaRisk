import geopandas as gpd

def load_file(arg):
    output = gpd.read_file(arg[1])
    return arg[0], output


