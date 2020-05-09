import rasterio
import geopandas as gpd

import pandas as pd

cleandata = pd.read_csv('data/csvs/cleandata.csv')
cleandata.head()

# Read points from shapefile
pts = cleandata[["longitude", "latitude"]].copy()
coords = [(x, y) for x, y in zip(pts.longitude, pts.latitude)]

# Open the raster and store metadata
src = rasterio.open('data/features/January/access.tif')

# Sample the raster at every point location and store values in DataFrame
cleandata['raster_value'] = [rasterValue[0]
                             for rasterValue in src.sample(coords)]
cleandata.to_csv("data/covid/cleandata_raster.csv", index=False)
