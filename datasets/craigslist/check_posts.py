import pandas as pd
# Install the [speedup] version with fast Levenstein distance
from fuzzywuzzy import fuzz, process as fuzproc

df = pd.read_csv('craigslist_posts.df.csv').sort_values('listing_price')
#!+TODO: analyze text and send requests o ones that are interesting

fuzproc.extract('mateo', df.title, limit=10)
fuzproc.extract('fitness', df.title, limit=10)
