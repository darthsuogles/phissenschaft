import pandas as pd
from fuzzywuzzy import fuzz, process as fuzproc

df = pd.read_csv('craigslist_posts.df.csv').sort_values('listing_price')
#!+TODO: analyze text and send requests o ones that are interesting

fuzproc.extract('mateo', df.title, limit=10)
