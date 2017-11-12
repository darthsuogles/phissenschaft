from pathlib import Path
import requests
import pandas as pd

local_data_root = Path.home() / 'local' / 'data'
uci_root_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases'

def beijing_pollution():
    fp = local_data_root / 'beijing_pollution_monitor.csv'
    if not fp.exists():
        url = '{}/00381/PRSA_data_2010.1.1-2014.12.31.csv'.format(uci_root_url)
        resp = requests.get(url, stream=True)
        with fp.open('wb') as fout:
            for chunk in resp.iter_content(chunk_size=None):
                fout.write(chunk)

    return pd.read_csv(str(fp), index_col='No')
