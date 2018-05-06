"""
NYC Geolocated Vehicle Collitions 
Ref: https://data.cityofnewyork.us/Public-Safety/NYPD-Motor-Vehicle-Collisions/h9gi-nx95
"""
# Ref: https://docs.python.org/2/library/__future__.html
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from __future__ import unicode_literals

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# # Ref: https://flask-socketio.readthedocs.org
# from flask import Flask, render_template
# from flask_socketio import SocketIO

from datetime import datetime
import os
import json
import webbrowser
from tempfile import NamedTemporaryFile

"""
latitude
longitude
id
borough
contributing_factor_vehicle_1
contributing_factor_vehicle_2
number_of_cyclist_injured
number_of_cyclist_killed
number_of_motorist_injured
number_of_motorist_killed
number_of_pedestrians_injured
number_of_pedestrians_killed
number_of_persons_injured
number_of_persons_killed
off_street_name
on_street_name
vehicle_type_code1
vehicle_type_code2
zip_code
"""

def draw_coords_heatmap(df):
    """
    Plot dataframe latlon coordinates on map
    
    Input =>
      df: Pandas DataFrame, should contain columns ['lat', 'lon']
    """        
    def to_geojson_feature(rec):
        return {
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [rec['lon'], rec['lat']]
            },
        }

    def to_geojson_feature_collection(df):
        feature_list = list(df.apply(
            to_geojson_feature, axis=1))
        return {
            "type": "FeatureCollection",
            "features": feature_list
        }

    module_dirname = os.path.dirname(__file__)
    TMP_DIRNAME = "{dirnm}/._geojson_olyr3_render".format(dirnm=module_dirname)
    if not os.path.exists(TMP_DIRNAME):
        os.makedirs(TMP_DIRNAME)

    geojson = to_geojson_feature_collection(df.dropna())
    #fout = NamedTemporaryFile(suffix=".json", dir=TMP_DIRNAME)
    with open("{}/collisions_latlon.json".format(TMP_DIRNAME), 'w') as fout:
        json.dump(geojson, fout)
    #fout.close()

    webbrowser.open('{dirnm}/olyr3_nypd.html'.format(dirnm=module_dirname))

############################################################
if __name__ == "__main__":
    
    nyc_fname = "nyc_vehicle_collisions.csv"
    df_orig = pd.read_csv(nyc_fname)
        
    col_names = {
        'DATE': 'date',
        'TIME': 'time',
        'LATITUDE': 'lat',
        'LONGITUDE': 'lon',
        'NUMBER OF PERSONS INJURED': 'num_injured',
        'NUMBER OF PERSONS KILLED': 'num_killed'
    }
    df = df_orig[list(col_names.keys())].rename(columns=col_names)
    df["datetime"] = pd.to_datetime(ss, format="%m/%d/%Y %H:%M")

    df_with_death = df[ df['num_killed'] > 0 ]
    df_with_injury = df[ df['num_injured'] > 0 ]

############################################################


