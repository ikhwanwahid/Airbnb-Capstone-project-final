import pandas as pd
import numpy as np
import os
import geopandas
import folium
from matplotlib.collections import PatchCollection
from matplotlib import pyplot
from shapely.geometry import LineString
from descartes import PolygonPatch
import streamlit as st


@st.cache
def get_data():
    return  pd.read_csv('./final_december_df.csv')

listing_comments = get_data()
listing_comments.head()
