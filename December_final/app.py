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

@st.cache
def get_data1():
    return  pd.read_csv('./test.csv')

airbnb_coord = get_data1()

@st.cache
def get_data2():
    return  pd.read_csv('./k_means_visual.csv')

k_means_visual = get_data2()

@st.cache
def get_data4():
    return pd.concat([k_means_visual, airbnb_coord], axis=1)
k_means_map =get_data4()

@st.cache
def get_data5():
    return geopandas.GeoDataFrame(k_means_map, geometry=geopandas.points_from_xy(k_means_map.longitude,k_means_map.latitude))
k_means_map_geo = geopandas.GeoDataFrame(k_means_map, geometry=geopandas.points_from_xy(k_means_map.longitude,k_means_map.latitude))

@st.cache
def get_data3():
    return geopandas.read_file("./neighbourhoods.geojson")

nbhoods = geopandas.read_file("./neighbourhoods.geojson")


st.title("DS11 CAPSTONE PROJECT: AIRBNB RECOMMENDER")
st.markdown("Welcome to my beta recommender, set your variables and we are all set")


neighbour = st.sidebar.selectbox("Neighbourhood", k_means_map_geo.neighbourhood_cleansed.unique(), 0)
min_nights = st.sidebar.selectbox("min_nights", [1,2,3,4,5,6,7,8,9,10], 0)
no_beds = st.sidebar.selectbox("no_of_beds", [1,2,3,4,5], 0)
values = st.sidebar.selectbox("max_price", [100,150,200,250,300], 0)
#values = st.sidebar.slider("Price_range", int(k_means_map_geo.price.min()), 1000, (10, 300))

listing = nbhoods[nbhoods.neighbourhood == neighbour].geometry

nb_listings = k_means_map_geo[(k_means_map_geo.neighbourhood_cleansed == neighbour) &
            (k_means_map_geo.bedrooms == no_beds) & (k_means_map_geo.minimum_nights == min_nights) &
            (k_means_map_geo.price < values)]

nb_listings.head()

m = folium.Map(location=[40.74601, -73.99987], zoom_start=14.5)
folium.GeoJson(listing).add_to(m)

# Build markers and popups
for row in nb_listings.iterrows():
    row_values = row[1]
    center_point = row_values['geometry']
    location = [center_point.y, center_point.x]
    if row_values['cluster'] == 0:
        marker_color = 'lightred'
    elif row_values['cluster'] == 1:
        marker_color = 'lightblue'
    else:
        marker_color = 'lightgreen'
    popup = ('NAME: ' + str(row_values['name']) +
            '  ' + 'PRICE: $' + str(row_values['price'])
            +'     '+ f'<a href="{row_values.listing_url} "target="_blank" > "Find Me Here" </a>')
    marker = folium.Marker(location = location, popup = popup, icon=folium.Icon(color=marker_color, icon='home'))
    marker.add_to(m)
st.markdown(m._repr_html_(), unsafe_allow_html=True)
