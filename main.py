import requests
import pandas as pd
import geopandas as gpd
import numpy as np
from shapely.geometry import Point
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import folium
import os
import warnings
warnings.filterwarnings("ignore")

from test import (
    AIRBNB_CSV_PATH, SUBWAY_CSV_PATH, CENSUS_API_KEY,
    OUT_DIR, ACS_YEAR, TIGER_YEAR, STATE_FIPS
)

os.makedirs(OUT_DIR, exist_ok=True)


# Data Reading
def read_airbnb(path):
    if path.endswith(".gz"):
        return pd.read_csv(path, compression='gzip', low_memory=False)
    return pd.read_csv(path, low_memory=False)

def read_subway(path):
    df = pd.read_csv(path, low_memory=False)
    cols = {c.lower(): c for c in df.columns}
    def pick(variants):
        for v in variants:
            if v in cols:
                return cols[v]
        return None
    latcol = pick(['gtfs latit', 'gtfs latitude', 'lat', 'y'])
    loncol = pick(['gtfs long', 'gtfs longitude', 'lon', 'lng', 'x'])
    namecol = pick(['stop name', 'station_name', 'name'])
    df = df.rename(columns={latcol: 'latitude', loncol: 'longitude'})
    df['station_name'] = df[namecol] if namecol else df.index.astype(str)
    return df


# Spatial Data Acquisition
def get_tract_shapefile(state, year):
    url = f"https://www2.census.gov/geo/tiger/TIGER{year}/TRACT/tl_{year}_{state}_tract.zip"
    return gpd.read_file(url)

def get_acs_data(state, year, vars, key):
    varstr = ",".join(vars)
    url = f"https://api.census.gov/data/{year}/acs/acs5?get=NAME,{varstr}&for=tract:*&in=state:{state}&key={key}"
    res = requests.get(url)
    data = res.json()
    df = pd.DataFrame(data[1:], columns=data[0])
    df['GEOID'] = df['state'] + df['county'] + df['tract']
    for v in vars:
        df[v] = pd.to_numeric(df[v], errors='coerce')
    return df


if __name__ == "__main__":
    df_airbnb_raw = read_airbnb(AIRBNB_CSV_PATH)
    df_subway_raw = read_subway(SUBWAY_CSV_PATH)

    # Cleaning Airbnb
    price_col = next((c for c in df_airbnb_raw.columns if 'price' in c.lower()), None)
    lat_col = next((c for c in df_airbnb_raw.columns if c.lower() == 'latitude'), None)
    lon_col = next((c for c in df_airbnb_raw.columns if c.lower() == 'longitude'), None)
    df_airbnb = df_airbnb_raw[[lat_col, lon_col, price_col]].copy()
    df_airbnb = df_airbnb.rename(columns={lat_col: 'latitude', lon_col: 'longitude', price_col: 'price'})
    df_airbnb['price'] = df_airbnb['price'].replace(r'[\$,]', '', regex=True).astype(float, errors='ignore')
    df_airbnb = df_airbnb.dropna(subset=['latitude', 'longitude', 'price'])
    gdf_airbnb = gpd.GeoDataFrame(df_airbnb, geometry=gpd.points_from_xy(df_airbnb.longitude, df_airbnb.latitude), crs="EPSG:4326")

    # Cleaning Subway
    df_subway = df_subway_raw.dropna(subset=['latitude', 'longitude'])
    gdf_subway = gpd.GeoDataFrame(df_subway, geometry=gpd.points_from_xy(df_subway.longitude, df_subway.latitude), crs="EPSG:4326")

    # Acquire ACS and tract data
    vars_acs = ['B01003_001E', 'B19013_001E']
    df_acs = get_acs_data(STATE_FIPS, ACS_YEAR, vars_acs, CENSUS_API_KEY)
    gdf_tracts = get_tract_shapefile(STATE_FIPS, TIGER_YEAR)
    gdf_tracts = gdf_tracts.merge(df_acs[['GEOID', 'B01003_001E', 'B19013_001E']], on='GEOID', how='left')
    gdf_tracts = gdf_tracts.rename(columns={'B01003_001E': 'population', 'B19013_001E': 'median_income'})

    # Spatial Matching
    gdf_airbnb_m = gdf_airbnb.to_crs(epsg=3857)
    gdf_tracts_m = gdf_tracts.to_crs(epsg=3857)
    gdf_airbnb_with_tract = gpd.sjoin(gdf_airbnb_m, gdf_tracts_m[['GEOID', 'median_income', 'geometry']], how='left', predicate='within')

    # Nearest subway distance
    gdf_subway_m = gdf_subway.to_crs(epsg=3857)
    from scipy.spatial import cKDTree
    coords_sub = list(zip(gdf_subway_m.geometry.x, gdf_subway_m.geometry.y))
    tree = cKDTree(coords_sub)
    coords_air = list(zip(gdf_airbnb_m.geometry.x, gdf_airbnb_m.geometry.y))
    dists, idxs = tree.query(coords_air, k=1)
    gdf_airbnb_with_tract['dist_to_subway_m'] = dists
    gdf_airbnb_with_tract['nearest_station'] = [gdf_subway.iloc[i]['station_name'] for i in idxs]

    # Clustering
    coords = pd.DataFrame({'x': gdf_airbnb_m.geometry.x, 'y': gdf_airbnb_m.geometry.y})
    gdf_airbnb_with_tract['cluster'] = KMeans(n_clusters=5, random_state=42).fit_predict(coords)

    # Regression
    df_model = gdf_airbnb_with_tract.dropna(subset=['price', 'dist_to_subway_m', 'median_income'])
    df_model['log_price'] = np.log1p(df_model['price'])
    df_model['dist_km'] = df_model['dist_to_subway_m'] / 1000
    X = df_model[['dist_km', 'median_income']]
    y = df_model['log_price']
    reg = LinearRegression().fit(X, y)

    # Visualization
    m = folium.Map(location=[40.7128, -74.0060], zoom_start=11)
    for _, row in gdf_airbnb_with_tract.sample(2000).iterrows():
        folium.CircleMarker([row.geometry.y, row.geometry.x], radius=2,
                            color=['blue','green','purple','orange','red'][row['cluster']]).add_to(m)
    m.save(os.path.join(OUT_DIR, 'map.html'))

    gdf_airbnb_with_tract.drop(columns='geometry').to_csv(os.path.join(OUT_DIR, 'result.csv'), index=False)