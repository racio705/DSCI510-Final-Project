import requests
import pandas as pd
import geopandas as gpd
import numpy as np
from shapely.geometry import Point
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns
import folium
import os
import warnings

warnings.filterwarnings("ignore")

# Import configurations
from test import (
    AIRBNB_CSV_PATH, SUBWAY_CSV_PATH, CENSUS_API_KEY,
    OUT_DIR, ACS_YEAR, TIGER_YEAR, STATE_FIPS
)

os.makedirs(OUT_DIR, exist_ok=True)


# Data reading functions
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

    if latcol is None or loncol is None:
        raise ValueError("Latitude/longitude columns not identified in subway data")

    df = df.rename(columns={latcol: 'latitude', loncol: 'longitude'})
    df['station_name'] = df[namecol] if namecol else df.index.astype(str)
    return df


def get_tract_shapefile(state, year):
    url = f"https://www2.census.gov/geo/tiger/TIGER{year}/TRACT/tl_{year}_{state}_tract.zip"
    return gpd.read_file(url)


def get_acs_data(state, year, vars, key):
    varstr = ",".join(vars)
    url = f"https://api.census.gov/data/{year}/acs/acs5?get=NAME,{varstr}&for=tract:*&in=state:{state}&key={key}"
    res = requests.get(url)
    res.raise_for_status()
    data = res.json()
    df = pd.DataFrame(data[1:], columns=data[0])
    df['GEOID'] = df['state'] + df['county'] + df['tract']

    for v in vars:
        df[v] = pd.to_numeric(df[v], errors='coerce')

    return df


if __name__ == "__main__":
    df_airbnb_raw = read_airbnb(AIRBNB_CSV_PATH)
    df_subway_raw = read_subway(SUBWAY_CSV_PATH)

    # Clean Airbnb data
    expected_cols = ['id', 'name', 'host_id', 'neighbourhood', 'latitude', 'longitude', 'price', 'room_type',
                     'number_of_reviews']
    available_cols = {c.lower(): c for c in df_airbnb_raw.columns}

    price_col = next((c for c in df_airbnb_raw.columns if 'price' in c.lower()), None)
    lat_col = next((c for c in df_airbnb_raw.columns if c.lower() == 'latitude'), None)
    lon_col = next((c for c in df_airbnb_raw.columns if c.lower() == 'longitude'), None)

    keep_cols = [available_cols.get(c.lower()) for c in expected_cols if
                 available_cols.get(c.lower()) in df_airbnb_raw.columns]
    df_airbnb = df_airbnb_raw[keep_cols].copy()
    df_airbnb = df_airbnb.rename(columns={lat_col: 'latitude', lon_col: 'longitude', price_col: 'price'})

    def parse_price(x):
        if pd.isna(x):
            return None
        if isinstance(x, (int, float)):
            return x
        s = str(x).replace('$', '').replace(',', '').strip()
        try:
            return float(s)
        except:
            return None


    df_airbnb['price'] = df_airbnb['price'].apply(parse_price)
    df_airbnb = df_airbnb.dropna(subset=['latitude', 'longitude', 'price'])
    df_airbnb['latitude'] = pd.to_numeric(df_airbnb['latitude'], errors='coerce')
    df_airbnb['longitude'] = pd.to_numeric(df_airbnb['longitude'], errors='coerce')
    df_airbnb = df_airbnb.dropna(subset=['latitude', 'longitude'])

    # Convert to GeoDataFrame (WGS84)
    gdf_airbnb = gpd.GeoDataFrame(
        df_airbnb,
        geometry=gpd.points_from_xy(df_airbnb.longitude, df_airbnb.latitude),
        crs="EPSG:4326"
    )

    # Clean subway data
    df_subway = df_subway_raw.dropna(subset=['latitude', 'longitude'])
    gdf_subway = gpd.GeoDataFrame(
        df_subway,
        geometry=gpd.points_from_xy(df_subway.longitude, df_subway.latitude),
        crs="EPSG:4326"
    )

    # Fetch ACS data
    vars_acs = ['B01003_001E', 'B19013_001E', 'B17001_002E']  # Population, median income, poverty count
    df_acs = get_acs_data(STATE_FIPS, ACS_YEAR, vars_acs, CENSUS_API_KEY)
    gdf_tracts = get_tract_shapefile(STATE_FIPS, TIGER_YEAR)

    acs_rename = {
        'B01003_001E': 'population',
        'B19013_001E': 'median_household_income',
        'B17001_002E': 'poverty_count'
    }
    df_acs = df_acs.rename(columns=acs_rename)
    gdf_tracts = gdf_tracts.merge(
        df_acs[['GEOID', 'population', 'median_household_income', 'poverty_count']],
        on='GEOID',
        how='left'
    )

    gdf_airbnb_m = gdf_airbnb.to_crs(epsg=3857)
    gdf_tracts_m = gdf_tracts.to_crs(epsg=3857)

    gdf_airbnb_with_tract = gpd.sjoin(
        gdf_airbnb_m,
        gdf_tracts_m[['GEOID', 'median_household_income', 'population', 'geometry']],
        how='left',
        predicate='within'
    )

    # Calculate distance to nearest subway station
    gdf_subway_m = gdf_subway.to_crs(epsg=3857)
    from scipy.spatial import cKDTree

    coords_sub = list(zip(gdf_subway_m.geometry.x, gdf_subway_m.geometry.y))
    tree = cKDTree(coords_sub)
    coords_air = list(zip(gdf_airbnb_m.geometry.x, gdf_airbnb_m.geometry.y))
    dists, idxs = tree.query(coords_air, k=1)

    gdf_airbnb_with_tract['dist_to_station_m'] = dists
    gdf_airbnb_with_tract['nearest_station'] = [gdf_subway.iloc[i]['station_name'] for i in idxs]

    # KMeans spatial clustering
    coords = pd.DataFrame({'x': gdf_airbnb_m.geometry.x, 'y': gdf_airbnb_m.geometry.y})
    gdf_airbnb_with_tract['cluster'] = KMeans(n_clusters=5, random_state=42).fit_predict(coords)

    # Regression: Price ~ Distance + Income
    df_model = gdf_airbnb_with_tract.dropna(subset=['price', 'dist_to_station_m', 'median_household_income']).copy()
    df_model['median_household_income'] = pd.to_numeric(df_model['median_household_income'], errors='coerce')
    df_model = df_model.dropna(subset=['median_household_income'])

    df_model['log_price'] = np.log1p(df_model['price'])
    df_model['dist_km'] = df_model['dist_to_station_m'] / 1000

    X = df_model[['dist_km', 'median_household_income']]
    y = df_model['log_price']
    reg = LinearRegression().fit(X, y)
    coef_dist, coef_income = reg.coef_[0], reg.coef_[1]
    intercept = reg.intercept_
    r2 = reg.score(X, y)

    # Visualization
    # Interactive Folium map
    m = folium.Map(location=[40.7128, -74.0060], zoom_start=11, tiles='CartoDB positron')
    cluster_colors = ['blue', 'green', 'purple', 'orange', 'red', 'darkred', 'cadetblue', 'darkgreen', 'beige']

    # Airbnb listings
    sample_size = min(2000, len(gdf_airbnb_with_tract))
    for _, row in gdf_airbnb_with_tract.sample(sample_size).iterrows():
        popup = folium.Popup(
            html=f"<b>{row.get('name', 'No Name')}</b><br>Price: ${row.get('price', 0):.0f}<br>Cluster: {row.get('cluster')}<br>Nearest Station: {row.get('nearest_station')}",
            max_width=250
        )
        folium.CircleMarker(
            location=[row.geometry.y, row.geometry.x],
            radius=2,
            color=cluster_colors[int(row['cluster']) % len(cluster_colors)],
            fill=True,
            fill_opacity=0.7,
            popup=popup
        ).add_to(m)

    sub_layer = folium.FeatureGroup(name='Subway Stations', show=True)
    for _, r in gdf_subway.iterrows():
        folium.CircleMarker(location=[r.latitude, r.longitude], radius=3, color='black', fill=True).add_to(sub_layer)
    m.add_child(sub_layer)

    gdf_tracts_4326 = gdf_tracts_m.to_crs(epsg=4326)
    tracts_geojson = os.path.join(OUT_DIR, 'tracts_income.geojson')
    gdf_tracts_4326[['GEOID', 'median_household_income', 'geometry']].to_file(tracts_geojson, driver='GeoJSON')

    folium.Choropleth(
        geo_data=tracts_geojson,
        name="Median Household Income",
        data=gdf_tracts_4326,
        columns=['GEOID', 'median_household_income'],
        key_on='feature.properties.GEOID',
        fill_opacity=0.6,
        line_opacity=0.1,
        legend_name='Median Income (USD)'
    ).add_to(m)

    folium.LayerControl().add_to(m)
    m.save(os.path.join(OUT_DIR, 'nyc_airbnb_map.html'))

    # Cluster visualization
    plt.figure(figsize=(8, 8))
    sns.scatterplot(
        x=gdf_airbnb_with_tract.geometry.x,
        y=gdf_airbnb_with_tract.geometry.y,
        hue=gdf_airbnb_with_tract['cluster'],
        s=5,
        palette='tab10',
        legend='brief'
    )
    plt.title('Airbnb Spatial Clusters (Projected Coordinates)')
    plt.axis('off')
    plt.savefig(os.path.join(OUT_DIR, 'airbnb_clusters.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # Price vs Subway Distance
    plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica']
    plt.rcParams['axes.unicode_minus'] = False

    price_low = df_model['price'].quantile(0.025)
    price_high = df_model['price'].quantile(0.975)
    df_filtered = df_model[(df_model['price'] >= price_low) & (df_model['price'] <= price_high)].copy()
    median_income_mean = df_filtered['median_household_income'].mean()

    plt.figure(figsize=(6, 4))
    sns.scatterplot(
        x=df_filtered['dist_km'],
        y=df_filtered['price'],
        alpha=0.2,
        s=8,
        color='steelblue',
        label='Listings (outliers removed)'
    )

    xs = np.linspace(df_filtered['dist_km'].min(), df_filtered['dist_km'].max(), 100)
    ys_log = intercept + coef_dist * xs + coef_income * median_income_mean
    ys = np.expm1(ys_log)
    plt.plot(
        xs, ys,
        color='red',
        linewidth=3,
        linestyle='-',
        label='Trend line (enhanced)'
    )

    plt.ylim(df_filtered['price'].min() * 0.9, df_filtered['price'].max() * 1.1)
    plt.xlabel('Distance to Nearest Subway (km)', fontsize=11)
    plt.ylabel('Airbnb Price (USD)', fontsize=11)
    plt.title('Price vs Subway Distance: Closer = Higher?', fontsize=12, pad=15)
    plt.legend(fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    output_path = os.path.join(OUT_DIR, 'price_vs_subway_final.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    # Moran's I spatial autocorrelation
    try:
        import libpysal
        import esda

        tract_price_mean = gdf_airbnb_with_tract.dropna(subset=['price', 'GEOID']).groupby('GEOID')[
            'price'].mean().reset_index()
        gdf_tracts_price = gdf_tracts_m.merge(tract_price_mean, on='GEOID', how='left').dropna(subset=['price'])
        w = libpysal.weights.contiguity.Queen.from_dataframe(gdf_tracts_price)
        mi = esda.Moran(gdf_tracts_price['price'], w)

        with open(os.path.join(OUT_DIR, 'morans_i.txt'), 'w') as f:
            f.write(f"Moran's I: {mi.I:.6f}, p-value: {mi.p_sim:.6f}\n")
    except Exception:
        pass


    out_csv = os.path.join(OUT_DIR, 'airbnb_processed.csv')
    gdf_airbnb_with_tract.drop(columns='geometry').to_csv(out_csv, index=False)

    gdf_tracts_4326[['GEOID', 'median_household_income', 'population', 'poverty_count', 'geometry']].to_file(
        os.path.join(OUT_DIR, 'tracts_income_final.geojson'), driver='GeoJSON'
    )

    summary = {
        'airbnb_records_initial': len(df_airbnb_raw),
        'airbnb_records_cleaned': len(gdf_airbnb_with_tract),
        'subway_stations': len(gdf_subway),
        'tracts_count': len(gdf_tracts),
        'acs_tracts': len(df_acs),
        'regression_R2': r2
    }
    pd.Series(summary).to_csv(os.path.join(OUT_DIR, 'run_summary.csv'))
