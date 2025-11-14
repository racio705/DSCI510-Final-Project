# DSCI510-Final-Project
This project utilizes Airbnb listings in New York City, combining spatial analysis, clustering algorithms, and regression models to investigate the spatial relationships between subway accessibility, neighborhood income levels, and housing prices. It achieves visualization and quantitative analysis of influencing factors.

# Data Source
Airbnb listing data: CSV file downloaded from InsideAirbnb, containing over 4,000 active listings (including prices, geographic coordinates, property types, and guest ratings). 

New York subway station data: CSV file obtained from NY Open Data, containing geographic locations and line information for 473 subway stations. 

Census socioeconomic data: Retrieved via Python code calling the U.S. Census Bureau's ACS API, covering median household income and population density data for 195 New York City neighborhoods.

# Results 
Completed Airbnb listing spatial clustering (K-Means, n=5), made an interactive map was generated using folium and quantified the straight-line distance from listings to the nearest subway station.

# Installation
- You can obtain the Census API key from https://api.census.gov.
- You can download the Airbnb CSV file from https://insideairbnb.com/get-the-data/.
- You can download the subway CSV file from https://data.ny.gov/Transportation/MTA-Subway-Stations/39hk-dx4f/data_preview.

# Running analysis 
From `src/` directory run:
`main.py`
Results will appear in `output/` folder.
