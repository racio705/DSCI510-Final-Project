from dotenv import load_dotenv
import os

load_dotenv()

CENSUS_API_KEY = os.getenv("CENSUS_API_KEY")
AIRBNB_CSV_PATH = os.getenv("AIRBNB_CSV_PATH")
SUBWAY_CSV_PATH = os.getenv("SUBWAY_CSV_PATH")

OUT_DIR = "output"
ACS_YEAR = "2023"
TIGER_YEAR = "2023"
STATE_FIPS = "36"


# Verify API Data Source Availability
def test_census_api():
    import requests
    try:
        url = f"https://api.census.gov/data/{ACS_YEAR}/acs/acs5?get=NAME,B01003_001E&for=tract:010100&in=state:{STATE_FIPS}&in=county:061&key={CENSUS_API_KEY}"
        res = requests.get(url)
        res.raise_for_status()
        data = res.json()
        if len(data) > 1 and data[0] == ['NAME', 'B01003_001E', 'state', 'county', 'tract']:
            print("API testing passed")
        else:
            print("API response data format is abnormal")
    except Exception as e:
        print(f"API test failed: {str(e)[:50]}")


if __name__ == "__main__":
    test_census_api()