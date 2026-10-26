# Data sources and provenance

This extension uses the original course project's October 2025 Airbnb snapshot, with 2019–2023 ACS estimates and 2023 Census tract boundaries. It does not substitute a current Airbnb snapshot for the historical sample. The MTA input is the original local CSV; its original download date has not been established.

## Input register

| Input | Period or version | Publisher and reproducible location | Local cache |
| --- | --- | --- | --- |
| Detailed NYC Airbnb listings | Archive directory `2025-10-01`; local data has scrape ID `20251001171547` and `last_scraped` dates spanning October 1–2, 2025 | [Inside Airbnb historical listings](https://data.insideairbnb.com/united-states/ny/new-york-city/2025-10-01/data/listings.csv.gz) | `data/raw/listings.csv.gz` |
| Subway stations | Original local `MTA_Subway_Stations.csv`; download date unknown | [MTA Subway Stations, dataset 39hk-dx4f](https://data.ny.gov/Transportation/MTA-Subway-Stations/39hk-dx4f) | `data/raw/subway.csv` |
| Median household income and margin of error | ACS 5-year release ending 2023, covering 2019–2023; income in 2023 inflation-adjusted USD | [Official B19013 table file](https://www2.census.gov/programs-surveys/acs/summary_file/2023/table-based-SF/data/5YRData/acsdt5y2023-b19013.dat) | `data/raw/acs_income_2023.dat` |
| Population and margin of error | ACS 2019–2023 5-year estimates, table B01003 | [Official B01003 table file](https://www2.census.gov/programs-surveys/acs/summary_file/2023/table-based-SF/data/5YRData/acsdt5y2023-b01003.dat) | `data/raw/acs_population_2023.dat` |
| Census tract polygons and land area | New York State, TIGER/Line 2023 | [Official tract ZIP](https://www2.census.gov/geo/tiger/TIGER2023/TRACT/tl_2023_36_tract.zip) | `data/raw/tl_2023_36_tract.zip` |

The historical Airbnb archive and the three Census files returned HTTP 200 to HEAD checks on October 25, 2026, America/Los_Angeles. Reported lengths were 17,748,368 bytes for listings, 17,884,943 for B19013, 18,311,997 for B01003, and 8,854,208 for tract geometry. Availability can change. These response lengths establish the checked resources, not a substitute for content hashes.

The historical listing cache is retained from the user's original project. A filename, filesystem timestamp, or matching byte length alone does not establish byte identity with a later download. Keep the generated `data/raw/manifest.json` with the run: its SHA256 hashes identify the actual local inputs used. Preserve the MTA hash even when refreshing other inputs. Replacing it with today's portal export changes the input version and requires a new manifest and rerun.

## Geographic joins and ACS values

Use the five New York City county FIPS codes: Bronx `005`, Kings `047`, New York `061`, Queens `081`, and Richmond `085`, in New York State `36`. The ACS bulk tables are pipe-delimited. Tract identifiers have the form `1400000US` followed by an 11-character GEOID. Retain identifiers as strings, including leading zeros.

The bulk field names differ from API field names:

| Bulk field | API equivalent | Meaning |
| --- | --- | --- |
| `B19013_E001` | `B19013_001E` | Median household income estimate |
| `B19013_M001` | `B19013_001M` | Income margin of error |
| `B01003_E001` | `B01003_001E` | Total population estimate |
| `B01003_M001` | `B01003_001M` | Population margin of error |

Negative Census annotation codes are not numeric income or population measurements. Convert them to missing values before modeling. Open-ended income medians can be capped, and some margins of error are unavailable. Keep these limitations visible when interpreting the income coefficient. The [Census annotation guide](https://www.census.gov/data/developers/data-sets/acs-1year/notes-on-acs-estimate-and-annotation-values.html) explains the codes. Population density uses Census `ALAND` land area, rather than population alone or water-inclusive polygon area.

The key-free bulk route avoids a credential dependency. During the source checks, Census API data requests without a key returned an HTML “Missing Key” page even though the HTTP status was 200. Treat a successful status as insufficient: validate the response format and required columns. The [Census table-based summary file documentation](https://www.census.gov/programs-surveys/acs/data/summary-file.2023.html) describes the bulk release.

## What the sample measures

- `price` is a listing's advertised nightly price in this snapshot, not a completed transaction, revenue, housing sale price, or achieved long-term rent. Missing prices are a selection issue; report the raw and usable sample sizes separately.
- The primary population excludes `Hotel room` records and property types matching `hotel|hostel|resort`, case-insensitive. The full catalog is retained as a separate sensitivity analysis. Suspected extreme hospitality price anomalies are not corrected into invented values. Recorded property taxonomy does not verify legal classification, and the primary sample does not use price-tail trimming.
- The archive date and listing scrape timestamps describe collection timing. ACS income and population describe 2019–2023, not conditions measured in October 2025. They are lagged neighborhood context. The MTA file's unknown download date adds a temporal alignment limitation.
- Inside Airbnb reports that listing coordinates are anonymized by up to approximately 150 m. Nearest-station distances and tract membership can therefore be uncertain near thresholds or boundaries. Straight-line distance to a station coordinate does not measure walking distance, entrance access, travel time, service frequency, or accessibility for a specific traveler. See [Inside Airbnb data assumptions](https://insideairbnb.com/data-assumptions/).
- The MTA source distinguishes stations and station complexes and can contain multiple records for a station. Count and deduplicate the actual local file rather than copying a current portal count. If Staten Island Railway records are excluded, call the feature subway distance; otherwise label the broader rail definition explicitly.
- Listings within the same neighborhood, host portfolio, or building may share characteristics. Spatial validation estimates transfer to held-out areas under its stated partition; it does not guarantee host-independent or future-period performance.
- An adjusted subway coefficient is an observational association conditional on the included controls. It does not establish that building a station changes Airbnb prices. Room size, amenities, location, regulation, and unobserved listing quality can confound the relationship.
- Availability calendars do not distinguish booked nights from host-blocked nights. This project does not infer observed occupancy or profitability from unavailability.

## Attribution, retention, and reuse

Attribute Airbnb inputs to Inside Airbnb and link the exact snapshot. Its [download page](https://insideairbnb.com/get-the-data/) states Creative Commons Attribution 4.0, and its [data policies](https://insideairbnb.com/data-policies/) request downloading only what is needed, caching downloads, and avoiding republication of raw data. Raw listing data is kept in the ignored local cache; the public project can share code, aggregate results, and source references.

Attribute station data to the Metropolitan Transportation Authority / New York State Open Data and ACS/TIGER data to the U.S. Census Bureau. The repository's MIT license covers its code; it does not replace the publishers' terms for third-party inputs. Census methodology and release notes are available in the [2023 release documentation](https://www.census.gov/programs-surveys/acs/news/data-releases/2023/release.html).

Inside Airbnb normally exposes a rolling window of free downloads. A historical URL that works now may later require an [archived data request](https://insideairbnb.com/data-policies/). Retain the original local cache and hashes for exact reruns. If that cache is unavailable and the archive expires, document the access gap; do not silently replace the historical snapshot and reuse the old metrics.
