import pandas as pd
import geopandas as gpd
import numpy as np
from datetime import datetime

import sys
sys.path.append('../../d03_src/')
import vars

############################################################################################

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--geocoding_subdir", type=str, default='CRD4/addresses/geocoding_outputs/')
args = parser.parse_args()

############################################################################################

start_time = datetime.now()

############################################################################################

#Get the full addresses:
addresses = pd.read_csv(f'{vars._infutor_dir}CRD4/addresses/full.csv',
                        dtype={c:'Int32' for c in ['ZIP', 'Z4', 'DPC', 'FIPSCD']},
                        index_col=['ADDRID'], low_memory=False)

############################################################################################
#1. INCLUDE RESULTS FROM CENSUS GEOCODER:

#Get the Census geocoding output:
geocoding_census = pd.read_csv(f'{vars._infutor_dir_processed}{args.geocoding_subdir}Census_geocoding.csv',
                               dtype={'CENSUS_BLOCK_2010':'Int64'}, index_col='ADDRID', low_memory=False)

#Merge with the addresses dataframe:
geocoded_addresses_df = pd.merge(addresses, geocoding_census, how='left',
                                 left_index=True, right_index=True).reset_index().set_index('ADDRID')

#Ensure that on not clean addresses we don't have geocoder values:
cols_to_ignore = ['match', 'matchtype', 'parsed', 'lat', 'lon', 'geocoder', 'CENSUS_BLOCK_2010']
geocoded_addresses_df.loc[geocoded_addresses_df.address_type != 'clean', cols_to_ignore] = np.nan

#Sort:
geocoded_addresses_df.sort_index(inplace=True)

#Report:
clean = geocoded_addresses_df.address_type=='clean'
not_attempted = clean&geocoded_addresses_df.match.isna()
print(f"{not_attempted.sum():,} addresses ({not_attempted.sum()/clean.sum():.2%} of the clean addresses) were not attempted.")
match_rate = geocoded_addresses_df.groupby(['address_type'])['match'].mean()['clean']
print(f"On attempted clean addresses, the Census gives a match rate of {match_rate:.2%}.")

############################################################################################
#2. INCLUDE RESULTS FROM ARCGIS GEOCODER:
geocoding_arcGIS = pd.read_csv(f'{vars._infutor_dir_processed}{args.geocoding_subdir}ArcGIS_geocoding.csv',
                               usecols=['ADDRID', 'Match_addr', 'Score', 'lat', 'lon', 'cb_fips10'],
                               dtype={'Score':'float'}, index_col='ADDRID', low_memory=False)
geocoding_arcGIS['cb_fips10'] = pd.to_numeric(geocoding_arcGIS['cb_fips10'], errors='coerce').astype('Int64')

#Find the un-matched, clean addresses which have been matched by ArcGIS:
unmatched = (geocoded_addresses_df.address_type == 'clean')&(geocoded_addresses_df.match == False)
arcGIS_matches = geocoding_arcGIS.loc[unmatched.index[unmatched]].dropna(how='any', axis=0)
print(f'We found matches for {len(arcGIS_matches)/unmatched.sum():.2%} of unmatched addresses on ArcGIS')

#Replace the values in the original dataframe:
geocoded_addresses_df.loc[arcGIS_matches.index, ['match', 'geocoder']] = [True, 'ARC_GIS']
cols_to_replace = ['parsed', 'matchtype', 'lat', 'lon', 'CENSUS_BLOCK_2010']
arcGIS_cols = ['Match_addr', 'Score', 'lat', 'lon', 'cb_fips10']
geocoded_addresses_df.loc[arcGIS_matches.index, cols_to_replace] = arcGIS_matches[arcGIS_cols].rename(dict(zip(arcGIS_cols, cols_to_replace)), axis=1)

#Report:
new_match_rate = geocoded_addresses_df.groupby(['address_type'])['match'].mean()['clean']
print(f'The match rate on clean addresses increased from {match_rate:.2%} to {new_match_rate:.2%} with ArcGIS')

############################################################################################
#3. GEOCODE POSTAL BOXES, RURAL ROUTES, AND REMAINING ADDRESSES WITH ZIP CODES:
# NOTE: THIS INCLUDES THE MOST LIKELY ZIP CODE 

#Select the remaining addresses:
M = geocoded_addresses_df.match.astype('boolean')
all_unmatched = (~M)|(M.isna())
remaining_addresses_df = geocoded_addresses_df.loc[all_unmatched]

#Collect the ZIP code point and BLOCK representation and make sure columns are datetime objects:
ZIP_centroids = {coord: pd.read_csv(f'{vars._census_spatial_dir}crosswalks/HUD_ZIP_monthly_{coord}.csv', dtype={'ZIP':'Int32'}, index_col='ZIP')
                 for coord in ['lat', 'lon', 'CENSUS_BLOCK_2010']}
for coord, df in ZIP_centroids.items():
    df.columns = pd.to_datetime(df.columns, format='%Y-%m', errors='coerce')

#Find the best date for each address (use latest available date in case we don't have effective dates on the address):
dates_available = ZIP_centroids['lat'].columns
best_date = pd.to_datetime(remaining_addresses_df['EFFDATE'], errors='coerce').apply(lambda date: dates_available[dates_available >= date][0] if not pd.isna(date) else dates_available[-1])

#Match the ZIP code to the best centroid:
def match_ZIP_to_point(address):
    lat = ZIP_centroids['lat'].loc[:,address.EFFDATE].get(address.ZIP, default=np.nan)
    lon = ZIP_centroids['lon'].loc[:,address.EFFDATE].get(address.ZIP, default=np.nan)
    blo = ZIP_centroids['CENSUS_BLOCK_2010'].loc[:,address.EFFDATE].get(address.ZIP, default=np.nan)
    return lat, lon, blo
ZIP_and_date = pd.concat([best_date, remaining_addresses_df['ZIP']],axis=1)
coordinates = ZIP_and_date.apply(match_ZIP_to_point, result_type='expand', axis=1)
coordinates.columns=['lat', 'lon', 'CENSUS_BLOCK_2010']

#Include the latitude, longitude, and 2010 CBG in the original dataframe:
geocoded_addresses_df.loc[all_unmatched, ['lat', 'lon', 'CENSUS_BLOCK_2010']] = coordinates

#Update match and geocoder rows:
rows_to_update = all_unmatched&(~geocoded_addresses_df['CENSUS_BLOCK_2010'].isna())
geocoded_addresses_df.loc[rows_to_update, ['match', 'matchtype', 'geocoder']] = [True, 'Tract', 'ZIP_CODE']

#Report:
final_match_rate = geocoded_addresses_df.groupby(['address_type'])['match'].mean()['clean']
print(f'The match rate on clean addresses increased from {new_match_rate:.2%} to {final_match_rate:.2%} with ZIP codes')

############################################################################################
#4. CLASSIFY INTO URBAN OR RURAL:

#Get the urban areas:
urban_areas = gpd.read_file(f'{vars._census_spatial_dir}processed/URBAN_AREA.gpkg', layer='2010')

#Geocode addresses:
address_pts = gpd.points_from_xy(geocoded_addresses_df.lon, geocoded_addresses_df.lat, crs='EPSG:4326')
address_gdf = gpd.GeoDataFrame(geocoded_addresses_df.parsed, geometry=address_pts, crs='EPSG:4326').to_crs(urban_areas.crs)

#Spatial join, and include name of urban areas if available:
join = gpd.sjoin(address_gdf, urban_areas, how='left', predicate='within')
geocoded_addresses_df.loc[:,['URBAN_AREA', 'URBAN_AREA_TYPE']] = join.loc[:,['NAME', 'TYPE']].values

############################################################################################

#Save:
geocoded_addresses_df.to_csv(f'{vars._infutor_dir_processed}CRD4/addresses/full.csv')
geocoded_addresses_df_by_type = geocoded_addresses_df.groupby(['address_type'])
for type, df in geocoded_addresses_df_by_type:
    df.to_csv(f'{vars._infutor_dir_processed}CRD4/addresses/by_type/{type}.csv')