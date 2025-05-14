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
                        index_col=['STATE', 'CITY', 'ZIP', 'ADDRESS'])

############################################################################################

#Get the geocoding output:
cols_to_keep = ['Match_addr', 'Score', 'cb_fips90', 'cb_fips00', 'cb_fips10']
cols_to_match = ['add_state','add_city', 'add_zip', 'add_v2']
geocoding = pd.read_csv(f'{vars._infutor_dir}DerivedData/address_partitions/final_address_CensusBlock_table/AllStates_address_CensusBlock_table.csv',
                        encoding='latin-1',
                        dtype={c:'Int64' for c in ['add_zip', 'cb_fips10']},
                        usecols=cols_to_match+cols_to_keep,
                        index_col=cols_to_match)
geocoding.index.names = ['STATE', 'CITY', 'ZIP', 'ADDRESS']

############################################################################################

#Let's load the 2010 Census Block centroids:
census_blocks_and_centroids = gpd.read_file(f'{vars._census_spatial_dir}raw/blocks.gpkg', layer='2010',
                                            ignore_geometry=True,
                                            usecols=['GEOID10', 'INTPTLAT10', 'INTPTLON10'],
                                            dtype={'GEOID10':'Int64'}|{c:'float' for c in ['INTPTLAT10', 'INTPTLON10']})
census_blocks_and_centroids.columns = ['GEOID', 'lat', 'lon']
census_blocks_and_centroids['GEOID'] = census_blocks_and_centroids['GEOID'].astype('Int64')
census_block_mapping = census_blocks_and_centroids.set_index('GEOID')

#Map Census blocks to centroids:
geocoding_with_centroids = pd.merge(geocoding, census_block_mapping, how='left', left_on='cb_fips10', right_index=True)

############################################################################################

#Merge:
merged_df = pd.merge(addresses, geocoding_with_centroids, how='left', left_index=True, right_index=True).reset_index().set_index('ADDRID')

#Re-index, sort, and save:
merged_df.sort_index(inplace=True)
merged_df.to_csv(f'{vars._infutor_dir_processed}{args.geocoding_subdir}ArcGIS_geocoding.csv')
print(f'Runtime (final): {datetime.now()-start_time}')

############################################################################################