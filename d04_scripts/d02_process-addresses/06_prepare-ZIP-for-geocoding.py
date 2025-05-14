import pandas as pd
import geopandas as gpd
import numpy as np
from datetime import datetime
import scipy.sparse as ss
from collections import defaultdict
from tqdm import tqdm, trange

import sys
sys.path.append('../../d03_src/')
import vars

####################################################################################################################################

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--n_chunks", type=int, default=1_000)
parser.add_argument("--ZIP_pairs_dir", type=str, default='CRD4/addresses/ZIP_date_pairs/')
args = parser.parse_args()

####################################################################################################################################

start_time = datetime.now()

####################################################################################################################################
#1. LOAD THE FRACTIONAL ZIP CODE CROSSWALK:
start = datetime.now()
ZIP_to_BLOCK_crosswalk = pd.read_csv(f'{vars._census_spatial_dir}crosswalks/HUD_ZIP_BLOCK.csv',
                                     usecols=['ZIP', 'CENSUS_BLOCK_2010', 'ZIP_BLOCK_FRAC', 'DATE'],
                                     dtype={'ZIP':'Int32', 'CENSUS_BLOCK_2010':'Int64',
                                            'ZIP_BLOCK_FRAC':'float', 'DATE':'string'})
ZIP_to_BLOCK_crosswalk['DATE'] = pd.to_datetime(ZIP_to_BLOCK_crosswalk['DATE'],
                                                errors='coerce', format='%Y-%m-%d')
print(f'Loading the ZIP to Block crosswalk runtime: {datetime.now()-start}')

####################################################################################################################################
#2. LOAD THE DATAFRAME:
cols_dtype = defaultdict(lambda:'string')
for col in ['ADDRID', 'ZIP']: cols_dtype[col] = 'Int32'
for col in ['CENSUS_BLOCK_2010']: cols_dtype[col] = 'Int64'
start = datetime.now()
geocoded_addresses_df = pd.read_csv(f'{vars._infutor_dir_processed}CRD4/addresses/full.csv',
                                    index_col='ADDRID', usecols=['ADDRID', 'geocoder', 'ZIP', 'CENSUS_BLOCK_2010', 'EFFDATE'],
                                    dtype=cols_dtype)
mask_ZIP = geocoded_addresses_df['geocoder'].isin(['ZIP_CODE'])
ZIP_geocodings = geocoded_addresses_df[mask_ZIP][['ZIP', 'EFFDATE']].reset_index()
runtime = datetime.now()-start
print(f'Loading the dataframe runtime: {runtime}')

####################################################################################################################################
#3. PROCESSING ZIP VALUES:
start = datetime.now()

#First, reduce the dates to the crosswalk span:
max_date = pd.Timestamp('2021-12-01')
min_date = pd.Timestamp('2012-01-01')
ZIP_geocodings['CROSSWALKDATE'] = pd.to_datetime(ZIP_geocodings['EFFDATE'], errors='coerce').clip(min_date, max_date)

#Impute the most recent date:
ZIP_geocodings['CROSSWALKDATE'] = ZIP_geocodings['CROSSWALKDATE'].fillna(max_date)

#We can restrict to quarters:
ZIP_geocodings['CROSSWALKDATE_qt'] = ZIP_geocodings['CROSSWALKDATE'].apply(lambda x: x.replace(month=((x.month - 1) // 3) * 3 + 1))
print(f'Processing the ZIP geocodings: {datetime.now()-start}')

####################################################################################################################################
#4. SAVING:
start = datetime.now()

#Group, sort, and slice:
ZIP_geocodings_sort_by_ZIP = ZIP_geocodings.sort_values(['ZIP', 'CROSSWALKDATE']).reset_index(drop=True)
no_duplicates = ZIP_geocodings_sort_by_ZIP.loc[:,['ZIP', 'CROSSWALKDATE_qt']].drop_duplicates().reset_index(drop=True)

#Iterate:
n_rows_per_chunk = int(len(no_duplicates)//args.n_chunks + 1)
for k in trange(args.n_chunks):
    df = no_duplicates.iloc[k*n_rows_per_chunk:(k+1)*n_rows_per_chunk]
    df.to_csv(f'{vars._infutor_dir_processed}{args.ZIP_pairs_dir}{k+1}.csv', index=False)
    #Find unique values in the dataframe and crop teh crosswalk:
    ZIPs = df['ZIP'].unique()
    ZIPs_crosswalk = ZIP_to_BLOCK_crosswalk.loc[ZIP_to_BLOCK_crosswalk['ZIP'].isin(ZIPs)]
    ZIPs_crosswalk.to_csv(f'{vars._infutor_dir_processed}{args.ZIP_pairs_dir}crosswalks/{k+1}.csv', index=False)

print(f'Saving the ZIP geocodings in chunks: {datetime.now()-start}')

####################################################################################################################################