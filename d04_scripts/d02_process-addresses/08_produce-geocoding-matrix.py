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
import process_census as prc

#####################################################################################################################################

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--exact_coordinates_saved", default=False, action=argparse.BooleanOptionalAction)
parser.add_argument("--ZIP_coordinates_saved", default=False, action=argparse.BooleanOptionalAction)
parser.add_argument("--n_ZIP_chunks", default=1_000, type=int)
parser.add_argument("--n_chunks", type=int, default=100)
parser.add_argument("--ZIP_pairs_dir", type=str, default='CRD4/addresses/ZIP_date_pairs/')
parser.add_argument("--geocoding_dir", type=str, default='CRD4/addresses/geocoding_matrix/')
parser.add_argument("--geography", type=str, default='block')
args = parser.parse_args()

#####################################################################################################################################

start_time = datetime.now()

############################################################################################
#1. LOAD THE GEOCODING DATAFRAME AND INDICES:

#Specify dtypes beyond string:
cols_dtype = defaultdict(lambda:'string')
for col in ['ADDRID', 'ZIP']: cols_dtype[col] = 'Int32'
for col in ['CENSUS_BLOCK_2010']: cols_dtype[col] = 'Int64'

#Load:
start = datetime.now()
geocoded_addresses_df = pd.read_csv(f'{vars._infutor_dir_processed}CRD4/addresses/full.csv',
                                    index_col='ADDRID', usecols=['ADDRID', 'geocoder', 'ZIP', 'CENSUS_BLOCK_2010', 'EFFDATE'],
                                    dtype=cols_dtype)
runtime = datetime.now()-start
print(f'Loading the dataframe runtime: {runtime}')

#Turn into indices
cbs_idx = prc.get_geography_indices('block')
geography_idx = prc.get_geography_indices(args.geography)
idx_add = geocoded_addresses_df.reset_index()['ADDRID'].astype('Int32')
add_idx = {v: k for k, v in idx_add.items()}

############################################################################################
#3. LIST THE COORDINATES OF THE PRECISELY GEOCODED ITEMS:
start = datetime.now()
exact_filename = f'{vars._infutor_dir_processed}{args.geocoding_dir}coords_for_sparse_matrix_{args.geography}_exact.csv'
if args.exact_coordinates_saved:
    exact_coordinates_df = pd.read_csv(exact_filename, dtype={'AD_idx':'Int32', 'geography_idx':'Int32', 'p':'float'})
else:
    exact_coordinates_df = pd.DataFrame([], columns=['AD_idx', 'CB_idx', 'geography_idx', 'p'])
    #The exactly-geocoded addresses can be resolved as a binary mapping:
    mask_exact = geocoded_addresses_df['geocoder'].isin(['CENSUS', 'ARC_GIS'])
    exact_geocodings = geocoded_addresses_df[mask_exact]['CENSUS_BLOCK_2010'].reset_index()
    #Populate:
    exact_coordinates_df['AD_idx'] = exact_geocodings['ADDRID'].map(add_idx)
    exact_coordinates_df['CB_idx'] = exact_geocodings['CENSUS_BLOCK_2010'].map(cbs_idx)
    exact_coordinates_df['p'] = 1
    #Convert Census Block to whatever geography:
    if args.geography != 'block':
        C_map = prc.get_geography_matrices('block', args.geography, as_dict=True)
        exact_coordinates_df['geography_idx'] = exact_coordinates_df['CB_idx'].map(C_map)
        exact_coordinates_df = exact_coordinates_df.loc[:,['AD_idx', 'geography_idx', 'p']].groupby(['AD_idx', 'geography_idx']).sum().reset_index()
    else: 
        exact_coordinates_df['geography_idx'] = exact_coordinates_df['CB_idx']
    #Save:
    exact_coordinates_df = exact_coordinates_df.loc[:,['AD_idx', 'geography_idx', 'p']].dropna()
    exact_coordinates_df.to_csv(exact_filename, index=False)

runtime = datetime.now()-start
print(f'Creating exact matches coordinates runtime: {runtime}')

############################################################################################
#4. LIST THE COORDINATES OF THE ZIP GEOCODED ITEMS:
start = datetime.now()
ZIP_filename = f'{vars._infutor_dir_processed}{args.geocoding_dir}coords_for_sparse_matrix_{args.geography}_ZIP.csv'
if args.ZIP_coordinates_saved:
    ZIP_coordinates_df = pd.read_csv(ZIP_filename, dtype={'AD_idx':'Int32', 'geography_idx':'Int32', 'p':'float'})
else:
    #Collect the ZIP geocodings into a single dataframe:
    ZIP_dfs = []
    for k in trange(args.n_ZIP_chunks):
        result = pd.read_csv(f'{vars._infutor_dir_processed}{args.ZIP_pairs_dir}geocoded/{args.geography}/{k+1}.csv',
                             usecols=['ZIP', 'DATE_address', 'ZIP_BLOCK_FRAC', 'geography_idx'],
                             dtype={'ZIP':'Int32', 'DATE_address':'string', 'ZIP_BLOCK_FRAC':'float', 'geography_idx':'Int32'})
        ZIP_dfs.append(result)
    ZIP_df = pd.concat(ZIP_dfs, ignore_index=True)
    ZIP_df['DATE_address'] = pd.to_datetime(ZIP_df['DATE_address'], errors='coerce')
    ZIP_df = ZIP_df.rename({'DATE_address':'DATE', 'ZIP_BLOCK_FRAC':'p'}, axis=1)
    #To merge with the ZIP-geocoded items, first, we need to get the rows:
    mask_ZIP = geocoded_addresses_df['geocoder'].isin(['ZIP_CODE'])
    ZIP_geocodings = geocoded_addresses_df.loc[mask_ZIP, ['ZIP', 'EFFDATE']].reset_index()
    #Find the quarter date to merge on:
    max_date = pd.Timestamp('2021-12-01')
    min_date = pd.Timestamp('2012-01-01')
    ZIP_geocodings['CROSSWALKDATE'] = pd.to_datetime(ZIP_geocodings['EFFDATE'], errors='coerce').clip(min_date, max_date)
    ZIP_geocodings['CROSSWALKDATE'] = ZIP_geocodings['CROSSWALKDATE'].fillna(max_date)
    ZIP_geocodings['CROSSWALKDATE_qt'] = ZIP_geocodings['CROSSWALKDATE'].apply(lambda x: x.replace(month=((x.month-1)//3)*3+1))
    ZIP_geocodings['AD_idx'] = ZIP_geocodings['ADDRID'].astype('Int32').map(add_idx)
    ADDRID_df = ZIP_geocodings.loc[:,['AD_idx', 'ZIP', 'CROSSWALKDATE_qt']].rename(columns={'CROSSWALKDATE_qt':'DATE'})
    #Merge in pieces (to avoid running in memory errors:
    n_rows_per_chunk = int(len(ADDRID_df)//args.n_chunks + 1)
    ZIP_coordinates_df = pd.DataFrame([], columns=['AD_idx', 'geography_idx', 'p'])
    ZIP_coordinates_df.to_csv(ZIP_filename, index=False)
    print(f'...processing ZIP coordinates in {args.n_chunks:,} chunks of {n_rows_per_chunk:,} address IDs each')
    for k in trange(args.n_chunks):
        df = ADDRID_df.iloc[k*n_rows_per_chunk:(k+1)*n_rows_per_chunk]
        merged_df = df.merge(ZIP_df, how='left', on=['ZIP','DATE'])
        #Log the values and save:
        ZIP_coordinates_df_k = merged_df.dropna().loc[:,['AD_idx', 'geography_idx', 'p']]
        ZIP_coordinates_df_k.to_csv(ZIP_filename, index=False, header=False, mode='a')
    #Read everything:
    ZIP_coordinates_df = pd.read_csv(ZIP_filename, dtype={'AD_idx':'Int32', 'geography_idx':'Int32', 'p':'float'})
        
runtime = datetime.now()-start
print(f'Creating ZIP matches coordinates runtime: {runtime}')

############################################################################################
#5. CONVERT AND SAVE:
coordinates_df = pd.concat([exact_coordinates_df, ZIP_coordinates_df])
coordinates_clean_df = coordinates_df.dropna(how='any', axis=0).reset_index(drop=True)
coordinates_clean_df.to_csv(f'{vars._infutor_dir_processed}{args.geocoding_dir}coords_for_sparse_matrix_{args.geography}.csv', index=False)

#Initially do a COO sparse matrix:
start = datetime.now()
data = coordinates_clean_df['p']
indices = (coordinates_clean_df['AD_idx'], coordinates_clean_df['geography_idx'])
coo_sparse_matrix = ss.coo_matrix((data, indices), shape=(len(idx_add), len(geography_idx)))
ss.save_npz(f'{vars._infutor_dir_processed}{args.geocoding_dir}geocoding_{args.geography}_coo.npz', coo_sparse_matrix)
runtime = datetime.now()-start
print(f'Converting to COO runtime: {runtime}')

#Convert to CSC sparse matrix:
start = datetime.now()
csc_sparse_matrix = coo_sparse_matrix.tocsc()
ss.save_npz(f'{vars._infutor_dir_processed}{args.geocoding_dir}geocoding_{args.geography}_csc.npz', csc_sparse_matrix)
runtime = datetime.now()-start
print(f'Converting to CSR runtime: {runtime}')

############################################################################################