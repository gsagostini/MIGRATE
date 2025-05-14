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

####################################################################################################################################

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--ZIP_chunk_idx", type=int)
parser.add_argument("--geography", type=str, default='block')
parser.add_argument("--ZIP_pairs_dir", type=str, default='CRD4/addresses/ZIP_date_pairs/')
args = parser.parse_args()

####################################################################################################################################

start_time = datetime.now()

####################################################################################################################################
#1. LOAD THE FRACTIONAL ZIP CODE CROSSWALK:
start = datetime.now()
ZIP_to_BLOCK_crosswalk = pd.read_csv(f'{vars._infutor_dir_processed}{args.ZIP_pairs_dir}crosswalks/{args.ZIP_chunk_idx}.csv',
                                     usecols=['ZIP', 'CENSUS_BLOCK_2010', 'ZIP_BLOCK_FRAC', 'DATE'],
                                     dtype={'ZIP':'Int32', 'CENSUS_BLOCK_2010':'Int64',
                                            'ZIP_BLOCK_FRAC':'float', 'DATE':'string'})
ZIP_to_BLOCK_crosswalk['DATE'] = pd.to_datetime(ZIP_to_BLOCK_crosswalk['DATE'],
                                                errors='coerce', format='%Y-%m-%d')
print(f'Loading the ZIP to Block crosswalk runtime: {datetime.now()-start}')

####################################################################################################################################
#2. LOAD THE PAIRS:
start = datetime.now()
unique_ZIPxDATE = pd.read_csv(f'{vars._infutor_dir_processed}{args.ZIP_pairs_dir}{args.ZIP_chunk_idx}.csv',
                              dtype={'ZIP':'Int32', 'CROSSWALKDATE_qt':'string'})
unique_ZIPxDATE['DATE'] = pd.to_datetime(unique_ZIPxDATE['CROSSWALKDATE_qt'], errors='coerce')
print(f'Loading the ZIPxDATE pairs runtime: {datetime.now()-start}')

####################################################################################################################################
#3. GEOCODE:
start = datetime.now()

#Merge with crosswalk:
merged_crosswalk = unique_ZIPxDATE.merge(ZIP_to_BLOCK_crosswalk, on='ZIP',
                                         suffixes=('_address', '_crosswalk'), how='inner')

#Calculate the absolute difference in dates and find the minimum date difference:
merged_crosswalk['DATE_diff'] = abs(merged_crosswalk['DATE_address'] - merged_crosswalk['DATE_crosswalk'])
minimum_date_difference = merged_crosswalk.groupby(['ZIP', 'DATE_address'])['DATE_diff'].transform('min')

#Select rows where the minimum discrepancy is attained, ensuring we keep only one crosswalk date:
crosswalk_only_min = merged_crosswalk.loc[merged_crosswalk['DATE_diff'] == minimum_date_difference]
best_date = crosswalk_only_min.groupby(['ZIP', 'DATE_address'])['DATE_crosswalk'].transform('min')
crosswalk_only_best = crosswalk_only_min.loc[crosswalk_only_min['DATE_crosswalk'] == best_date]

#Aggregate the results into lists for each group:
crosswalk_only_best_per_address = crosswalk_only_best.groupby(['ZIP', 'DATE_address'])
result = crosswalk_only_best_per_address[['CENSUS_BLOCK_2010', 'ZIP_BLOCK_FRAC']].agg(list).reset_index()
results_expanded = result.explode(['CENSUS_BLOCK_2010', 'ZIP_BLOCK_FRAC'], ignore_index=True)

print(f'Geocoding ZIPxDATE pairs runtime: {datetime.now()-start}')

####################################################################################################################################
#4. MAP CENSUS BLOCKS:
start = datetime.now()
cbs_idx = prc.get_geography_indices('block')
results_expanded['CB_idx'] = results_expanded['CENSUS_BLOCK_2010'].astype('Int64').map(cbs_idx)
print(f'Mapping Census Blocks runtime: {datetime.now()-start}')

####################################################################################################################################
#5. MAP INTO DESIRED GEOGRAPHY:
if args.geography != 'block':
    start = datetime.now()
    C_map = prc.get_geography_matrices('block', args.geography, as_dict=True)
    results_expanded['geography_idx'] = results_expanded['CB_idx'].map(C_map)
    ZIP_to_geography_expanded = results_expanded.loc[:, ['ZIP', 'DATE_address', 'geography_idx', 'ZIP_BLOCK_FRAC']]
    ZIP_to_geography = ZIP_to_geography_expanded.groupby(['ZIP', 'DATE_address', 'geography_idx']).sum().reset_index()
    print(f'Mapping Census Blocks into {args.geography} runtime: {datetime.now()-start}')
else: 
    results_expanded['geography_idx'] = results_expanded['CB_idx']
    ZIP_to_geography = results_expanded.loc[:, ['ZIP', 'DATE_address', 'geography_idx', 'ZIP_BLOCK_FRAC']]

####################################################################################################################################

ZIP_to_geography.to_csv(f'{vars._infutor_dir_processed}{args.ZIP_pairs_dir}geocoded/{args.geography}/{args.ZIP_chunk_idx}.csv', index=False)

####################################################################################################################################