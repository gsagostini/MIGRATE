# Process INFUTOR individuals into flows:
#####################################################################################################################################
import sys
sys.path.append('../../d03_src/')
import vars
import process_infutor as INFUTOR

import numpy as np
import pandas as pd

from collections import defaultdict
from datetime import datetime
from tqdm import tqdm

#####################################################################################################################################

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--idx", type=int, default=1)
parser.add_argument("--output_chunks_dir", type=str, default='CRD4/OD_pairs/ADDRID/chunks/')
args = parser.parse_args()

#####################################################################################################################################

#Specify dtypes beyond string:
cols_dtype = defaultdict(lambda:'string')
pid_columns = ['PID', 'DeceasedCD']
dat_columns = ['IDATE', 'ODATE']
add_columns = [f'ADDRID{k}' for k in range(1, 11)]
eff_columns = [f'EFFDATE{k}' for k in range(1, 11)]
cat_columns = [f'ADDRCAT{k}' for k in range(1, 11)]
for col in add_columns: cols_dtype[col] = 'Int32'
cols_dtype['DeceasedCD'] = 'boolean'

#Load:
start = datetime.now()
INFUTOR_df = pd.read_csv(f'{vars._infutor_dir_processed}CRD4/individuals/chunks/{args.idx}.csv',
                         usecols=pid_columns+dat_columns+add_columns+eff_columns+cat_columns,
                         index_col='PID', dtype=cols_dtype,
                         parse_dates=dat_columns+eff_columns, date_format='%Y-%m-%d')
#Enforce date types:
for c in dat_columns+eff_columns: INFUTOR_df.loc[:,c] = pd.to_datetime(INFUTOR_df[c], format='%Y-%m-%d', errors='coerce')

runtime = datetime.now()-start
print(f'Loading the dataframe runtime: {runtime}')

#####################################################################################################################################

#Apply:
start = datetime.now()
individual_responses = INFUTOR_df.apply(INFUTOR.process_individual, axis=1)
individual_responses.rename('RESPONSES').to_csv(f'{vars._infutor_dir_processed}{args.output_chunks_dir}individual/{args.idx}.csv')
n_zero = (individual_responses.apply(len) == 0).sum()
print(f' -Number of discarded PIDs: {n_zero:,} ({n_zero/len(INFUTOR_df):.2%})')
runtime = datetime.now()-start
print(f'Collecting individual responses runtime: {runtime}')

#####################################################################################################################################

#Aggregate:
start = datetime.now()
aggregated_responses = INFUTOR.aggregate_individual_responses(individual_responses, verbose=True)
aggregated_responses.to_csv(f'{vars._infutor_dir_processed}{args.output_chunks_dir}{args.idx}.csv', index=False)
runtime = datetime.now()-start
print(f'Aggregating responses runtime: {runtime}')

#####################################################################################################################################

#Group by year:
start = datetime.now()
for year, yearly_OD_df in tqdm(aggregated_responses.groupby('year')):
    #Drop any missing values (and report if there are any!)
    nonan_yearly_OD_df = yearly_OD_df.dropna().loc[:,['origin', 'destination', 'flow']]
    if len(nonan_yearly_OD_df) != len(yearly_OD_df):
        print(f'{year:.0f}: {len(yearly_OD_df)-len(nonan_yearly_OD_df):.0f} nan values')
    #Aggregate:
    clean_yearly_OD_df = nonan_yearly_OD_df.groupby(['origin', 'destination']).sum().reset_index()
    #Save:
    clean_yearly_OD_df.to_csv(f'{vars._infutor_dir_processed}{args.output_chunks_dir}yearly/{year}-{args.idx}.csv', index=False)
runtime = datetime.now()-start
print(f'Grouping by year runtime: {runtime}')

#####################################################################################################################################