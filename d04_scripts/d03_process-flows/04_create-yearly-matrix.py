# Process INFUTOR individuals into flows:
#####################################################################################################################################
import sys
sys.path.append('../../d03_src/')
import vars

import os
import pandas as pd
import scipy.sparse as ss
from datetime import datetime
from tqdm import tqdm

#####################################################################################################################################

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--base_year", type=int, default=1900)
parser.add_argument("--delta_year", type=int, default=110)
parser.add_argument("--ADDRID_dir", type=str, default='CRD4/OD_pairs/ADDRID/')
args = parser.parse_args()

#####################################################################################################################################

start = datetime.now()
year = args.base_year + args.delta_year
print(f'Year: {year}')

#####################################################################################################################################

#Get the number of addresses:
geocoded_addresses_df = pd.read_csv(f'{vars._infutor_dir_processed}CRD4/addresses/full.csv',
                                    usecols=['ADDRID'], dtype='Int32')
idx_add = geocoded_addresses_df.reset_index()['ADDRID'].astype('Int32')
add_idx = {v: k for k, v in idx_add.items()}
N_addresses = len(idx_add)

#####################################################################################################################################

yearly_chunk_dir = f'{vars._infutor_dir_processed}{args.ADDRID_dir}chunks/yearly/'
all_files = os.listdir(yearly_chunk_dir)
#Collect all chunks in that given year:
yearly_chunks = []
for file in tqdm(all_files):
    if f'{year}-' in file:
        df = pd.read_csv(f'{yearly_chunk_dir}{file}', dtype={'origin':'Int32', 'destination':'Int32', 'flow':'float'})
        yearly_chunks.append(df)

#Concatenate and clean duplicates:
yearly_df_full = pd.concat(yearly_chunks, ignore_index=True).dropna()
yearly_df = yearly_df_full.groupby(['origin', 'destination']).sum().reset_index()
print(f'Aggregation removed {(len(yearly_df_full)-len(yearly_df))/len(yearly_df):.2%} of the entries')

#Map into indices:
yearly_df['origin_idx'] = yearly_df['origin'].astype('Int32').map(add_idx)
yearly_df['destination_idx'] = yearly_df['destination'].astype('Int32').map(add_idx)
assert yearly_df.isna().sum().sum() == 0

#####################################################################################################################################

#Initially do a COO sparse matrix:
data = yearly_df['flow']
indices = (yearly_df['origin_idx'], yearly_df['destination_idx'])
coo_sparse_matrix = ss.coo_matrix((data, indices), shape=(N_addresses, N_addresses))
ss.save_npz(f'{vars._infutor_dir_processed}{args.ADDRID_dir}A_matrices/{year}_coo.npz', coo_sparse_matrix)
runtime = datetime.now()-start
print(f'Converting to COO runtime: {runtime}')

#Convert to CSC sparse matrix:
start = datetime.now()
csc_sparse_matrix = coo_sparse_matrix.tocsc()
ss.save_npz(f'{vars._infutor_dir_processed}{args.ADDRID_dir}A_matrices/{year}_csc.npz', csc_sparse_matrix)
runtime = datetime.now()-start
print(f'Converting to CSC runtime: {runtime}')

#####################################################################################################################################