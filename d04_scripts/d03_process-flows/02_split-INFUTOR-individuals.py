# Split the dataframe of INFUTOR people into smaller chunks
#####################################################################################################################################
import sys
sys.path.append('../../d03_src/')
import utils
import vars

import ast
import numpy as np
import pandas as pd
from tqdm import tqdm, trange

from collections import defaultdict
from datetime import datetime

#####################################################################################################################################

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--N_chunks", type=int, default=1_000)
args = parser.parse_args()

#####################################################################################################################################

#Find the desired number of rows:
chunksize = int(np.ceil(614_949_844/args.N_chunks))

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
INFUTOR_df_in_chunks = pd.read_csv(f'{vars._infutor_dir_processed}CRD4/individuals/full.csv',
                                   usecols=pid_columns+dat_columns+add_columns+eff_columns+cat_columns,
                                   index_col='PID', dtype=cols_dtype, low_memory=False,
                                   chunksize=chunksize)
runtime = datetime.now()-start
print(f'Loading the dataframe in chunks runtime: {runtime}')

#####################################################################################################################################

for idx, df in tqdm(enumerate(INFUTOR_df_in_chunks), total=args.N_chunks):
    #Convert dates:
    for c in dat_columns+eff_columns: df.loc[:,c] = pd.to_datetime(df[c], format='%Y-%m-%d', errors='coerce')
    #Save:
    df.to_csv(f'{vars._infutor_dir_processed}CRD4/individuals/chunks/{idx+1}.csv')
    
#####################################################################################################################################