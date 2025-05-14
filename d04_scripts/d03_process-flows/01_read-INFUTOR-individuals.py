# Create a dataframe of INFUTOR people
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

#Specify dtypes beyond stirng:
cols_dtype = defaultdict(lambda:'string')
pid_columns = ['PID', 'DeceasedCD']
dat_columns = ['IDATE', 'ODATE']
add_columns = ['ADDRID'] + [f'ADDRID{k}' for k in range(2, 11)]
eff_columns = ['EFFDATE'] + [f'EFFDATE{k}' for k in range(2, 11)]
for col in add_columns: cols_dtype[col] = 'Int32'

#Load:
start = datetime.now()
INFUTOR_df = pd.read_csv(f'{vars._infutor_dir_processed}CRD4/full.csv',
                         usecols=pid_columns+dat_columns+add_columns+eff_columns,
                         index_col='PID', dtype=cols_dtype)
runtime = datetime.now()-start
print(f'Loading the dataframe runtime: {runtime}')

#Process ADDRESS columns to include index::
INFUTOR_df.rename({'EFFDATE':'EFFDATE1', 'ADDRID':'ADDRID1'}, axis=1, inplace=True)
add_columns = [f'ADDRID{k}' for k in range(1, 11)]
eff_columns = [f'EFFDATE{k}' for k in range(1, 11)]

#Parse deceased with boolean:
INFUTOR_df.loc[:,'DeceasedCD'] = INFUTOR_df['DeceasedCD'].map({'Y': True}).astype('boolean')

#Convert dates:
for c in dat_columns: INFUTOR_df.loc[:,c] = pd.to_datetime(INFUTOR_df[c], format='%Y%m%d', errors='coerce')
for c in eff_columns: INFUTOR_df.loc[:,c] = pd.to_datetime(INFUTOR_df[c], format='%Y%m', errors='coerce')

#Include address categories:
geocoded_addresses_df = pd.read_csv(f'{vars._infutor_dir_processed}CRD4/addresses/full.csv',
                                    index_col='ADDRID',
                                    usecols=['ADDRID', 'address_type'],
                                    dtype={'ADDRID':'Int32', 'address_type':'string'})
geocoded_addresses_types = geocoded_addresses_df.address_type
for address_idx in range(1, 11):
    INFUTOR_df[f'ADDRCAT{address_idx}'] = INFUTOR_df[f'ADDRID{address_idx}'].map(geocoded_addresses_types)

#Sample:
N = len(INFUTOR_df)
print(f'There are {N:,} unique PIDs. Sampling 10,000,000 ({10_000_000/N:.2%})')
INFUTOR_df.sample(10_000_000).to_csv(f'{vars._infutor_dir_processed}CRD4/individuals/sample.csv')

#Save:
INFUTOR_df.to_csv(f'{vars._infutor_dir_processed}CRD4/individuals/full.csv')