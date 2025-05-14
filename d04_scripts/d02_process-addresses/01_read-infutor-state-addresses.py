############################################################################################
#This script reads files of the form XX.csv, which contain raw address information per PID
# (i.e. in a row, we have a PID and a sequence of addresses associated with the PID) into
# files with unique addresses per state. Each row in the resulting XX.csv file will be a
# unique address, with identifying columns from the INFUTOR data. Additionally, a column
# `LASTDATE` will mark the latest date the addressed appeared on the INFUTOR state file,
# and a column `ADDRCOUNT` will mark the number of times the addressed appeared on the
# INFUTOR state file.
############################################################################################

import pandas as pd
import numpy as np
from datetime import datetime

import sys
sys.path.append('../../d03_src/')
import vars
import process_infutor as infutor

############################################################################################
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--state", type=str)
parser.add_argument("--chunksize", type=int, default=1_000_000)

args = parser.parse_args()

############################################################################################

start_time = datetime.now()
print(f'State: {args.state}')

############################################################################################
file_csv = f'{vars._infutor_dir_processed}CRD4/states/{args.state}.csv'

#Get the column names:
n_cols_per_address = 20 #INFUTOR parameter
columns = infutor.get_CRD4_columns()
columns_addr = [c for c in infutor.get_CRD4_columns()[92:340] if 'INTERNAL' not in c]
columns_addr_first = columns_addr[:n_cols_per_address]

##############################################################################################################

#Initialize a dataframes of addresses:
out_file = f'{vars._infutor_dir}CRD4/addresses/states/{args.state}.csv'
empty_df = pd.DataFrame(columns=columns_addr_first+['COUNT'])
empty_df.to_csv(out_file, index=False)

##############################################################################################################

#Collect the dataframe in chunks:
chunked_df = pd.read_csv(file_csv,
                         dtype='str',
                         usecols=columns_addr,
                         chunksize=args.chunksize)

#Process each chunk:
for chunk in chunked_df:
    address_arr = chunk.values
    assert address_arr.shape[1] == 10*n_cols_per_address
    #Reshape so each address is on one row and drop nans:
    address_df = pd.DataFrame(address_arr.reshape(-1,n_cols_per_address), columns=columns_addr_first).dropna(axis=0, how='all').reset_index(drop=True)
    #Make sure address id is an integer:
    address_df['ADDRID'] = address_df['ADDRID'].astype(str).astype(float)
    address_df = address_df.dropna(subset='ADDRID').reset_index(drop=True)
    address_df['ADDRID'] = address_df['ADDRID'].astype(int)
    #Make sure the date is parseable and sort so that latest date appears first:
    address_df['EFFDATE'] = pd.to_datetime(address_df['EFFDATE'], format='%Y%m', errors='coerce')
    address_df.sort_values(by='EFFDATE', ascending=False, inplace=True)
    #Count occurrences per address:
    address_df['COUNT'] = address_df.groupby('ADDRID')['ADDRID'].transform('count')
    #Keep only unique addresses:
    unique_addresses = address_df.drop_duplicates('ADDRID', keep='first')
    assert unique_addresses.COUNT.sum() == len(address_df)
    #Add addresses:
    unique_addresses.to_csv(out_file, index=False, mode='a', header=False)

print(f'Runtime (final): {datetime.now()-start_time}')

############################################################################################