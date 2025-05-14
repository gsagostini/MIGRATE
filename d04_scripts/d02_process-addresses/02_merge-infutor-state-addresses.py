############################################################################################
#This script reads state addresses files XX.csv, which contain information on unqiue addresses
# into a single file of US-wide addresses. The addresses are classified according to their 
# nature (PO boxes, rural routes, territories, incomplete, or clean).
############################################################################################

import pandas as pd
import numpy as np
from datetime import datetime

import sys
sys.path.append('../../d03_src/')
import vars
import process_infutor as infutor

from tqdm import tqdm

############################################################################################

start_time = datetime.now()

############################################################################################

#Get the column names:
n_cols_per_address = 20 #INFUTOR parameter
columns = infutor.get_CRD4_columns()
columns_addr = [c for c in infutor.get_CRD4_columns()[92:340] if 'INTERNAL' not in c]
columns_addr_first = columns_addr[:n_cols_per_address]

############################################################################################

#Start empty dataframe:
out_file = f'{vars._infutor_dir}CRD4/addresses/full.csv'
empty_df = pd.DataFrame(columns=[c if c !='Z4TYPE\n' else 'Z4TYPE' for c in columns_addr_first]+['COUNT'])
empty_df.to_csv(out_file, index=False)

############################################################################################
#Collect all the states:
with open(vars._states_filepath, 'r') as file:
    states = file.read().splitlines()
    
for state in tqdm(states):
    #Load the file:
    address_df = pd.read_csv(f'{vars._infutor_dir}CRD4/addresses/states/{state}.csv')
    #Process some entries:
    address_df = address_df.rename({'Z4TYPE\n':'Z4TYPE'}, axis=1)
    for col in ['ZIP', 'Z4', 'DPC', 'FIPSCD']:
        address_df[col] = address_df[col].astype('Int32')
    address_df['COUNT'] = address_df.groupby('ADDRID')['COUNT'].transform('sum')
    #Drop duplicates:
    address_df.sort_values(by='EFFDATE', ascending=False, inplace=True)
    address_df = address_df.drop_duplicates('ADDRID', keep='first').reset_index(drop=True)
    #Include into main file:
    address_df.to_csv(out_file, index=False, mode='a', header=False)

print(f'Runtime (create file): {datetime.now()-start_time}')

############################################################################################
#Drop duplicate address IDs:
raw_addresses = pd.read_csv(out_file, dtype={c:'Int32' for c in ['ZIP', 'Z4', 'DPC', 'FIPSCD']})
raw_addresses.sort_values(by=['ADDRID', 'EFFDATE'], ascending=[True, False], inplace=True)
addresses = raw_addresses.drop_duplicates('ADDRID').set_index('ADDRID')

print(f'The total number of unique addresses is {len(addresses):,}')
print(f'Runtime (drop duplicates on file): {datetime.now()-start_time}')

############################################################################################
#Classify addresses:
address_lines = addresses['ADDRESS'].astype(str).str

#Select territory, incomplete, rural, and PO box addresses:
mask_territr = ~addresses['STATE'].isin(vars._abbreviations)
mask_missing = addresses['ADDRESS'].isna()
mask_rroutes = address_lines.contains(' RR ')|address_lines.startswith('RR ')|address_lines.contains(' HC ')|address_lines.startswith('HC ')
mask_poboxes = address_lines.contains(' BOX ')|address_lines.startswith(' BOX ')|address_lines.endswith(' BOX')

#Include categorization
conditions = [(mask_territr),
              (~mask_territr)&(mask_missing),
              (~mask_territr)&(~mask_missing)&(mask_rroutes),
              (~mask_territr)&(~mask_missing)&(~mask_rroutes)&(mask_poboxes)]
values = ['territory', 'incomplete', 'rural_route', 'PO_box']
addresses['address_type'] = np.select(conditions, values, default='clean')

#Report:
address_counts = addresses.address_type.value_counts(normalize=True).to_dict()
for address_type, proportion in address_counts.items():
    print(f"Addresses classified as {address_type.replace('_', ' ')}: {proportion:.2%} ")

#Save:
addresses.to_csv(out_file)
print(f'Runtime (classify addresses on file): {datetime.now()-start_time}')

############################################################################################