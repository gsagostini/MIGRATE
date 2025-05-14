############################################################################################
#This script merges state files of the form XX.csv into a single csv file.
############################################################################################

import pandas as pd
from datetime import datetime

import sys
sys.path.append('../../d03_src/')
import vars
import process_infutor as infutor

############################################################################################
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--chunksize", default=10_000_000, type=int) #should allocate around 50GB RAM
args = parser.parse_args()

############################################################################################

start_time = datetime.now()

############################################################################################

#Collect all the states:
with open(vars._states_filepath, 'r') as file:
    states = file.read().splitlines()
csv_file_list = [f'{vars._infutor_dir_processed}CRD4/{state}.csv' for state in states]

############################################################################################

#Collect the columns:
columns = infutor.get_CRD4_columns()
columns_internal = [col for col in columns if 'INTERNAL' in col]

#Create a dataframe with no data:
file_output = f'{vars._infutor_dir_processed}CRD4/full.csv'
merged_df = pd.DataFrame([], columns=columns).drop(columns_internal, axis=1).set_index('PID')
merged_df.to_csv(file_output, escapechar='\\')

############################################################################################

#Iterate over states:
for file_csv in csv_file_list:

    #Collect the dataframe in chunks:
    chunked_df = pd.read_csv(file_csv,
                             dtype='str',
                             index_col='PID',
                             chunksize=args.chunksize)

    #Add one chunk at a time to the merged dataframe:
    for chunk in chunked_df:
        chunk.to_csv(file_output, mode='a', header=False)

############################################################################################
print(f'Runtime: {datetime.now()-start_time}')