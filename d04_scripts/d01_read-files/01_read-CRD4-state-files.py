############################################################################################
#This script reads files of the form CRD4_XX.txt, which contain raw address information per
# PID (i.e. in a row, we have a PID and a sequence of addresses associated with the PID) into
# csv files of the same information.
############################################################################################

import pandas as pd
import numpy as np
import random
from datetime import datetime

import sys
sys.path.append('../../d03_src/')
import vars
import process_infutor as infutor

############################################################################################
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--state", type=str)
parser.add_argument("--in_chunks", default=False, action=argparse.BooleanOptionalAction)
parser.add_argument("--chunksize", default=100_000, type=int) #should allocate around 5GB RAM

args = parser.parse_args()

############################################################################################

start_time = datetime.now()
print(f'State: {args.state}')

#Get the file names:
file_input  = f'{vars._infutor_dir}CRD4/CRD4_{args.state}.txt'
file_output = f'{vars._infutor_dir_processed}CRD4/{args.state}.csv'

#Get the column names:
columns = infutor.get_CRD4_columns()
columns_internal = [col for col in columns if 'INTERNAL' in col]

############################################################################################
#Option 1: Process the state file all at once (faster, uses more RAM):
if not args.in_chunks:

    print('File will be processed all at once')
    #Open the file and read all lines:
    with open(file_input, 'r', encoding='latin-1') as file:
        lines=file.readlines()
    print(f'# lines: {len(lines)}')

    #Turn into a dataframe and clean:
    df = pd.DataFrame([s.strip('\n').split('\t') for s in lines], columns=columns).set_index('PID')
    df_clean = infutor.process_df(df, cols_to_drop=columns_internal)

    #Save:
    df_clean.to_csv(file_output, escapechar='\\')
    
############################################################################################
#Option 2: Process the state file in a few chunks (slower, uses less RAM)
def process_chunk(lines_in_chunk,
                  file=file_output, columns=columns, columns_to_drop=columns_internal):
    
    df_chunk = pd.DataFrame(lines_in_chunk, columns=columns).set_index('PID')
    df_clean = infutor.process_df(df_chunk, cols_to_drop=columns_to_drop)
    df_clean.to_csv(file, escapechar='\\', mode='a', header=False) 
    
    return [] #zero the lines list

if args.in_chunks:
    
    #Create a dataframe with no data:
    raw_df = pd.DataFrame([], columns=columns).drop(columns_internal, axis=1).set_index('PID')
    raw_df.to_csv(file_output, escapechar='\\')

    #Read the file and process into a Pandas DataFrame line by line:
    print(f'File will be processed in chunks of size {args.chunksize}')
    with open(file_input, 'r', encoding='latin-1') as file:
        
        #Initialize an empty list of lines:
        lines = []
        n_lines = 0
        #Iterate over the lines, saving one chunk at a time:
        for line in file:
            lines.append(line.strip('\n').split('\t'))
            n_lines += 1
            
            #If we reached the chunk limit, we process the dataframe:
            if n_lines % args.chunksize == 0: lines = process_chunk(lines)

        #We also save the dataframe for the last line:
        lines = process_chunk(lines)
                
    #Comput the total number of lines:
    print(f'# lines: {n_lines}')
############################################################################################
print(f'Runtime: {datetime.now()-start_time}')