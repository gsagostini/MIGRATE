# Create a crosswalk of ZIP to Census blocks (with fractional values)
#####################################################################################################################################
import sys
sys.path.append('../../d03_src/')
import utils
import vars

import ast
import numpy as np
import pandas as pd
from tqdm import tqdm
from datetime import datetime
import itertools

#####################################################################################################################################

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--start_year", type=int, default=2012)
parser.add_argument("--crosswalk_dir", type=str, default=f'{vars._census_spatial_dir}crosswalks/')
parser.add_argument("--output_filename", type=str, default='HUD_ZIP_BLOCK.csv')
args = parser.parse_args()

#####################################################################################################################################

start = datetime.now()

#####################################################################################################################################

#Collect the TRACT TO BLOCK crosswalk:
TRACT_to_BLOCK_df = pd.read_csv(f'{vars._census_demographics_dir}processed/CENSUS-2010-tract-block-description.csv', index_col='TRACT')
TRACT_to_BLOCK_df.index = TRACT_to_BLOCK_df.index.astype(int)
TRACT_to_BLOCK_df['BLOCK'] = TRACT_to_BLOCK_df['BLOCK'].apply(ast.literal_eval)
TRACT_to_BLOCK_df['TRACT_FRAC'] = TRACT_to_BLOCK_df['TRACT_FRAC'].apply(ast.literal_eval)

print(f'Loading TRACT to BLOCK crosswalk in {datetime.now()-start}')

#####################################################################################################################################

def get_blocks_and_ZIP_fractions(ZIP):
    """
    NOTE: Recall that the columns must have been evaluated to lists with ast.literal
    """
    #Get the values:
    tracts = ZIP['TRACT']
    tracts_fractions = ZIP['ZIP_FRAC']
    #Iterate through tracts and probabilities:
    all_blocks = []
    all_blocks_fractions = []
    for tract, p_tract in zip(tracts, tracts_fractions):
        #Get the tract row:
        if tract in TRACT_to_BLOCK_df.index:
            tract_row = TRACT_to_BLOCK_df.loc[tract]
            blocks = tract_row['BLOCK']
            blocks_fractions = tract_row['TRACT_FRAC']
            #Multiply all blocks
            all_blocks.extend(blocks)
            all_blocks_fractions.extend(np.array(blocks_fractions)*p_tract)
    #Merge:
    all_blocks_df = pd.DataFrame([all_blocks, all_blocks_fractions], index=['BLOCKS', 'FRACTIONS']).T
    sorted = all_blocks_df.groupby('BLOCKS').sum().sort_values('FRACTIONS', ascending=False)
    blocks_clean, fractions_clean = sorted.index.values, sorted['FRACTIONS'].values
    return blocks_clean, fractions_clean

###################################################################################################################################

#Start with an empty dataframe:
crosswalk_cols = ['CENSUS_BLOCK_2010', 'ZIP_BLOCK_FRAC']
crosswalk_header = pd.DataFrame([], columns=['ZIP']+crosswalk_cols+['DATE'])
crosswalk_header.to_csv(f"{args.crosswalk_dir}{args.output_filename}", index=False)

#Produce a crosswalk for every month:
crosswalk_raw_dir = f"{args.crosswalk_dir}HUD_ZIP_to_tract/" #collected with API
for year, quarter in tqdm(itertools.product(range(args.start_year, 2022), [1, 2, 3, 4]), total=4*(2022-args.start_year)):
    #Collect the file:
    df_dtypes = {'zip':'int', 'geoid':'int', 'res_ratio':'float', 'bus_ratio':'float', 'oth_ratio':'float', 'tot_ratio':'float'}
    df = pd.read_csv(f"{crosswalk_raw_dir}{year}_quarter-{quarter}.csv", dtype=df_dtypes)
    #Find the tracts corresponding to each ZIP code:
    df = df.sort_values(by=['zip', 'res_ratio', 'tot_ratio', 'bus_ratio', 'oth_ratio'], ascending=[True, False, False, False, False])
    ZIP_to_TRACT_df = df.groupby('zip').agg(TRACT=('geoid',list), ZIP_FRAC=('res_ratio',list))
    ZIP_to_TRACT_df.index.name = 'ZIP'
    #Adjust lists where no tracts have population:
    ZIP_to_TRACT_df['ZIP_FRAC'] = ZIP_to_TRACT_df['ZIP_FRAC'].apply(utils.adjust_zeros)
    #Merge with the most populated block per tract:
    ZIP_to_TRACT_df[crosswalk_cols] = ZIP_to_TRACT_df.apply(get_blocks_and_ZIP_fractions, axis=1, result_type='expand')
    monthly_crosswalk = ZIP_to_TRACT_df[crosswalk_cols].explode(crosswalk_cols).reset_index()
    #Include year and month when quarter starts:
    monthly_crosswalk['DATE'] = datetime.fromisoformat(f"{year}-{(quarter-1)*3+1:02}-01")
    #Ignore zeros:
    clean_crosswalk = monthly_crosswalk[monthly_crosswalk['ZIP_BLOCK_FRAC'] > 0].reset_index(drop=True)
    #Save:
    clean_crosswalk.to_csv(f"{args.crosswalk_dir}{args.output_filename}", index=False, header=False, mode='a')

print(f'Collected all monthly crosswalks in {datetime.now()-start}')
        
#####################################################################################################################################