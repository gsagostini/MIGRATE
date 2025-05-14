############################################################################################
# Functions to process ZIP codes
#     1. Functions to download HUD crosswalks
#     2. Functions to read HUD crosswalks

############################################################################################

import pandas as pd
import requests
from tqdm import tqdm, trange

import sys
sys.path.append('../d03_src/')
import vars

############################################################################################
#1. Functions to read HUD crosswalks:
def download_HUD_crosswalks(token=vars._census_crosswalk_key,
                            out_dir=f"{vars._census_spatial_dir}crosswalks/HUD-ZIP_to_tract/"):
    
    #Configure the API:
    base_url = 'https://www.huduser.gov/hudapi/public/usps?'
    headers = {'Authorization': f"Bearer {token}"}
        
    #Get the result for every quarter:
    for year in trange(2012, 2022):
        for quarter in [1, 2, 3, 4]:
            #Build the query:
            url = f"{base_url}type=1&query=All&year={year}&quarter={quarter}"
            #Make an API request:
            response = requests.get(url, headers=headers)
            #If we did not faile, proceed:
            if response.status_code == 200:
                df = pd.DataFrame(response.json()["data"]["results"])
                df.to_csv(f"{out_dir}{year}_quarter-{quarter}.csv", index=False)
            else:
                print(f'Failure on quarter {quarter} of {year}. Status code: {response.status_code}')

############################################################################################
#2. Functions to process crosswalks:
def geocode_ZIP_and_DATE(ZIP, date,
                         crosswalk_df):
    """
    Query the ZIP to BLOCK crosswalk in order to find the
        ZIP code composition at the best available date

    Returns
    ----------
    1D array, 1D array
        Census Blocks and their respective probabilities
    """
    
    #Select the ZIP code entries in the crosswalk:
    ZIP = int(ZIP)
    ZIP_crosswalk = crosswalk_df.loc[crosswalk_df['ZIP'] == ZIP]

    #If not present, we return an empty list:
    if len(ZIP_crosswalk) == 0:
        blocks, p = np.array([]), np.array([])
    #Otherwise, find the best date:
    else:
        #If the date isn't available, impute latest date:
        if pd.isna(date): date = ZIP_crosswalk['DATE'].max()
        #Find the best among available dates in the ZIP code:
        available_dates = ZIP_crosswalk['DATE'].values
        best_date = min(available_dates, key=lambda x:abs(x-date))
        #Get the values:
        ZIP_crosswalk_best = ZIP_crosswalk.loc[ZIP_crosswalk['DATE']==best_date]
        blocks = ZIP_crosswalk_best['CENSUS_BLOCK_2010'].values
        p = ZIP_crosswalk_best['ZIP_BLOCK_FRAC'].values

    return blocks, p

def get_blocks_and_ZIP_fractions(ZIP, TRACT_to_BLOCK_df):
    """
    Recall that the columns must have been eluated to lists with ast.literal
    """
    #Get the values:
    tracts = ZIP['TRACT']
    tracts_fractions = ZIP['ZIP_FRAC']
    #Iterate through tracts and probabilities:
    lat, lon = None, None
    all_blocks = []
    all_blocks_fractions = []
    for tract, p_tract in zip(tracts, tracts_fractions):
        #Get the tract row:
        if tract in TRACT_to_BLOCK_df.index:
            tract_row = TRACT_to_BLOCK_df.loc[tract]
            blocks = tract_row['BLOCK']
            blocks_fractions = tract_row['TRACT_FRAC']
            if lat is None: lat = tract_row['INTPTLAT']
            if lon is None: lon = tract_row['INTPTLON']
            #Multiply all blocks
            all_blocks.extend(blocks)
            all_blocks_fractions.extend(np.array(blocks_fractions)*p_tract)
    #Merge:
    all_blocks_df = pd.DataFrame([all_blocks, all_blocks_fractions], index=['BLOCKS', 'FRACTIONS']).T
    sorted = all_blocks_df.groupby('BLOCKS').sum().sort_values('FRACTIONS', ascending=False)
    blocks_clean, fractions_clean = sorted.index.values, sorted['FRACTIONS'].values
    return lat, lon, blocks_clean, fractions_clean
    
############################################################################################

def validate_ZIP(zip_code):
    """
    Processes a ZIP code into a 5-digit numeric string or nan
    """
    
    #Skip np.nan values:
    if pd.isna(zip_code): return np.nan
        
    #Convert to string
    zip_str = str(zip_code)
    
    #Remove trailing '.0' if present (on floats)
    if zip_str.endswith('.0'): zip_str = zip_str[:-2]
    
    #Check if the zip code is a valid 5 or 9 digit number
    if zip_str.isdigit():
        if len(zip_str) == 5: return zip_str
        elif len(zip_str) == 9: return zip_str[:5]
            
    #Return np.nan for invalid zip codes
    return np.nan