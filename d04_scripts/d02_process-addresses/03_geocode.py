import pandas as pd
from datetime import datetime
import sys
sys.path.append('../../d03_src/')
import vars

import os
from censusgeocode import CensusGeocode
import time
import backoff
import requests

############################################################################################

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--array_idx", type=int)
parser.add_argument("--base_idx", type=int, default=0)   #multiplied by 1_000
parser.add_argument("--geocoding_subdir", type=str, default='Census_geocoding')
parser.add_argument("--max_hours", type=float, default=2)   # maximum number of hours per request
args = parser.parse_args()

############################################################################################

start_time = datetime.now()

############################################################################################

@backoff.on_exception(backoff.expo,
                      requests.exceptions.RequestException,
                      max_time=60*60*args.max_hours)
def batch_geocode(file):
    geocoder = CensusGeocode(vintage='Census2010_Current')
    result = geocoder.addressbatch(file)
    result_df = pd.DataFrame(result)
    return result_df

idx = args.base_idx*1_000 + args.array_idx
print('Index:', idx)
input_file = f'{vars._infutor_dir}CRD4/addresses/{args.geocoding_subdir}_input/addresses_to_geocode_{idx}.csv'
output_file = f'{vars._infutor_dir}CRD4/addresses/{args.geocoding_subdir}_output/geocoded_{idx}.csv'

#Check if input file exists:
if os.path.isfile(input_file):
    #Skip if output file exists:
    if not os.path.isfile(output_file):
        print('Geocoding')
        time.sleep(2*(idx%50))
        result_df = batch_geocode(input_file)
        result_df.to_csv(output_file)
        print(f'Runtime: {datetime.now()-start_time}')
        print(f'Output length: {len(result_df)}')
    else:
        print('Output file already exists')
else:
    print('Input file does not exist')