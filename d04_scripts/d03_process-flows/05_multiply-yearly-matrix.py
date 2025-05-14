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
parser.add_argument("--geocoding_dir", type=str, default='CRD4/addresses/geocoding_matrix/')
parser.add_argument("--geography", type=str, default='block')
args = parser.parse_args()

#####################################################################################################################################

start = datetime.now()
year = args.base_year + args.delta_year
print(f'Year: {year}')

#####################################################################################################################################

#Get the matrices:
_start = datetime.now()
A = ss.load_npz(f'{vars._infutor_dir_processed}{args.ADDRID_dir}A_matrices/{year}_csc.npz')
G = ss.load_npz(f'{vars._infutor_dir_processed}{args.geocoding_dir}geocoding_{args.geography}_csc.npz')
print(f'Collecting matrices runtime: {datetime.now()-_start}')

#####################################################################################################################################

#Multiply the matrices:
_start = datetime.now()
S = A.diagonal()
E_movers = G.T @ ((A - ss.diags(S, format='csc')) @ G)
E_stayers = ss.diags((G.T).dot(S), format='csc')
E = E_movers + E_stayers
print(f'Multiplying matrices runtime: {datetime.now()-_start}')

#####################################################################################################################################

ss.save_npz(f'{vars._infutor_dir_processed}{args.ADDRID_dir}E_matrices/{year}_{args.geography}.npz', E)
print(f'Runtime: {datetime.now()-start}')

#####################################################################################################################################