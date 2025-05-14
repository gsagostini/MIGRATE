# Spatiotemporal smoothing:
#####################################################################################################################################
import sys
sys.path.append('../../d03_src/')
import vars
import os
import numpy as np
import pandas as pd
import scipy.sparse as ss
from scipy.optimize import nnls

from tqdm import tqdm,trange
from datetime import datetime
import process_infutor as pri
import process_census as prc
import optimization as opt
from utils import get_descriptor

#####################################################################################################################################

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--ADDRID_dir", type=str, default='CRD4/OD_pairs/ADDRID/')

#Holdout options:
parser.add_argument("--holdout_ACSyear", type=int, default=0) #only pass this if you want to holdout a specific year
parser.add_argument("--holdout_CBG_t0", default=False, action=argparse.BooleanOptionalAction)
parser.add_argument("--holdout_CBG_t1", default=False, action=argparse.BooleanOptionalAction)
parser.add_argument("--holdout_county_flows", default=False, action=argparse.BooleanOptionalAction)
parser.add_argument("--holdout_state_flows", default=False, action=argparse.BooleanOptionalAction)
parser.add_argument("--holdout_state_nonmovers", default=False, action=argparse.BooleanOptionalAction)
parser.add_argument("--begin_with_state_nonmovers", default=False, action=argparse.BooleanOptionalAction)

#Use the least squares rolling average:
parser.add_argument("--retrieve_yearly_CBG_population", default=False, action=argparse.BooleanOptionalAction)
parser.add_argument("--NNLS", default=False, action=argparse.BooleanOptionalAction)
parser.add_argument("--use_2010_ACS", default=False, action=argparse.BooleanOptionalAction)

args = parser.parse_args()

#####################################################################################################################################

start = datetime.now()

#####################################################################################################################################

E_dict = {year: pri.load_INFUTOR_matrix(year, ignore_PR=True, 
                                        E_matrix_dir=f'{vars._infutor_dir_processed}{args.ADDRID_dir}E_matrices/') 
          for year in trange(2007,2021)}

#####################################################################################################################################

ACS_pop_years  = [x for x in range(2013,2020) if x != args.holdout_ACSyear]
print('For CBG population we use: ', ACS_pop_years)
ACS_flow_years = [x for x in range(2011,2020) if x != args.holdout_ACSyear]
print('For flows we use: ', ACS_flow_years)

ACS_CBG = prc.get_demographics(features=('Total', 'Population'), geography='BLOCKGROUP', ignore_PR=True)
Census_CBG = prc.get_census2010_CBG_demographics('Population', ignore_PR=True)
ACS_flows, ACS_flows_moe = {'state':{}, 'county':{}}, {'state':{}, 'county':{}}
for year in ACS_flow_years:
    ACS_flows['state' ][year], ACS_flows_moe['state' ][year] = prc.get_ACS1_state_to_state(  year, ignore_PR=True)
    ACS_flows['county'][year], ACS_flows_moe['county'][year] = prc.get_ACS5_county_to_county(year, ignore_PR=True)

#Collect non-movers per state and adjust (to account for deaths and emmigrants):
same_house = {year: prc.get_ACS1_state_nonmovers(year, ignore_PR=True)['Estimate'].values for year in ACS_flow_years}
state_components = {year: prc.get_PEP_state_components(year, ignore_PR=True) for year in ACS_flow_years}
diagonal_corrections = {year: state_components[year]['deaths'] + state_components[year]['emmigr'] for year in ACS_flow_years}

#Non-movers: people who stay in their houses, die, or emmigrate
nonmovers = {year: same_house[year] + diagonal_corrections[year] for year in ACS_flow_years}

#Movers: people who reside at a state in time t1, did not immigrate nor were born in the previous year, and did not stay in their homes.
movers = {year: state_components[year]['P1'] - state_components[year]['immigr'] - state_components[year]['births'] - same_house[year] for year in ACS_flow_years}
for year in ACS_flow_years: assert np.all(movers[year]) > 0

#Adjust state-to-state flows with the diagonal corrections:
for year in ACS_flow_years:
    ACS_flows['state'][year] += np.diag(diagonal_corrections[year])

#####################################################################################################################################

_geographies = ['blockgroup', 'tract', 'county', 'state']
C_dict = {}
for fine_idx, fine_geography in enumerate(_geographies):
    for coarse_geography in _geographies[1+fine_idx:]:
        C_dict[(fine_geography, coarse_geography)] = prc.get_geography_matrices(fine_geography,
                                                                                coarse_geography,
                                                                                ignore_PR=True)

#####################################################################################################################################
S_dict = {}
#####################################################################################################################################
#Scale the diagonals of matrices according to state non-movers:
if not args.holdout_state_nonmovers and args.begin_with_state_nonmovers:
    print('Scaling according to state non-movers')
    geography, delta = 'state', 1
    C = C_dict[('blockgroup', geography)]
    for year in tqdm(ACS_flow_years):
        #Get the matrices:
        years_to_average = range(year-delta+1, year+1)
        E_matrices_to_average = {y: S_dict[y] if y in S_dict else E_dict[y] for y in years_to_average}
        E_average = sum(E_matrices_to_average.values())/len(E_matrices_to_average)
        #Collect current values for both the diagonal and the off-diagonal entries:
        current_diag   = C.T @ E_average.diagonal()
        current_colsum = np.array((C.T @ E_average @ C).sum(axis=0)).flatten()
        current_offd   = current_colsum - current_diag
        #We need to match to actual non-movers and mover values:
        target_offd = movers[year]
        target_diag = nonmovers[year]
        #Get scalers:
        scalers_offd = opt.get_IPF_scaling(current_offd, target_offd, ignore_zeros=True, tolerance=None, verbose=False)
        scalers_diag = opt.get_IPF_scaling(current_diag, target_diag, ignore_zeros=True, tolerance=None, verbose=False)
        #Now, we can first scale the off-diagonal entries and then replace the diagonal:
        cast_scales_offd = ss.diags(C @ scalers_offd)
        cast_scales_diag = C @ scalers_diag
        #Go through all matrices in average (this will be only one matrix if we use state flows only---yearly):
        for y,E in E_matrices_to_average.items():
            #First, extract the diagonal:
            d = E.diagonal()
            #Column scaling:
            S_dict[y] = E @ cast_scales_offd
            #Diagonal replacement:
            S_dict[y].setdiag(d * cast_scales_diag)

#####################################################################################################################################
delta = 5 #5-year populations
#Scale the 5-year matrices according to CBG populations:
if args.retrieve_yearly_CBG_population:
    assert args.holdout_ACSyear not in range(2013,2020), 'Rolling average method is problematic if holding out a specific 2013-2019 year from ACS'
    ACS_hat_CBG = prc.resolve_ACS(ACS_CBG, Census_2010=Census_CBG, use_ACS_2010=args.use_2010_ACS, use_NNLS=args.NNLS)
    #Updates:
    for year, population in tqdm(ACS_hat_CBG.items()):
            
        #First, scale rows:
        if not args.holdout_CBG_t0:
            E = S_dict.get(year+1, E_dict[year+1])   #We will use the ACS t=year population on the rows of the t > t+1 matrix, indexed by t+1
            current_values = np.array(E.sum(axis=1)).flatten()
            scalers = opt.get_IPF_scaling(current_values, population, ignore_zeros=True, tolerance=None, verbose=False)
            R = ss.diags(scalers)
            S_dict[year+1] = R@E
            
        #Column:
        if not args.holdout_CBG_t1:
            E = S_dict.get(year, E_dict[year])
            current_values = np.array(E.sum(axis=0)).flatten()
            scalers = opt.get_IPF_scaling(current_values, population, ignore_zeros=True, tolerance=None, verbose=False)
            R = ss.diags(scalers)
            S_dict[year] = E@R
else:
    for year, population in tqdm(ACS_CBG.items()):
        
        #First, scale rows:
        if not args.holdout_CBG_t0:
            years_to_average = range(year-delta+2, year+2)      #Be careful here, we want to go one year further, because we get the P0
            E_matrices_to_average = {y: S_dict[y] if y in S_dict else E_dict[y] for y in years_to_average}
            E_average = sum(E_matrices_to_average.values())/len(E_matrices_to_average)
            current_values = np.array(E_average.sum(axis=1)).flatten()
            scalers = opt.get_IPF_scaling(current_values, population, ignore_zeros=True, tolerance=None, verbose=False)
            R = ss.diags(scalers)
            for y,E in E_matrices_to_average.items():
                S_dict[y] = R@E
    
        #Then, scale columns:
        if not args.holdout_CBG_t1:
            years_to_average = range(year-delta+1, year+1)
            E_matrices_to_average = {y: S_dict[y] if y in S_dict else E_dict[y] for y in years_to_average}
            E_average = sum(E_matrices_to_average.values())/len(E_matrices_to_average)
            current_values = np.array(E_average.sum(axis=0)).flatten()
            scalers = opt.get_IPF_scaling(current_values, population, ignore_zeros=True, tolerance=None, verbose=False)
            R = ss.diags(scalers)
            for y,E in E_matrices_to_average.items():
                S_dict[y] = E@R
                
#####################################################################################################################################

#Scale the diagonals of matrices according to state non-movers:
if not args.holdout_state_nonmovers and not args.begin_with_state_nonmovers:
    print('Scaling according to state non-movers')
    geography, delta = 'state', 1
    C = C_dict[('blockgroup', geography)]
    for year in tqdm(ACS_flow_years):
        #Get the matrices:
        years_to_average = range(year-delta+1, year+1)
        E_matrices_to_average = {y: S_dict[y] if y in S_dict else E_dict[y] for y in years_to_average}
        E_average = sum(E_matrices_to_average.values())/len(E_matrices_to_average)
        #Collect current values for both the diagonal and the off-diagonal entries:
        current_diag   = C.T @ E_average.diagonal()
        current_colsum = np.array((C.T @ E_average @ C).sum(axis=0)).flatten()
        current_offd   = current_colsum - current_diag
        #We need to match to actual non-movers and mover values:
        target_offd = movers[year]
        target_diag = nonmovers[year]
        #Get scalers:
        scalers_offd = opt.get_IPF_scaling(current_offd, target_offd, ignore_zeros=True, tolerance=None, verbose=False)
        scalers_diag = opt.get_IPF_scaling(current_diag, target_diag, ignore_zeros=True, tolerance=None, verbose=False)
        #Now, we can first scale the off-diagonal entries and then replace the diagonal:
        cast_scales_offd = ss.diags(C @ scalers_offd)
        cast_scales_diag = C @ scalers_diag
        #Go through all matrices in average (this will be only one matrix if we use state flows only---yearly):
        for y,E in E_matrices_to_average.items():
            #First, extract the diagonal:
            d = E.diagonal()
            #Column scaling:
            S_dict[y] = E @ cast_scales_offd
            #Diagonal replacement:
            S_dict[y].setdiag(d * cast_scales_diag)
            #Assert:
            #assert abs(C.T @ S_dict[year].diagonal() - nonmovers[year]).max() < 1e-5
            #assert abs(np.array((C.T @ S_dict[year] @ C).sum(axis=0)).flatten() - C.T @ S_dict[year].diagonal() - movers[year]).max() <1e-5

#####################################################################################################################################

#Scale the 5-year matrices according to county-to-county flows:
if not args.holdout_county_flows:
    delta = 5 #5-year matrix
    C = C_dict[('blockgroup', 'county')]              #NOTE: C-matrix here is not suitable for fast processing due to Bedford county
    for year, flows in tqdm(ACS_flows['county'].items()):
        years_to_average = range(year-delta+1, year+1)
        E_matrices_to_average = {y: S_dict[y] if y in S_dict else E_dict[y] for y in years_to_average}
        E_average = sum(E_matrices_to_average.values())/len(E_matrices_to_average)
        current_values = C.T @ E_average @ C
        scalers = opt.get_IPF_scaling(current_values, flows, ignore_zeros=True, tolerance=None, verbose=False)
        #Slow step!
        A = C @ scalers @ C.T
        for y,E in E_matrices_to_average.items():
            S_dict[y] = E.multiply(A).tocsr()

#####################################################################################################################################

#Scale each matrix according to state-to-state flows:
if not args.holdout_state_flows:
    delta = 1 #yearly matrix
    C = C_dict[('blockgroup', 'state')]
    idx = opt.verify_aggregation_matrix(C) #state-to-state flows are optimized!
    for year, flows in tqdm(ACS_flows['state'].items()):
        years_to_average = range(year-delta+1, year+1)
        E_matrices_to_average = {y: S_dict[y] if y in S_dict else E_dict[y] for y in years_to_average}
        E_average = sum(E_matrices_to_average.values())/len(E_matrices_to_average)
        current_values = C.T @ E_average @ C
        scalers = opt.get_IPF_scaling(current_values, flows, ignore_zeros=True, tolerance=None, verbose=False)
        for y,E in E_matrices_to_average.items():
            S_dict[y] = opt.scale_checkerboard_matrix(E, scalers, C, C_idx_dict=idx)

#####################################################################################################################################

#Save the matrix:

descriptor = get_descriptor(holdout_ACS_year=args.holdout_ACSyear,
                            retrieve_yearly_CBG_population=args.retrieve_yearly_CBG_population,
                            NNLS=args.NNLS,
                            use_2010_ACS=args.use_2010_ACS,
                            holdout_CBG_t0=args.holdout_CBG_t0,
                            holdout_CBG_t1=args.holdout_CBG_t1,
                            holdout_county_flows=args.holdout_county_flows,
                            holdout_state_flows=args.holdout_state_flows,
                            holdout_state_nonmovers=args.holdout_state_nonmovers,
                            begin_with_nonmovers=args.begin_with_state_nonmovers)

for year, S in S_dict.items():
    ss.save_npz(f"{vars._infutor_dir_processed}{args.ADDRID_dir}S_matrices/{year}_blockgroup{descriptor}.npz", S)

#####################################################################################################################################

print(f'Runtime: {datetime.now()-start}')

#####################################################################################################################################