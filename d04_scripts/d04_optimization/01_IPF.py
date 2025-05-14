############################################################################################
# Functions to apply the generalized IPF to a noisy INFUTOR matrix
############################################################################################
import numpy as np
import pandas as pd
import scipy.sparse as ss

from tqdm import tqdm,trange
from datetime import datetime

import sys
sys.path.append('../../d03_src/')
import vars
import process_census as prc
import process_infutor as pri
import optimization as opt
from utils import get_descriptor

############################################################################################
import argparse

parser = argparse.ArgumentParser()

#Basic parameters:
parser.add_argument("--year", type=int, default=2011)

#Data options:
parser.add_argument("--harmonize_constraints", default=False, action=argparse.BooleanOptionalAction)
parser.add_argument("--rescale_flows", default=False, action=argparse.BooleanOptionalAction)
parser.add_argument("--rescale_marginals", default=False, action=argparse.BooleanOptionalAction)
parser.add_argument("--ignore_PR", default=False, action=argparse.BooleanOptionalAction)
parser.add_argument("--fit_E_to_flows", default=False, action=argparse.BooleanOptionalAction)
parser.add_argument("--E_subdir", type=str, default='ADDRID')

parser.add_argument("--use_S", default=False, action=argparse.BooleanOptionalAction)
parser.add_argument("--holdout_ACSyear", type=int, default=0) #only pass this if you want to holdout a specific year
parser.add_argument("--holdout_CBG_t0", default=False, action=argparse.BooleanOptionalAction)
parser.add_argument("--holdout_CBG_t1", default=False, action=argparse.BooleanOptionalAction)
parser.add_argument("--holdout_county_flows", default=False, action=argparse.BooleanOptionalAction)
parser.add_argument("--holdout_state_flows", default=False, action=argparse.BooleanOptionalAction)
parser.add_argument("--holdout_state_nonmovers", default=False, action=argparse.BooleanOptionalAction)
parser.add_argument("--begin_with_state_nonmovers", default=False, action=argparse.BooleanOptionalAction)

#Use the least squares rolling average:
parser.add_argument("--retrieve_yearly_CBG_population", default=False, action=argparse.BooleanOptionalAction)
parser.add_argument("--use_2010_ACS", default=False, action=argparse.BooleanOptionalAction)
parser.add_argument("--NNLS", default=False, action=argparse.BooleanOptionalAction)

#IPF options:
parser.add_argument("--n_iterations", type=int, default=100_000)
parser.add_argument("--coarse_flows", default=False, action=argparse.BooleanOptionalAction)
parser.add_argument("--finish_on_row", default=False, action=argparse.BooleanOptionalAction)

#Saving:
parser.add_argument("--save_every_k", type=int, default=1_000)
parser.add_argument("--log_every_k", type=int, default=50)
parser.add_argument("--output_subdir", type=str, default='results')

args = parser.parse_args()

############################################################################################
output_dir = f'{vars._outputs_dir}IPF/{args.output_subdir}/'
constraint_types = ['row', 'column', 'checkerboard']
columns_for_results_df = ['iter', 'sub_iter',
                          'update_maximum', 'update_total',
                          'update_from0_maximum', 'update_from0_total',
                          'violation_row_maximum', 'violation_column_maximum', 'violation_checkerboard_maximum',
                          'violation_row_total', 'violation_column_total', 'violation_checkerboard_total']
if not args.coarse_flows:
    columns_for_results_df.remove('violation_checkerboard_maximum')
    columns_for_results_df.remove('violation_checkerboard_total')

############################################################################################

#Collect membership matrices and validate the state-level matrix (for speed up):
CBG_to_county = prc.get_geography_matrices('blockgroup', 'county', ignore_PR=args.ignore_PR)
CBG_to_state  = prc.get_geography_matrices('blockgroup', 'state',  ignore_PR=args.ignore_PR)
state_to_CBG_idx = opt.verify_aggregation_matrix(CBG_to_state)

############################################################################################

#Collect the constraints:
P0, P1, F, Fhat = opt.get_constraints(args.year-1,         #by default, function asks for initial year
                                      process_birthsdeaths=args.harmonize_constraints,
                                      rescale_marginals=args.rescale_marginals,
                                      rescale_flows=args.rescale_flows,
                                      ignore_PR=args.ignore_PR)

#Log updates as marginal, aggregation, type, error:
row_update    = (P0, CBG_to_county, 'row',          None)
column_update = (P1, CBG_to_county, 'column',       None)
flows_update  = (F,  CBG_to_state,  'checkerboard', Fhat)

#Order the constraints:
IPF_constraints = [column_update]
if args.coarse_flows: IPF_constraints.append(flows_update)
IPF_constraints.append(row_update) if args.finish_on_row else IPF_constraints.insert(0, row_update)

############################################################################################

#If we have not scaled beforehand:
if not args.use_S:
    #Collect the INFUTOR matrix:
    E = pri.load_INFUTOR_matrix(args.year, ignore_PR=args.ignore_PR,
                                E_matrix_dir=f'{vars._infutor_dir_processed}CRD4/OD_pairs/{args.E_subdir}/E_matrices/')
    
    #Consider pre-processing INFUTOR:
    if args.fit_E_to_flows:
        current_sums = opt.get_IPF_current_values(E, 'checkerboard', CBG_to_state)
        aggregated_scalers = opt.get_IPF_scaling(current_sums, F, ignore_zeros=True, tolerance=None, verbose=False)
        E = opt.scale_checkerboard_matrix(E, aggregated_scalers, CBG_to_state, C_idx_dict=state_to_CBG_idx)

#Otherwise, get the scaled version:
if args.use_S:
    assert args.ignore_PR, 'Scaled version is currently configured to 51 states, please ignore PR'
    #Collect the descriptor:
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
        
    S_matrix_str = f"{args.year}_blockgroup{descriptor}"
    E = ss.load_npz(f"{vars._infutor_dir_processed}CRD4/OD_pairs/{args.E_subdir}/S_matrices/{S_matrix_str}.npz")
    
############################################################################################

#Iterate:
M = E
results = []
for iteration in trange(1, args.n_iterations+1):
    #Is this a special iteration?
    iteration_to_log  = (iteration == 1) or (iteration % args.log_every_k == 0 ) or (iteration == args.n_iterations)
    iteration_to_save = (iteration % args.save_every_k == 0) or (iteration == args.n_iterations)
    if iteration_to_log: previous_M = M
    #We may pass several constraints, let's go one at a time:
    for (m, C, m_type, m_tol) in IPF_constraints:
        assert m_type.lower() in constraint_types
        #First, we get the current sums:
        current_sums = opt.get_IPF_current_values(M, m_type, C)
        #Second, we get the scalers:
        aggregated_scalers = opt.get_IPF_scaling(current_sums, m, ignore_zeros=True, tolerance=m_tol, verbose=False)
        #Third, cast scalers to the dimensions of M and multiply (varies with constraint type!)
        if m_type.lower() == 'checkerboard':
            updated_M = opt.scale_checkerboard_matrix(M, aggregated_scalers, C, C_idx_dict=state_to_CBG_idx)
        else:
            scalers = C @ aggregated_scalers
            R = ss.diags(scalers)
            updated_M = M @ R if m_type.lower() == 'column' else R @ M
        #Fourth, log violations and updates in the dataframe:
        if iteration_to_log:
            delta, delta_0 = abs(updated_M-M), abs(updated_M-previous_M)
            max_violation, tot_violation = opt.compute_violations(updated_M, IPF_constraints)
            max_update,    tot_update    = delta.max(),   delta.sum()
            max_update_0,  tot_update_0  = delta_0.max(), delta_0.sum()
            results.append([iteration, m_type, max_update, tot_update, max_update_0, tot_update_0]+max_violation+tot_violation)
        #Update:
        M = updated_M
    #Consider saving after one full iteration:
    if iteration_to_save:
        #Save the matrix:
        ss.save_npz(f"{output_dir}{args.year}_M{iteration if iteration != args.n_iterations else ''}{descriptor}.npz", M)
        #Save the dataframe:
        results_df = pd.DataFrame(results, columns=columns_for_results_df)
        results_df.to_csv(f'{output_dir}{args.year}_results{descriptor}.csv', index=False)
        
############################################################################################