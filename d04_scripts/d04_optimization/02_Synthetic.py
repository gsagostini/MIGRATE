############################################################################################
# Functions to perform a synthetic experiment
############################################################################################
import pandas as pd
import scipy.sparse as ss

from tqdm import trange

import sys
sys.path.append('../../d03_src/')
import vars
import process_census as prc
import optimization as opt
import synthetic as syn

############################################################################################
import argparse

parser = argparse.ArgumentParser()

#Basic parameters:
parser.add_argument("--experiment",  type=int, default=1)
parser.add_argument("--idx",         type=int, default=0)
parser.add_argument("--year",        type=int, default=2011)

#Data options:
parser.add_argument("--ignore_PR", default=False, action=argparse.BooleanOptionalAction)

#Noise options:
parser.add_argument("--noise_mean",   type=float, default=1.00)
parser.add_argument("--noise_scale",  type=float, default=0.25)
parser.add_argument("--noise_type",   type=str,   default='LogNormal')
parser.add_argument("--noise_random", default=False, action=argparse.BooleanOptionalAction)
parser.add_argument("--noise_zscore", default=False, action=argparse.BooleanOptionalAction)

#Bias options:
parser.add_argument("--bias",             type=float, default=0.)
parser.add_argument("--bias_scale",       type=float, default=0.)
parser.add_argument("--bias_demographic", type=str,   default='Race')
parser.add_argument("--bias_group",       type=str,   default='White')

#IPF options:
parser.add_argument("--n_iterations", type=int, default=3_000)

args = parser.parse_args()
print(args)

############################################################################################
#Collect the data:
_geographies = ['blockgroup', 'tract', 'county', 'state']
C_dict = {}
for fine_idx, fine_geography in enumerate(_geographies):
    for coarse_geography in _geographies[1+fine_idx:]:
        C_dict[(fine_geography, coarse_geography)] = prc.get_geography_matrices(fine_geography, coarse_geography, ignore_PR=args.ignore_PR)

#Load the final version of our M matrix:
M = ss.load_npz(f"{vars._outputs_dir}d02_IPF/Main/{args.year}_M_match-CBGP0_match-state_match-state-nonmovers_NNLS_wACS2010.npz")

#Get the bias weights:
w = prc.get_demographics(geography='BLOCKGROUP', features=(args.bias_demographic, args.bias_group), ignore_PR=args.ignore_PR, years=args.year, pct=True)[args.year]

############################################################################################
#Generate synthetic data:
E, b, bias = syn.generate_synthetic_infutor(M, C_dict=C_dict,
                                            noise_type=args.noise_type, noise_mean=args.noise_mean, noise_scale=args.noise_scale, random_noise=args.noise_random, zscore=args.noise_zscore,
                                            bias=args.bias, b=args.bias_scale, bias_weights=w)
if E is None:
    print(f'Target bias {args.bias:.2%} on {args.bias_group} population is unreachable.')
    sys.exit(0)
else:
    print(f'Generated a random matrix.')
    print(f'Bias on {args.bias_group} population: {bias}')
    print(f'Bias scale: {b}')

#Save the synthetic matrix:
ss.save_npz(f'{vars._path_to_repo}d05_outputs/d06_Synthetic/IPF/matrix_perturbed/{args.year}_{args.idx}_{args.experiment}.npz', E)

############################################################################################
#Collect Census constraints:
P0_CBG, _ = syn.collect_population_sums(M, C=None)
P0_County, P1_County = syn.collect_population_sums(M, C=C_dict[('blockgroup', 'county')])
F_State = syn.collect_population_flows(M, C=C_dict[('blockgroup', 'state')])
stayers_State, movers_State  = syn.collect_population_movers(M, C=C_dict[('blockgroup', 'state')])

############################################################################################
#First we scale rows according the CBG populations:
current_values, _ = syn.collect_population_sums(E, C=None)
scalers = opt.get_IPF_scaling(current_values, P0_CBG, ignore_zeros=True)
scalers_cast = ss.diags(scalers)
E = scalers_cast @ E

############################################################################################
#Second, we scale diagonals according to state non-movers:
current_stayers, current_movers = syn.collect_population_movers(E, C=C_dict[('blockgroup', 'state')])
current_diagonal = E.diagonal()

scalers_offd = opt.get_IPF_scaling(current_movers,  movers_State,  ignore_zeros=True)
scalers_diag = opt.get_IPF_scaling(current_stayers, stayers_State, ignore_zeros=True)
scalers_cast_offd = ss.diags(C_dict[('blockgroup', 'state')] @ scalers_offd)
scalers_cast_diag = C_dict[('blockgroup', 'state')] @ scalers_diag

E = E @ scalers_cast_offd
E.setdiag(current_diagonal * scalers_cast_diag)

############################################################################################
#Third, we scale flows according to state-to-state flows:
C = C_dict[('blockgroup', 'state')]
idx = opt.verify_aggregation_matrix(C)

current_flows = syn.collect_population_flows(E, C=C)
scalers = opt.get_IPF_scaling(current_flows, F_State, ignore_zeros=True)
E = opt.scale_checkerboard_matrix(E, scalers, C, C_idx_dict=idx)

############################################################################################
#Fourth, we iteratively scale columns and rows according to county populations:
C=C_dict[('blockgroup', 'county')]
total_update = []
for i in trange(args.n_iterations, desc='IPF iterations'):
    _E = E
    #Fit:
    E = syn.IPF_update(E, P1_County, 'column', C=C)
    E = syn.IPF_update(E, P0_County, 'row',    C=C)
    #Log total update:
    total_update.append(abs(E - _E).sum())

############################################################################################
#Save the updates:
updates = pd.Series(total_update)
updates.index.name = 'iteration'
updates.index = updates.index + 1
updates.to_csv(f'{vars._path_to_repo}d05_outputs/d06_Synthetic/IPF/iterations/{args.year}_{args.idx}_{args.experiment}.csv')

############################################################################################
#Save the matrix:
ss.save_npz(f'{vars._path_to_repo}d05_outputs/d06_Synthetic/IPF/matrix_inferred/{args.year}_{args.idx}_{args.experiment}.npz', E)

print('Completed!')