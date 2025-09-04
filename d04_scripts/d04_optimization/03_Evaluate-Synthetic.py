############################################################################################
# Functions to read results from synthetic experiments:
############################################################################################
import os
import sys
sys.path.append('../../d03_src/')
import vars
import utils
import process_census as prc
import evaluation

import scipy.sparse as ss
from tqdm import tqdm
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
import statsmodels.stats.weightstats as smw

############################################################################################
import argparse
import multiprocessing as mp  # ### [PARALLEL]

parser = argparse.ArgumentParser()

#Input directories:
parser.add_argument("--out_files_dir", type=str, default=f'{vars._path_to_repo}d04_scripts/jobs/_outputs/synth/')
parser.add_argument("--synthetic_dir", type=str, default=f'{vars._path_to_repo}d05_outputs/d06_Synthetic/')

#Data options:
parser.add_argument("--ignore_PR", default=False, action=argparse.BooleanOptionalAction)
parser.add_argument('--years', nargs='*', type=int, default=list(range(2011,2020)))
parser.add_argument('--experiments', nargs='*', type=int, default=list(range(1,100)))

#Tolerance:
parser.add_argument("--zero_tol", type=float, default=1e-6)

# ### [PARALLEL] number of workers (defaults to SLURM_CPUS_PER_TASK or os.cpu_count())
parser.add_argument("--n_workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 0)) or (os.cpu_count() or 1))

args = parser.parse_args()
print(args)

############################################################################################
#Collect the data shared by all workers:

geographies = ['blockgroup', 'tract', 'county', 'state']
C_dict = {}
for fine_idx, fine_geography in enumerate(geographies):
    for coarse_geography in geographies[1+fine_idx:]:
        C_dict[(fine_geography, coarse_geography)] = prc.get_geography_matrices(fine_geography, coarse_geography, ignore_PR=args.ignore_PR)

M_dict = {year: ss.load_npz(f"{vars._outputs_dir}d02_IPF/Main/{year}_M_match-CBGP0_match-state_match-state-nonmovers_NNLS_wACS2010.npz") for year in range(2011, 2020)}

############################################################################################

#Check what files we have:
files_available = [f for f in os.listdir(f'{args.synthetic_dir}IPF/matrix_inferred/') if f[-3:]=='npz' and int(f[:4]) in args.years and int(f.split('_')[2][:-4]) in args.experiments]
files_available.sort()
assert set(os.listdir(f'{args.synthetic_dir}IPF/matrix_perturbed/')).issuperset(set(files_available)), "Some inferred matrix have no perturbed counterpart"
print(f'Will process {len(files_available)} files')

############################################################################################
#We will build a dataframe per quantity:
quantities = ['population', 'flows', 'movers', 'stayers', 'in-migration', 'in-migration_rate']

# ### [PARALLEL] Worker function containing the original per-file body.
def _process_one_file(file):
    try:
        data = {'year': int(file.split('_')[0]), 'configuration': int(file.split('_')[1]), 'experiment': int(file.split('_')[2][:-4])}

        out_filename = f"{args.out_files_dir}Y{data['year']}_E{data['experiment']}_{data['configuration']}.out"
        if not os.path.exists(out_filename):
            # Skip if output not present yet
            return None
        with open(out_filename, 'r') as f:
            out_file = [line.strip() for line in f]

        if not out_file or out_file[-1] != 'Completed!':
            # Skip unfinished runs
            return None

        # Confirm args in .out and update data dict
        import argparse as _argparse  # avoid shadowing
        arguments = eval(out_file[0], {"Namespace": _argparse.Namespace}).__dict__
        assert arguments['experiment'] == data['experiment']
        assert arguments['year'] == data['year']
        assert arguments['idx'] == data['configuration']
        data.update(arguments)

        # Add IPF last update
        data['IPF_last_update'] = pd.read_csv(f"{args.synthetic_dir}IPF/iterations/{file[:-4]}.csv")['0'].values[-1]

        # Update bias and scale based on empirical evidence:
        data['empirical_bias_scale'] = float(out_file[-2].split(': ')[-1])
        data['empirical_bias_value'] = float(out_file[-3].split(': ')[-1])

        # Load matrices
        M = M_dict[data['year']]
        E_inferred  = ss.load_npz(f"{args.synthetic_dir}IPF/matrix_inferred/{file}")
        E_perturbed = ss.load_npz(f"{args.synthetic_dir}IPF/matrix_perturbed/{file}")

        # Canonicalize
        _ = [utils.csr_in_canonical_form(x) for x in [M, E_inferred, E_perturbed]]

        # Rescale perturbed
        alpha = M.sum()/E_perturbed.sum()
        E_perturbed *= alpha
        assert abs(E_inferred.sum()-M.sum()) < args.zero_tol
        matrices = {'Ground Truth': M, 'Perturbed': E_perturbed, 'Inferred': E_inferred}

        # We need population for weights
        population = {area:{m:None for m in matrices} for area in geographies}

        per_file_series = {}  # quantity -> pd.Series

        for quantity in quantities:
            data_per_quantity = {}
            for area in geographies:
                C = C_dict[('blockgroup', area)] if area != 'blockgroup' else None
                values = {}
                for matrix_name, matrix in matrices.items():
                    flow_matrix = (C.T @ matrix @ C) if area != 'blockgroup' else matrix
                    values[matrix_name] = evaluation.extract_summary(flow_matrix, quantity=quantity.split('_')[0], rate=('rate' in quantity), normalize_by_column=True) if quantity != 'movers' else flow_matrix
                if quantity == 'population':
                    population[area] = values
                for matrix_name in ['Inferred', 'Perturbed']:
                    x, y = values[matrix_name], values['Ground Truth']
                    w = population[area][matrix_name] if 'migration' in quantity else None
                    if not ss.issparse(x):
                        x, y = utils.flat(x), utils.flat(y)
                        entries = np.isfinite(x)*np.isfinite(y)
                        x, y = x[entries], y[entries]
                        if w is not None:
                            w = w[entries]
                    RMSE = utils.RMSE(x, y, w, include_diagonal=(quantity!='movers'))
                    data_per_quantity[('RMSE', area, matrix_name)] = RMSE if RMSE > args.zero_tol else 0
                    if ss.issparse(x):
                        rho = utils.sparse_pearsonr(x, y, include_zeros=True, include_diagonal=(quantity!='movers'))
                    else:
                        if w is None:
                            rho = pearsonr(x, y)[0]
                        else:
                            rho = smw.DescrStatsW(np.column_stack((x, y)), weights=w).corrcoef[0,1]
                    data_per_quantity[('Pearson', area, matrix_name)] = rho
                data_per_quantity[('RMSE', area, 'Reduction (%)')] = (
                    100*(1 - data_per_quantity[('RMSE', area, 'Inferred')]/data_per_quantity[('RMSE', area, 'Perturbed')])
                    if data_per_quantity[('RMSE', area, 'Perturbed')] else np.nan
                )

            # Aux fields
            data_per_quantity[('Experiment', 'Setting', 'Year')] = data['year']
            data_per_quantity[('Experiment', 'Setting', 'Parameter configuration')] = data['configuration']
            data_per_quantity[('Experiment', 'Setting', 'Run #')] = data['experiment']

            data_per_quantity[('Experiment', 'IPF', 'Iterations')] = data['n_iterations']
            data_per_quantity[('Experiment', 'IPF', 'Last total update')] = data['IPF_last_update']

            data_per_quantity[('Simulation parameters', 'Noise', 'Mean')] = data['noise_mean']
            data_per_quantity[('Simulation parameters', 'Noise', 'Standard deviation')] = data['noise_scale']
            data_per_quantity[('Simulation parameters', 'Noise', 'Family')] = f"{data['noise_type']}{' (Z-scored)' if data['noise_zscore'] else ''}"
            data_per_quantity[('Simulation parameters', 'Noise', 'Structure')] = 'i.i.d' if data['noise_random'] else 'Structured multiplicative'

            data_per_quantity[('Simulation parameters', 'Bias', 'Demographic')] = data['bias_demographic']
            data_per_quantity[('Simulation parameters', 'Bias', 'Group')] = data['bias_group']
            data_per_quantity[('Simulation parameters', 'Bias', 'Relative overcounting (%)')] = 100*data['empirical_bias_value']
            data_per_quantity[('Simulation parameters', 'Bias', 'Bias scale')] = data['empirical_bias_scale']
            data_per_quantity[('Simulation parameters', 'Bias', 'Relative overcounting (%) -- Requested')] = 100*data['bias']
            data_per_quantity[('Simulation parameters', 'Bias', 'Bias scale -- Requested')] = data['bias_scale']

            # Re-index
            matrix_order = ['Perturbed', 'Inferred', 'Reduction (%)']
            data_series = pd.Series(data_per_quantity).sort_index()
            data_series = data_series.reindex(['Experiment', 'Simulation parameters', 'RMSE', 'Pearson'], level=0)
            data_series = data_series.reindex(
                geographies + [c for c in data_series.index.get_level_values(1).unique() if c not in geographies],
                level=1
            )
            data_series = data_series.reindex(
                matrix_order + [c for c in data_series.index.get_level_values(2).unique() if c not in matrix_order],
                level=2
            )

            per_file_series[quantity] = data_series

        return per_file_series

    except Exception as e:
        # Fail-safe: return None but keep going; you can also log e
        print(f"[WARN] Failed on {file}: {e}")
        return None


if __name__ == "__main__":  # ### [PARALLEL] guard for multiprocessing
    # ### [PARALLEL] run workers
    dfs = {q: [] for q in quantities}
    if len(files_available) == 0:
        print("No files to process.")
    else:
        # A small heuristic for chunksize keeps scheduling overhead low
        chunksize = max(1, len(files_available) // (args.n_workers * 4 or 1))
        with mp.Pool(processes=args.n_workers) as pool:
            for result in tqdm(pool.imap_unordered(_process_one_file, files_available, chunksize=chunksize),
                               total=len(files_available)):
                if result is None:
                    continue
                for q, series in result.items():
                    dfs[q].append(series)

    print('Saving files...')
    for quantity, quantity_dfs in tqdm(dfs.items()):
        if not quantity_dfs:
            continue
        #Concatenate:
        df = pd.concat(quantity_dfs,axis=1).T
        #Sort:
        experiment_settings_order = ['Year', 'Parameter configuration', 'Run #']
        sort_keys = [('Experiment', 'Setting', c) for c in experiment_settings_order]
        sorted_df = df.sort_values(sort_keys).reset_index(drop=True)
        #The output path:
        out_path = f'{vars._path_to_repo}/d05_outputs/d06_Synthetic/performance/{quantity}.csv'
        #Save if the file doesn't yet exist:
        if not os.path.exists(out_path):
            sorted_df.to_csv(out_path, index=False)
        #Load current file and append if file exists:
        else:
            current_df = pd.read_csv(out_path, header=[0,1,2])
            combined = pd.concat([current_df, sorted_df], ignore_index=True)
            combined = combined.sort_values(sort_keys).reset_index(drop=True)
            combined = combined.drop_duplicates(subset=sort_keys, keep='last').reset_index(drop=True)
            combined.to_csv(out_path, index=False)
