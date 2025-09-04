############################################################################################
# Functions for the synthetic validation:
############################################################################################
import numpy as np
import pandas as pd
import scipy.sparse as ss
from scipy.optimize import root_scalar

import sys
sys.path.append('../d03_src/')
import optimization as opt

############################################################################################

def generate_synthetic_infutor(M, C_dict=None,
                               noise_type='lognormal', noise_mean=1., noise_scale=0.25, random_noise=True, zscore=False,
                               bias=0., b=0., bias_weights=None):
    """
    Generate synthetic Infutor data by adding noise to a given migration matrix M.

    Parameters
    ----------
    M : csr sparse matrix
        CBG - CBG migration matrix
    noise_mean : float, optional
        Mean of the lognormal noise to be added, by default 1. (centered at the original values)
    noise_sd : float, optional
        Standard deviation of the lognormal noise to be added, by default 0.25.
        
    Returns
    ----------
    csr sparse matrix
        Synthetic Infutor data
    b float
        Bias scale
    bias float
        Bias
    """
    #Start with the original matrix:
    E = M.copy()
    
    #Multiply by lognormal noise:
    if noise_type.lower() == 'lognormal':
        #If noise is i.i.d.:
        if random_noise:
            rng = np.random.default_rng()
            multipliers = rng.lognormal(mean=np.log(noise_mean), sigma=noise_scale, size=E.nnz) if noise_scale > 0 else noise_mean
            E.data *= multipliers
        #Otherwise, impose the scaling structure in our 5 scalings:
        else:
            assert C_dict is not None, 'To impose non-random noise, must define aggregation matrices'
            E = add_structured_noise(E, C_dict, noise_mean=noise_mean, noise_scale=noise_scale) if noise_scale > 0 else E*(noise_mean**5)
    #Negative binomial noise:
    elif noise_type.lower() == 'negbin':
        print('Negative binomial noise not implemented, returning original matrix.')
    else:
        print('Noise must be one of `LogNormal` or `NegBin`, please adjust input.')

    #Z-score:
    if zscore:
        original_diagonal_values = E.diagonal()
        sparsity_pattern = ss.csr_matrix((np.ones_like(E.data), E.indices, E.indptr), shape=E.shape)

        #Get the summary statistics:
        summary_statistics = {}
        for matrix_name, matrix in {'E':E, 'M':M}.items():
            #Get diagonal and off diagonal values:
            diagonal_values = matrix.diagonal()
            offdiagonal_values = (matrix - ss.diags(diagonal_values)).data
            #Compute:
            for values_name, values in {'diagonal':diagonal_values, 'offdiagonal':offdiagonal_values}.items():
                summary_statistics[('mean', matrix_name, values_name)] = values.mean()
                summary_statistics[('sd',   matrix_name, values_name)] = values.std()

        #First rescale the offdiagonal:
        sd_scaler = summary_statistics[('sd', 'M', 'offdiagonal')]/summary_statistics[('sd', 'E', 'offdiagonal')]
        E = sd_scaler*E + (summary_statistics[('mean', 'M', 'offdiagonal')] - sd_scaler*summary_statistics[('mean', 'E', 'offdiagonal')])*sparsity_pattern

        #Now set the diagonal:
        sd_scaler = summary_statistics[('sd', 'M', 'diagonal')]/summary_statistics[('sd', 'E', 'diagonal')]
        E.setdiag(sd_scaler*(original_diagonal_values-summary_statistics[('mean', 'E', 'diagonal')]) + summary_statistics[('mean', 'M', 'diagonal')])

    #Incur bias along a certain population:
    if bias > 0:
        assert bias_weights is not None, 'If adding bias, must specify weights per row'
        #Get b:
        b = b_for_beta(1+bias, M, E, bias_weights)
        if b is None:
            print('This bias level could not be achieved')
            return None, None, None
    if b > 0:
        E, bias = rescale_given_b(b, E, bias_weights=bias_weights, M=M)

    return E, b, bias

def add_structured_noise(M, C_dict, noise_mean=1., noise_scale=0.25):
    """
    Add multiplicative noise 5x to each entry, mirroring our scalers:

    (1) Noise according to rows (CBG populations)
    (2) Noise according to columns at state level, on/off-diagonal (State movers)
    (3) Noise according to state-state pairs (State flows)
    (4-5) Noise according to rows and columns at county level (County populations)
    """
    assert noise_scale > 0.
    rng = np.random.default_rng()

    #(1) Add noise per CBG to row populations:
    D = ss.diags(rng.lognormal(mean=np.log(noise_mean), sigma=noise_scale, size=M.shape[0])) #CBG scalers
    E = D @ M

    #(2) Add noise per state to diagonal and off-diagonal elements (non-movers/movers):
    C = C_dict[('blockgroup', 'state')]
    new_diagonal = E.diagonal() * (C @ rng.lognormal(mean=np.log(noise_mean), sigma=noise_scale, size=C.shape[1]))
    E = E @ ss.diags(C @ rng.lognormal(mean=np.log(noise_mean), sigma=noise_scale, size=C.shape[1])) #off-diagonal
    E.setdiag(new_diagonal)                                                                          #diagonal

    #(3) Add noise per state-state pair (flows):
    multipliers = rng.lognormal(mean=np.log(noise_mean), sigma=noise_scale, size=(C.shape[1],C.shape[1]))
    E = opt.scale_checkerboard_matrix(E, multipliers, C)

    #(4-5) Add noise per county to rows and columns:
    C = C_dict[('blockgroup', 'county')]
    D = ss.diags(C @ rng.lognormal(mean=np.log(noise_mean), sigma=noise_scale, size=C.shape[1])) #county scalers
    E = (D @ E) @ D
    
    return E

def rescale_given_b(b, E, bias_weights, M=None):
    """
    Compute the population bias given a logarithmic
        scaling factor b and weights, and rescale
        the matrix E.
    """
    #If we didn't pass a ground-truth matrix, assume the same as E:
    if M is None:
        M = E.copy()

    #Get the standardized weights and exponentiate:
    bias_weights_z = (bias_weights - np.mean(bias_weights))/np.std(bias_weights)
    exp_weights = np.exp(b*bias_weights_z)

    #Scale:
    K = ss.diags(exp_weights)
    E = (K @ E) @ K

    #Bring back to original population:
    alpha = M.sum()/E.sum()
    E = alpha * E

    #Compute the bias:
    pop_E = np.array(E.sum(axis=1)).flatten()*bias_weights
    pop_M = np.array(M.sum(axis=1)).flatten()*bias_weights
    bias = pop_E.sum()/pop_M.sum() - 1

    return E, bias

def b_for_beta(beta_star, M, E, w, tol=1e-6):
    """Invert the function numerically to find b(beta)"""
    #Hard coded no-bias scenario:
    if abs(beta_star - 1.0) <= tol:
        return 0.0
    #Define a function to get bias robust to numerical errors:
    def f(b):
        # Safe oracle: never raise, never return None
        with np.errstate(all='ignore'):
            beta = rescale_given_b(b=b, M=M, E=E, bias_weights=w)[1]
        return (beta - beta_star) if np.isfinite(beta) else np.nan
    #Try to find roots with scipy:
    try:
        sol = root_scalar(f, x0=0.0, x1=1.0, method="secant", xtol=tol, maxiter=50)
    except Exception:
        return None
    if not getattr(sol, "converged", False) or not np.isfinite(sol.root):
        return None

    #Sanity check at the returned root
    with np.errstate(all='ignore'):
        beta_at = rescale_given_b(b=sol.root, M=M, E=E, bias_weights=w)[1]
    return sol.root if np.isfinite(beta_at) and abs(beta_at - beta_star) <= tol * max(1.0, beta_star) else None

############################################################################################

def collect_population_sums(M, C=None):
    """
    Collect the current population sums from the migration matrix M.

    Parameters
    ----------
    M : csr sparse matrix
        Migration matrix.
    C : csr sparse matrix, optional
        Coarse geography matrix, by default None.

    Returns
    -------
    tuple of np.ndarray
        Population sums for rows and columns.
    """

    #Collect current population sums:
    P0 = np.array(M.sum(axis=1)).flatten()
    P1 = np.array(M.sum(axis=0)).flatten()

    #Aggregate:
    if C is not None:
        P0 = C.T @ P0
        P1 = C.T @ P1

    return P0, P1

def collect_population_flows(M, C):
    """
    Collect the current population flows from the migration matrix M.

    Parameters
    ----------
    M : csr sparse matrix
        Migration matrix.
    C : csr sparse matrix
        Coarse geography matrix.

    Returns
    -------
    csr sparse matrix
        Aggregated migration matrix.
    """
    return C.T @ M @ C

def collect_population_movers(M, C=None):
    """
    Collect the current population non-movers from the migration matrix M.

    Parameters
    ----------
    M : csr sparse matrix
        Migration matrix.
    C : csr sparse matrix
        Coarse geography matrix.

    Returns
    -------
    np.ndarray, np.ndarray
        Population non-movers, Population movers.
    """
    #Collect non-movers:
    non_movers = C.T @ M.diagonal() if C is not None else M.diagonal()

    #Collect movers by discounting non-movers from the population:
    _, population = collect_population_sums(M, C=C)
    movers = population - non_movers

    return non_movers, movers

############################################################################################

def IPF_update(M, m, m_type, C):
    """
    Does one IPF update, for m_type == 'row' or 'column'
    """
    assert m_type.lower() in ['row', 'column']
    
    #First, get current sums:
    current_sums = opt.get_IPF_current_values(M, m_type, C)
    
    #Second, we get the scalers:
    aggregated_scalers = opt.get_IPF_scaling(current_sums, m, ignore_zeros=True, tolerance=None, verbose=False)
    
    #Third, cast scalers to the dimensions of M and multiply (varies with constraint type!)
    scalers = C @ aggregated_scalers
    S = ss.diags(scalers)
    updated_M = M @ S if m_type.lower() == 'column' else S @ M
    
    return updated_M

############################################################################################

def collect_tables(performance_df,
                   noise_family=None, noise_structure=None, bias_knob='Bias scale'):
    """
    Collect average and s.e.m. for Pearson corr. and RMSE reduction on synthetic experiments,
        across all years and by noise/bias levels.
    """
    #Grab df and mask:
    mask = pd.Series(True, index=performance_df.index)
    if noise_family is not None:
        mask &= performance_df[('Simulation parameters', 'Noise', 'Family')] == noise_family
    if noise_structure is not None:
        mask &= (performance_df[('Simulation parameters', 'Noise', 'Structure')] == noise_structure)
    #Group by parameter configuration:
    grouped_df = performance_df[mask].groupby([('Experiment', 'Setting', 'Parameter configuration')])
    #Collect the parameter values:
    parameters = grouped_df['Simulation parameters'].first()
    noise = parameters[('Simulation parameters', 'Noise', 'Standard deviation')].sort_index()
    bias  = parameters[('Simulation parameters', 'Bias',  bias_knob)].sort_index()
    #Ensure the parameter configuration is well-defined:
    _parameters = grouped_df['Simulation parameters'].last()
    assert (noise == _parameters[('Simulation parameters', 'Noise', 'Standard deviation')].sort_index()).all(), 'Parameter configuration not well-defined'
    assert (bias  == _parameters[('Simulation parameters', 'Bias',  bias_knob)].sort_index()).all(), 'Parameter configuration not well-defined'
    #Aggregate:
    metric_summary = {'Mean':{'RMSE reduction (%)':  grouped_df['RMSE'   ].mean().xs( 'Reduction (%)', axis=1, level=2)['RMSE'],
                              'Pearson correlation': grouped_df['Pearson'].mean().xs( 'Inferred',      axis=1, level=2)['Pearson']},
                      'SEM': {'RMSE reduction (%)':  grouped_df['RMSE'   ].sem( ).xs( 'Reduction (%)', axis=1, level=2)['RMSE'],
                              'Pearson correlation': grouped_df['Pearson'].sem( ).xs( 'Inferred',      axis=1, level=2)['Pearson']},
                      'std': {'RMSE reduction (%)':  grouped_df['RMSE'   ].std( ).xs( 'Reduction (%)', axis=1, level=2)['RMSE'],
                              'Pearson correlation': grouped_df['Pearson'].std( ).xs( 'Inferred',      axis=1, level=2)['Pearson']}}
    #Pivot:
    tables = {}
    for summary, metric_dict in metric_summary.items():
        tables[summary] = {}
        for metric, metric_df in metric_dict.items():
            metric_df['Noise scale'] = metric_df.index.map(noise)
            metric_df[bias_knob]     = metric_df.index.map(bias)
            tables[summary][metric]  = {geography: metric_df.pivot_table(index='Noise scale', columns=bias_knob, values=geography, sort=True)
                                        for geography in ['blockgroup', 'tract', 'county', 'state']}
    return tables