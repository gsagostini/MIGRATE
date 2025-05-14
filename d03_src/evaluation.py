############################################################################################
# Functions to evaluate estimates against other datasets
#     1. Functions to evaluate population count agreements
#     2. Functions to evaluate flow agreements
#     3. Functions to evaluate population agreement per demographic group

############################################################################################

import numpy as np
import pandas as pd
from tqdm import tqdm, trange
import scipy.sparse as ss

import sys
sys.path.append('../d03_src/')
import vars
import os

from utils import RMSE, flat, offdiagonal, haversine_vectorized

############################################################################################
#1. Evaluating pipeline:

def validation(quantity,
               estimates, metrics, data_for_validation,
               C_matrices,
               ground_truth_ready=False, rate=False, rate_col=False,
               full_data_range=range(2010,2020), estimates_range=range(2011,2020),
               weight_by_population=False, weight_col=False):
    """
    Computes the validation for a specific quantity (population, all flows,
        migration rates) and estimates.

    Parameters
    ----------
    quantity : `population`, `flows`, `flows-stayers`, `flows-outmigration`, `flows-inmigration`
        what summary of the estimates to validate
        
    estimates : dict
        dictionary with {dataset:{year:matrix}} structure
        
    metrics : dict
        dictionary with {metric:function}, where function takes two arguments
        
    data_for_validation : dict
        dictionary with {(data_source, data_geography):(values, data_period)}
        
    C_matrices : dict
        dictionary with membership matrices
        
    Returns
    ----------
    pd.DataFrame
        multiIndex DataFram whose index are years in the full_data_range and
        columns are indexed by metric, data source, and estimate.
    
    """
    #Verify input:
    quantity = quantity.lower()
    assert quantity in ['population', 'flows', 'stayers', 'movers', 'out-migration', 'in-migration', 'net-migration']

    validation_dict = {}

    #We do the comparison for every dataset released:
    for ground_truth_year in tqdm(full_data_range):
        validation_dict[ground_truth_year] = {}
        
        #Iterate over data sources:
        for (data_source, geography), (all_ground_truth, data_period_in_years) in data_for_validation.items():
            
            #Initialize dictionaries:
            data_string = f'{data_source} {data_period_in_years}-year {geography} {quantity}'
            validation_dict[ground_truth_year][data_string] = {}
            for dataset, all_estimates in estimates.items():
                validation_dict[ground_truth_year][data_string][dataset] = {m:np.nan for m in metrics}

            if ground_truth_year in all_ground_truth:
                #First, collect the ground truth data:
                ground_truth = all_ground_truth[ground_truth_year]
                ground_truth_coverage = range(ground_truth_year-data_period_in_years+1, ground_truth_year+1)
                C = C_matrices[('blockgroup', geography)] if geography != 'blockgroup' else None
    
                #Extract the quantity from ground truth data, if needed:
                true_quantity = extract_summary(ground_truth, quantity=quantity, rate=rate, normalize_by_column=rate_col) if not ground_truth_ready else ground_truth
    
                #The period of estimates will depend if we are collecting flows or populations:
                delta = 1 if quantity == 'population' else 0  #Use y-1 to correspond to row-population of the matrices!
                years_to_average = [y for y in estimates_range if y-delta in ground_truth_coverage]
                
                #Then, collect MIGRATE and Infutor matrices:
                for dataset, all_estimates in estimates.items():
                    
                    #We will average ang aggregate the matrices if any:
                    if len(years_to_average):
                        averaged_matrix = sum([all_estimates[y] for y in years_to_average])/len(years_to_average)
                        aggregated_matrix = C.T @ averaged_matrix @ C if C is not None else averaged_matrix
                        
                        #Extract the appropriate quantity from the matrix:
                        estimate_quantity = extract_summary(aggregated_matrix, quantity=quantity, rate=rate, normalize_by_column=rate_col)

                        #Flatten and crop to valid entries:
                        x, y = flat(estimate_quantity), flat(true_quantity)
                        entries = np.isfinite(x)*np.isfinite(y)
                        
                        #If we want to return weights, compute the appropriate population:
                        if weight_by_population:
                            population = extract_summary(aggregated_matrix, quantity='population_col' if weight_col else 'population')
                            weights = flat(population)
                            w = weights[entries]

                        #Compute the metrics:
                        for metric_name, metric_function in metrics.items():
                            rho = metric_function(x[entries], y[entries], w=w if weight_by_population else None)
                            validation_dict[ground_truth_year][data_string][dataset][metric_name] = rho
    
    #Return as multi-index DataFrame:
    df = pd.DataFrame.from_dict({(i, j, k): validation_dict[i][j][k] 
                                 for i in validation_dict.keys() 
                                 for j in validation_dict[i].keys() 
                                 for k in validation_dict[i][j].keys()}, 
                                orient='index')
    df.index = pd.MultiIndex.from_tuples(df.index)
    df = df.unstack(level=[1, 2])
    return df

############################################################################################
#2. Evaluating summary from flow estimates:
    
def extract_summary(flow_matrix, quantity='population', rate=False, normalize_by_column=False):
    """
    Extracts the appropriate quantity from a flow matrix
    """
    #Verify input:
    quantity = quantity.lower()
    assert quantity in ['population', 'population_col', 'flows', 'stayers', 'movers', 'out-migration', 'in-migration', 'net-migration']
    if rate: assert quantity in ['out-migration', 'in-migration', 'net-migration']

    #For population, do row sum:
    if quantity == 'population':
        summary = np.array(flow_matrix.sum(axis=1)).flatten()

    #For population (column), do column sum:
    if quantity == 'population_col':
        summary = np.array(flow_matrix.sum(axis=0)).flatten()

    #For flows, ignore:
    elif quantity == 'flows':
        summary = flow_matrix

    #For stayers:
    elif quantity == 'stayers':
        summary = flow_matrix.diagonal()

    #For movers:
    elif quantity == 'movers':
        summary = offdiagonal(flow_matrix)

    #For out-migration:
    elif quantity == 'out-migration':
        summary = np.array(flow_matrix.sum(axis=1)).flatten() - flow_matrix.diagonal()
        
    #For in-migration:
    elif quantity == 'in-migration':
        summary = np.array(flow_matrix.sum(axis=0)).flatten() - flow_matrix.diagonal()
        
    #For net-migration:
    elif quantity == 'net-migration':
        summary = np.array(flow_matrix.sum(axis=0)).flatten() - np.array(flow_matrix.sum(axis=1)).flatten()

    #If we want a rate, normalize:
    if rate:
        denominator = np.array(flow_matrix.sum(axis=0)).flatten() if normalize_by_column else np.array(flow_matrix.sum(axis=1)).flatten()
        summary = summary/denominator

    return summary

############################################################################################
#3. Getting flattened quantities for scatter plots:

def get_flattened_data(quantity,
                       estimates_yearly_dict,
                       ground_truth_yearly_dict,
                       ground_truth_span_in_years,
                       aggregation_matrix=None,
                       weight_by_population=False, weight_col=False,
                       ground_truth_ready=False, rate=False, rate_col=False):    
    """
    Function that retrieves two arrays of values---estimates and ground-truth values---for a
        series of years, all flattened in order.
    """

    #Verify input:
    quantity = quantity.lower()
    assert quantity in ['population', 'flows', 'stayers', 'movers', 'out-migration', 'in-migration', 'net-migration']

    #Initialize empty arrays:
    estimate_arr, data_arr, years_arr, weight_arr  = [], [], [], []

    #Iterate over years of published ground-truth data:
    for data_year, data in ground_truth_yearly_dict.items():
        #Extract the quantity from ground truth data, if needed:
        data_quantity = extract_summary(data, quantity=quantity, rate=rate, normalize_by_column=rate_col) if not ground_truth_ready else data
        if not ss.issparse(data_quantity): data_entries = np.isfinite(data_quantity)
        #The period of estimates will depend if we are collecting flows or populations:
        ground_truth_coverage = range(data_year-ground_truth_span_in_years+1, data_year+1)
        delta = 1 if quantity == 'population' else 0  #Use y-1 to correspond to row-population of the matrices!
        #Extract_the quantity from averaged estimates, if any in the period:
        estimates_to_average = [matrix for y,matrix in estimates_yearly_dict.items() if y-delta in ground_truth_coverage]
        if len(estimates_to_average):
            averaged_matrix = sum(estimates_to_average)/len(estimates_to_average)
            aggregated_matrix = aggregation_matrix.T @ averaged_matrix @ aggregation_matrix if aggregation_matrix is not None else averaged_matrix
            estimate_quantity = extract_summary(aggregated_matrix, quantity=quantity, rate=rate, normalize_by_column=rate_col)
            if not ss.issparse(estimate_quantity): estimate_entries = np.isfinite(estimate_quantity)
            #Include both flattened quantities in the arrays:
            flat_estimate, flat_data = flat(estimate_quantity), flat(data_quantity)
            if not ss.issparse(estimate_quantity):
                entries = flat(data_entries)*flat(estimate_entries)
                flat_estimate, flat_data = flat_estimate[entries], flat_data[entries]
            assert len(flat_estimate) == len(flat_data), 'Missing entries in data!'
            #If we want to return weights, compute the appropriate population:
            if weight_by_population:
                population = extract_summary(aggregated_matrix, quantity='population_col' if weight_col else 'population')
                flat_population = flat(population)[entries]
            #Log:
            estimate_arr.append(flat_estimate)
            data_arr.append(flat_data)
            years_arr.append([data_year]*len(flat_data))
            weight_arr.append(flat_population if weight_by_population else [1]*len(flat_data))

    return flat(estimate_arr), flat(data_arr), flat(years_arr), flat(weight_arr)

############################################################################################
#4. Estimates and CIs for the RMSE reduction:
def get_RMSE_reductions(df, use_mean=True, use_sem=False):
    """
    Collect RMSE reductions and confidence intervals
    """
    #Select RMSE:
    df = df['RMSE']
    reductions = df.loc[:,df.columns.get_level_values(1) == 'Reduction (%)'].droplevel(level=1, axis=1)
    #Compute estimates and confidence intervals:
    est = reductions.mean() if use_mean else reductions.median()
    CIs = 1.96*reductions.std()/np.sqrt(reductions.shape[0]) if use_sem else reductions.std()
    return est, CIs


############################################################################################
#5. Classifying moves:
def average_demographic(M, values, movers_only=True, fill_nan=False, fill_nan_value=0):
    #Process the matrix:
    if movers_only: M -= ss.diags(M.diagonal())
    total_per_CBG = np.array(M.sum(axis=1)).flatten()
    valid_CBGs = total_per_CBG > 0

    #Average (only on CBGs with non-zero total!)
    aggregated_demographics = M @ values
    average_demographics = np.divide(aggregated_demographics, total_per_CBG,
                                     out=np.full_like(aggregated_demographics, fill_nan_value if fill_nan else np.nan),
                                     where=valid_CBGs)
    
    return average_demographics

def get_demographic_groups(demographics_df,
                           income_q=4, income_labels=['1st income quartile', '2nd income quartile', '3rd income quartile', '4th income quartile']):
    """
    Collects a dictionary of demographic groups at the CBG level
    """
    #Verify input:
    if len(income_labels) != income_q:
        income_labels=range(1, income_q+1)
    
    #Determine CBG groups based on these demographics:
    grouping_by = {'Race': 'Plurality ' + demographics_df['Race'].idxmax(axis=1).values,
                   'Urbanization': demographics_df['Urbanization'].idxmax(axis=1).values,
                   'Income': pd.qcut(demographics_df['Household Income']['Median income'], q=income_q, labels=income_labels).values}
    CBG_indices = {}
    for demographic_class, grouping in grouping_by.items():
        CBG_indices = CBG_indices|{group: np.where(grouping == group)[0] for group in np.unique(grouping)}

    return CBG_indices

def count_moves_within_CBGs(flow_matrix, CBG_indices, movers_only=True, name=None):
    """
    Collects a series counting moves from every CBG in a flow matrix
        to all CBGs of a given list of indices. That is, aggregates an N x N 
        flow matrix into N x 1, adding all movers whose destination fits in
        the group given by `CBG_indices`.
    """

    # Ensure CSR format to avoid duplicate entries in .data/.nonzero()
    if not ss.isspmatrix_csr(flow_matrix):
        flow_matrix = flow_matrix.tocsr(copy=False)
        
    #Remove diagonal from the matrix if using only movers (recommended):
    if movers_only: flow_matrix -= ss.diags(flow_matrix.diagonal())

    #Now filter:
    movers = np.array(flow_matrix[:,CBG_indices].sum(axis=1)).flatten()

    #Return as a Series:
    movers_series = pd.DataFrame(movers)
    if name is not None: movers_series.columns = [name]
    return movers_series

def count_moves_by_group(flow_matrix, groups_dict, movers_only=True):
    """
    Collects a dataframe counting moves from every CBG in a flow matrix
        to all CBGs of a given group. That is, aggregates an N x N 
        flow matrix into N x M, where M is the number of different groupings
        of CBGs (which are not necessairily exclusive).
    """

    # Ensure CSR format to avoid duplicate entries in .data/.nonzero()
    if not ss.isspmatrix_csr(flow_matrix):
        flow_matrix = flow_matrix.tocsr(copy=False)
        
    #Remove diagonal from the matrix if using only movers (recommended):
    if movers_only: flow_matrix -= ss.diags(flow_matrix.diagonal())

    #Now iterate over groups:
    movers_by_group = {}
    for CBG_group, CBG_indices_in_group in groups_dict.items():
        movers_by_group[CBG_group] = np.array(flow_matrix[:,CBG_indices_in_group].sum(axis=1)).flatten()

    #Return as a DataFrame:
    movers_by_group_df = pd.DataFrame(movers_by_group)
    return movers_by_group_df

def count_moves_by_distance(flow_matrix, geography_matrix_dictionary,
                            geographies=['tract', 'county', 'state']):
    """
    Collects a dataframe counting moves within each geography
    """

    # Ensure CSR format to avoid duplicate entries in .data/.nonzero()
    if not ss.isspmatrix_csr(flow_matrix):
        flow_matrix = flow_matrix.tocsr(copy=False)
        
    moves_by_distance = {}

    #Start with stayers (i.e. flows within Census Block Groups):
    flows_within = flow_matrix.diagonal()
    moves_by_distance['Not moving'] = flows_within

    #Now iterate over geographies:
    for geography in geographies:
        
        #Get a matrix representing within-geography entries:
        C = geography_matrix_dictionary[('blockgroup', geography)]
        C_within = C @ C.T

        #Multiply the within geography matrix with the flow matrix:
        flow_matrix_within = flow_matrix.multiply(C_within)
        new_flows_within = np.array(flow_matrix_within.sum(axis=1)).flatten()

        #Record, discarding flows within previously recorded geography, and update:
        moves_by_distance[f'Moving within {geography}'] = new_flows_within - flows_within
        flows_within = new_flows_within

    #We also record the missing flows as flows out of the last geography provided:
    population = np.array(flow_matrix.sum(axis=1)).flatten()
    moves_by_distance[f'Moving out of {geographies[-1]}'] = population - new_flows_within

    #Return as a DataFrame:
    movers_by_distance_df = pd.DataFrame(moves_by_distance)
    return movers_by_distance_df

def get_base_rates(demographic_groups, N=None, weights=None):

    #Fill weights and total:
    if weights is None: weights = np.ones(N) 
    N = np.sum(weights)

    #Compute average:
    share_in_group = {group: np.sum(weights[indices])/N for group, indices in demographic_groups.items()}

    return pd.Series(share_in_group)
    

def count_moves_by_haversine_distance(flow_matrix, centroids_latlon,
                                      distance_bins=[1, 5, 10, 50, 100], distance_unit='mile', include_distance_unit=False,
                                      return_weighted_average=False):
    """
    Collects a dataframe counting moves into distance bins using Haversine distance.

    Parameters
    ----------
    flow_matrix : csr_matrix
        Sparse origin-destination matrix (N x N), where entry (i, j) is number of moves from CBG i to j.

    centroids_latlon : np.ndarray of shape (N, 2)
        Array of [latitude, longitude] pairs for each CBG in EPSG:4326.

    distance_bins : list of float, default=[1, 5, 10, 50, 100]
        Distance bin edges (in units specified by `distance_unit`).
        An open-ended final bin will be added automatically.

    distance_unit : str, default='mile'
        Unit to compute Haversine distances. Can be 'mile', 'km', 'm', 'ft', 'in', etc.

    return_weighted_average : bool, default=False
        Whether to return a NumPy array of average move distances per origin CBG, weighted by move counts.

    Returns
    -------
    pd.DataFrame or (pd.DataFrame, np.ndarray)
        DataFrame of shape (N, K), where each column is a distance bin and each row corresponds to a CBG.
        If return_weighted_average is True, also returns a NumPy array of shape (N,) with weighted mean distances.
    """
    # Ensure CSR format to avoid duplicate entries in .data/.nonzero()
    if not ss.isspmatrix_csr(flow_matrix):
        flow_matrix = flow_matrix.tocsr(copy=False)

    #Adjust so that the last bin catches all:
    if not np.isinf(distance_bins[-1]): distance_bins = distance_bins + [np.inf]

    #Extract values from the matrix:
    row_idx, col_idx = flow_matrix.nonzero()
    weights = flow_matrix.data

    #Compute distances:
    distances = haversine_vectorized(centroids_latlon[row_idx],
                                     centroids_latlon[col_idx], unit=distance_unit)

    #Generate human-readable bin labels
    unit_name = ' '+ distance_unit if distance_unit.endswith('s') else distance_unit + 's'
    labels = [f"less than {distance_bins[0]}{unit_name if include_distance_unit else ''}"]
    for i in range(len(distance_bins) - 2):
        labels.append(f"{distance_bins[i]} to {distance_bins[i+1]}{unit_name if include_distance_unit else ''}")
    labels.append(f"more than {distance_bins[-2]}{unit_name if include_distance_unit else ''}")

    #Compute bins:
    N = flow_matrix.shape[0]
    bin_idx = np.digitize(distances, distance_bins)
    moves_by_bin = {label: np.zeros(N) for label in labels}
    total_distance = np.zeros(N) if return_weighted_average else None
    total_weight = np.zeros(N) if return_weighted_average else None

    #Compute distance on every row:
    for i in range(len(row_idx)):
        origin = row_idx[i]
        label = labels[bin_idx[i]]
        moves_by_bin[label][origin] += weights[i]

        if return_weighted_average:
            total_distance[origin] += distances[i] * weights[i]
            total_weight[origin] += weights[i]
    result_df = pd.DataFrame(moves_by_bin)

    #Compute weighted average if needed:
    if return_weighted_average:
        with np.errstate(divide='ignore', invalid='ignore'):
            avg_dist = np.divide(
                total_distance, total_weight,
                out=np.full_like(total_distance, np.nan),
                where=total_weight > 0
            )
        return result_df, avg_dist

    return result_df

def count_moves_upwards(flow_matrix, values,
                        threshold=0, 
                        movers_only=True,
                        outmovers=True):
    """
    Collects a dataframe counting upwards out-moves from every CBG in a
        flow matrix along a given value. That is, how many out-moves
        per CBG resulted in a (significant) increase on a certain value.
    """
    # Ensure CSR format to avoid duplicate entries in .data/.nonzero()
    if not ss.isspmatrix_csr(flow_matrix):
        flow_matrix = flow_matrix.tocsr(copy=False)

    #Remove diagonal from the matrix if using only movers (recommended):
    if movers_only: flow_matrix -= ss.diags(flow_matrix.diagonal())

    #Extract flow matrix as data for this computation to be efficient:
    row, col = flow_matrix.nonzero()
    data = flow_matrix.data

    #Find what flows were upwards in the values of interest:
    origin_values = values[row]  #median incomes of all origins (i.e. in all pairs)
    destin_values = values[col]  #median incomes of all destinations (i.e. in all pairs)
    mask = destin_values - origin_values > origin_values * threshold

    #Select upward flows:
    upward_flows_per_CBG = np.bincount(row[mask] if outmovers else col[mask], weights=data[mask], minlength=flow_matrix.shape[0])

    return upward_flows_per_CBG
    