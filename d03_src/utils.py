#######################################################################################################################
# UTILS (misc. functions)

import ast
import scipy.sparse as ss
import pandas as pd
import numpy as np
from matplotlib import colormaps
import geopandas as gpd
import warnings

#######################################################################################################################

def get_descriptor(holdout_ACS_year=False,
                   retrieve_yearly_CBG_population=False, NNLS=False, use_2010_ACS=False,
                   holdout_CBG_t0=False, holdout_CBG_t1=False,
                   holdout_county_flows=False, holdout_state_flows=False,
                   holdout_state_nonmovers=False,
                   begin_with_nonmovers=False):
        
    descriptor = f'_holdout{holdout_ACS_year}' if holdout_ACS_year else ''
    if not holdout_CBG_t0:
        descriptor += '_match-CBGP0'
    if not holdout_CBG_t1:
        descriptor += '_match-CBGP1'
    if not holdout_county_flows:
        descriptor += '_match-county'
    if not holdout_state_flows:
        descriptor += '_match-state'
    if not holdout_state_nonmovers:
        descriptor += '_match-state-nonmovers'
        if begin_with_nonmovers:
            descriptor +='-first'
    if retrieve_yearly_CBG_population:
        descriptor += '_OLS' if not NNLS else '_NNLS'
        if use_2010_ACS:
            descriptor += '_wACS2010'

    return descriptor
    
#######################################################################################################################

def haversine_vectorized(latlon1, latlon2, unit='mile'):
    """
    Computes the great-circle distance between two sets of geographic coordinates using the Haversine formula.

    This function computes distances between pairs of coordinates in a fully vectorized manner using NumPy.
    Coordinates must be in degrees (latitude, longitude) and use WGS84 (`EPSG:4326`).

    Parameters
    ----------
    latlon1 : np.ndarray of shape (N, 2)
        First set of coordinates. Each row should be [latitude, longitude] in degrees.

    latlon2 : np.ndarray of shape (N, 2)
        Second set of coordinates. Must be the same shape as `latlon1`.

    unit : str, default='mile'
        Unit for the output distance.

    Returns
    -------
    np.ndarray of shape (N,)
        Array of distances between each pair of coordinates, in the specified unit.

    """
    #Collect the unit and convert to standard notation:
    unit_aliases = {'km': ['km', 'kilometer', 'kilometers'],
                    'mile': ['mile', 'miles'],
                    'm': ['m', 'meter', 'meters'],
                    'ft': ['ft', 'foot', 'feet'],
                    'in': ['in', 'inch', 'inches']}
    unit_aliases_rev = {alias: canonical for canonical, aliases in unit_aliases.items() for alias in aliases}
    unit = unit.lower()
    if unit not in unit_aliases:
        raise ValueError(f"Unsupported unit: '{unit}'. Accepted units include: {', '.join(unit_aliases.keys())}")

    #Collect earth's radius:
    radius_km = 6371.0
    conversion_factors = {'km': 1.0, 'm': 1000.0, 'mile': 0.621371, 'ft': 3280.84, 'in': 39370.1}
    canonical_unit = unit_aliases_rev[unit]
    R = radius_km * conversion_factors[canonical_unit]

    #Compute lat and long in radians:
    lat1 = np.radians(latlon1[:, 0])
    lon1 = np.radians(latlon1[:, 1])
    lat2 = np.radians(latlon2[:, 0])
    lon2 = np.radians(latlon2[:, 1])

    #use the formula:
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2

    #Convert to unit:
    distance = 2 * R * np.arcsin(np.sqrt(a))
    
    return distance

    
#######################################################################################################################

def add_user_params(default_params, user_params=None):
    """
    Updates a dictionary of default hyper-parameters
        with (if passed), user-defined hyper-parameters
    """
    
    if user_params is not None:
        default_params.update(user_params)
    user_params = default_params

    return user_params
    
#######################################################################################################################

def flat(A):
    """
    Flatten an array or csr array
    """
    try:
        A_arr = A.toarray() if ss.issparse(A) else np.asarray(A)
    except ValueError:
        A_arr = np.hstack(A)
    flat_A = A_arr.ravel()
    return flat_A

def density(A):
    """
    Density of sparse matrix
    """
    return A.getnnz()/np.prod(A.shape)

def replace_infinite(array):
    array[~np.isfinite(array)] = np.nan
    return array

#######################################################################################################################

def offdiagonal(A, order='C'):
    """
    Collect off-diagonal elements of a 
        matrix (either 2D or flattened)

    order : 'C' or 'F', order on which A was flattened
    """

    if A is None:
        return None
    
    #Collect the size of th array:
    N = A.shape[0]

    #If the array is already flattened, take square root:
    if len(A.shape)==1 or A.shape[1]==1: N = int(np.sqrt(N))

    #Get the diagonal indices as bool and reshape into A:
    diagonal_idx = np.eye(N, dtype=bool)
    diagonal_idx_A = diagonal_idx.reshape(A.shape, order=order)
    
    A_offdiag = A[np.where(~diagonal_idx_A)]

    return A_offdiag

def diagonal(A, order='C'):
    """
    Collect diagonal elements of a 
        matrix (either 2D or flattened)

    order : 'C' or 'F', order on which A was flattened
    """

    if A is None:
        return None
        
    #Collect the size of th array:
    N = A.shape[0]

    #If the array is already flattened, take square root:
    if len(A.shape)==1 or A.shape[1]==1: N = int(np.sqrt(N))

    #Get the diagonal indices as bool and reshape into A:
    diagonal_idx = np.eye(N, dtype=bool)
    diagonal_idx_A = diagonal_idx.reshape(A.shape, order=order)
    
    A_diag = A[np.where(diagonal_idx_A)]

    return A_diag
    
#######################################################################################################################

def offdstr(off_diagonal_boolean):
    """
    Converts a boolean value to a descriptor of entries
        used in the optimization cost
    """
    assert off_diagonal_boolean in [True, False]
    
    if off_diagonal_boolean:
        return 'only movers'
        
    else:
        return 'all entries'
        
#######################################################################################################################

def get_colors(groups, cmap='viridis', start=.25, stop=.75):

    if type(cmap) == str:
        cmap = colormaps[cmap]

    #Get evenly spaced points:
    X = np.linspace(start, stop, num=len(groups), endpoint=True)

    #Assign colors:
    colors = [cmap(x) for x in X]
    colors_dict = dict(zip(groups, colors))

    return colors_dict

#######################################################################################################################

def grouped_weighted_average(df, group_col, weight_col, aggregation_col):
    """
    Weighted average per group on a pd.DataFrame
    """

    wm = lambda x: np.average(x, weights=df.loc[x.index, weight_col])
    grouped_df = df.groupby(group_col)
    aggregation = grouped_df[aggregation_col].agg(wm)

    return aggregation

#######################################################################################################################
def adjust_zeros(L):
    """
    Replace first element of a list of probabilities with zero if they are all zero
    """
    if all(frac == 0 for frac in L): L[0] = 1
    return L

#######################################################################################################################

def RMSE(X, Y, w=None):
    return np.sqrt(np.average((X-Y)**2, weights=w))

#######################################################################################################################