############################################################################################
# Functions to construct and solve the optimization problem:
############################################################################################
import numpy as np
import pandas as pd
import scipy.sparse as ss
import geopandas as gpd

from tqdm import tqdm, trange

import sys
sys.path.append('../d03_src/')
import utils
import vars
import process_census as prc

############################################################################################

def get_IPF_current_values(current_values,
                           constraint_type,
                           aggregation_matrix):
    """
    Get the current value of a given constraint
        
    Parameters
    ----------
    current_values : np.Array or ss.csc_matrix of shape N x M
        two-dimensional array containing the current values of the entries
    """
    assert constraint_type.lower() in ['row', 'column', 'checkerboard']
    if constraint_type.lower() == 'row':
        current_sums = aggregation_matrix.T @ np.array(current_values.sum(axis=1)).ravel()
    elif constraint_type.lower() == 'column':
        current_sums = aggregation_matrix.T @ np.array(current_values.sum(axis=0)).ravel()
    else:
        current_sums = aggregation_matrix.T @ current_values @ aggregation_matrix
    return current_sums

def get_IPF_scaling(current_values,
                    target_values,
                    ignore_zeros=False,
                    tolerance=None,
                    verbose=False):
    """
    Get a series of IPF scalers roughly determined by
        scalers = target_values/current_values
        
    Parameters
    ----------
    current_values : np.Array or ss.csc_matrix of shape N x M
        two-dimensional array containing the current values of the entries
        
    target_values : np.Array or ss.csc_matrix of shape N x M
        two-dimensional array containing the target values of the entries
        
    ignore_zeros : Bool, default False
        if True, avoids zero-ing entries in current values (because the
        target value is zero). The default behavior is to keep these 
        entries unchanged.
        
    tolerance : np.Array or ss.csc_matrix of shape N x M or None
        if not None and ignore_zeros is True, avoids zero-ing entries in
        current values but brings large values down to the margin of
        tolerance (if value is already below tolerance, leave unchanged).
        
    verbose : Bool, default False
    
    Returns
    ----------
    np.Array of shape N x M
        matrix of scalers
    """
    
    #Create default scalers:
    scalers = np.ones(target_values.shape)

    #Find indices where the scaler is defined:
    idx_to_update = np.nonzero(current_values)
    scalers[idx_to_update] = target_values[idx_to_update]/current_values[idx_to_update]

    #If we are ignoring zero scalers:
    if ignore_zeros:
        if verbose: print(f'{np.sum(scalers<=0)} zero scalers')
        #If we pass a valid MoE, turn some in MoE:
        if tolerance is not None:
            assert tolerance.shape == target_values.shape, "Tolerance has invalid shape"
            idx_to_update_with_tolerance = (scalers<=0)&(current_values > tolerance)
            tolerance_scalers = np.ones(target_values.shape)
            tolerance_scalers[idx_to_update] = tolerance[idx_to_update]/current_values[idx_to_update]
            scalers[idx_to_update_with_tolerance] = tolerance_scalers[idx_to_update_with_tolerance]
            if verbose: print(f'{np.sum(idx_to_update_with_tolerance)} zero scalers where value was above tolerance')
        #Turn 0 into 1:
        if verbose: print(f'{np.sum(scalers<=0)} zero scalers')
        scalers[scalers<=0] = 1
        assert np.sum(scalers<=0) == 0
    
    return scalers

def verify_aggregation_matrix(C):
    """
    Verifies that a membership matrix is in the correct
       format for our IPF algorithm.
    """
    #Verify all entries in C are zero or 1:
    assert np.all(np.isin(C.data, [0, 1]))
    
    #Verify all rows have exactly one 1:
    assert np.all(C.sum(axis=1) == 1)
    
    #Verify the rows are ordered by column-membership:
    for p in range(C.shape[1]):
        #Get the indices of rows with data (a 1) in column p:
        i = C.indices[C.indptr[p]:C.indptr[p+1]]
        #First, verify the array has the correct length:
        assert len(i) == i.max() - i.min() + 1
        #Second, verify the array is a range between min and max:
        sorted_i = np.sort(i)
        assert np.all(sorted_i == np.arange(i.min(), i.max()+1))
        #Third, verify the array starts at a point after the previous one
        assert i.min() == 0 or i.min() == previous_max + 1
        previous_max = i.max()
        
    #If all these are satisfied, we can return a dictionary of
    # row indices per column:
    C_idx_dict = {p:np.sort(C.indices[C.indptr[p]:C.indptr[p+1]]) for p in range(C.shape[1])}
    return C_idx_dict
        
def scale_checkerboard_matrix(M, A, C, C_idx_dict=None):
    """
    Scale the flow matrix M according to aggregated scalers A
        with aggregation matrix C. That is equivalent to a matrix
        multiplication of the form

        M * (C @ A @ C.T)

    But much more efficient when A is dense but small and M is sparse
        but large. 

    Note that the matrix C must satisfy certain criteria for this
        function to work, which can be checked with the
        `verify_aggregation_matrix` function. The dicionary returned
        in that function is the parameter `C_idx_dict` here, which
        will be computed if not passed (default is `None`)
    """
    #We will form blocks to be vertically stacked by processing origin
    # states first (and all corresponding destination states in the
    # order they appear):
    M_blocks = []
    #Must repeatedly find all i,j such that C_ip = 1 and C_jq = 1, which we can do
    # using the index pointer of each column index on a csc matrix:
    if C_idx_dict is None:
        #Note that the indices must be sorted!
        C_idx_dict = {p:np.sort(C.indices[C.indptr[p]:C.indptr[p+1]]) for p in range(C.shape[1])}
    #Iterate over entries of A, row-wise first:
    for p in range(A.shape[0]):
        #Get the row indices
        i = C_idx_dict[p]
        #We will append all column blocks to a single row bloc:
        M_blocks_row = []
        for q in range(A.shape[1]):
            #Get the column indices:
            j = C_idx_dict[q]
            #Compute the corresonding M value updates and add to list:
            M_blocks_row.append(A[p,q] * M[i[:,None], j])
        #After the row is ready, concatenate and add to vertical list:
        M_blocks.append(ss.hstack(M_blocks_row))
    #Concatenate vertically:
    Mprime = ss.vstack(M_blocks)
    return Mprime.tocsc()

def compute_violations(M, constraints):
    """
    Compute the maximum and total violation to
        each of the constraints.

    Returns
        max_violation : list, ordered according to `constraints`
        tot_violation : list, ordered according to `constraints`
    """
    max_violation = []
    tot_violation = []
    #Iterate over constraints:
    for (m, C, m_type, m_tol) in constraints:
        #Get the current value of the constraint and the violation
        current_value = get_IPF_current_values(M, m_type, C)
        violation = abs(current_value - m)
        #Save:
        max_violation.append(violation.max())
        tot_violation.append(violation.sum())
    return max_violation, tot_violation

############################################################################################

def get_constraints(base_year,
                    PEP_file='PEP.csv',
                    process_birthsdeaths=False,
                    rescale_flows=False,
                    rescale_marginals=False,
                    ignore_PR=True, PR_FIPs=72):
    """
    Collect the constraints for the base_year > base_year + 1 migration estimation.
    """
    
    #Get the PEP file and population constraints. Use the indices to remove PR:
    county_idx = prc.get_geography_indices('county', ignore_PR=ignore_PR)
    PEP_df = pd.read_csv(f'{vars._census_demographics_dir}processed/COUNTY_{PEP_file}', index_col='GEOID')
    PEP_df_ordered = PEP_df.loc[pd.Series(county_idx).sort_values().index]
    P0 = PEP_df_ordered[f'POPESTIMATE{base_year}'].values
    P1 = PEP_df_ordered[f'POPESTIMATE{base_year+1}'].values
    
    #Get the ACS matrix and flow constraints:
    F, Fhat =  prc.get_ACS1_state_to_state(base_year+1, ignore_PR=ignore_PR)
    
    #Process births and deaths if we would like:
    if process_birthsdeaths:
        #Collect the PEP components of change:
        births = PEP_df_ordered[f'BIRTHS{base_year+1}'].values
        deaths = PEP_df_ordered[f'DEATHS{base_year+1}'].values
        intMig = PEP_df_ordered[f'INTERNATIONALMIG{base_year+1}'].values
        #Adjust the destination population:
        P1_tilde = P1 - births + deaths - intMig
        P1 = P1_tilde
        
        #To adjust the migration matrix, collect net international migration and deaths per state from PEP:
        PEP_state_df = pd.read_csv(f'{vars._census_demographics_dir}processed/state_{PEP_file}', index_col='STATE')
        state_immigrant_df = pd.read_csv(f'{vars._census_migration_dir}d02_state/processed/{base_year}-{base_year+1}/immigrants.csv', index_col='State')
    
        #Consider cleaning:
        if ignore_PR:
            state_non_PR_indices = prc.collect_non_PR_indices('state')
            PEP_state_df = PEP_state_df.iloc[state_non_PR_indices]
            state_immigrant_df = state_immigrant_df.iloc[state_non_PR_indices]
        
        #Collect the quantities:
        state_deaths = PEP_state_df[f'DEATHS{base_year+1}'].values
        state_intMig = PEP_state_df[f'INTERNATIONALMIG{base_year+1}'].values        
        state_immig = state_immigrant_df.Estimate.values
        
        #Update F:
        delta_F = np.diag(state_deaths + state_immig - state_intMig)
        F_tilde = F + delta_F
        F = F_tilde

        #Rescale the flows to match the population estimates:
        if rescale_flows:
            alpha = P0.sum()/F.sum()
            F = alpha*F

        #Rescale the marginals per state to match the flow matrix:
        if rescale_marginals:
            C = prc.get_geography_matrices('county', 'state', ignore_PR=ignore_PR)
            #Get target values:
            P0_F = np.array(F.sum(axis=1)).flatten()
            P1_F = np.array(F.sum(axis=0)).flatten()
            #Get current values:
            P0_state = np.array(C.T @ P0).flatten()
            P1_state = np.array(C.T @ P1).flatten()
            #Get scalers k (target divided by current values, disaggregated):
            k0 = np.array(C @ (P0_F/P0_state)).flatten()
            k1 = np.array(C @ (P1_F/P1_state)).flatten()
            #Scale:
            P0 = k0 * P0
            P1 = k1 * P1
        
    return P0, P1, F, Fhat
    
############################################################################################
#FUNCTIONS TO SET UP IPF

def IPF(M,
        row_marginals=None,
        col_marginals=None,
        agg_marginals=None,
        agg_matrix=None,
        tolerance=1e-5,
        max_iterations=10_000,
        log_violations=True,
        ignore_zeros=False):
    """
    Performs IPFP (iterative proportional fitting procedure) to a matrix
        M with row and column marginal constraints (attained by summing 
        entries of M along the corresponding axis) and aggregation constraints
        (attained by aggregating entries of M with an auxiliary matrix C).

    The full IPF constraints are:

        row_sum(M) = row_marginals
        col_sum(M) = col_marginals
        agg_matrix.T @ M @ agg_matrix = agg_constraints

    Not passing one of these constraints (or setting them to None) will skip
        the updates relative to the constraint. Not passing any of the three
        constraints will simply return the matrix M.

    The function handles numpy arrays and csc sparse matrices for all two
        dimensional objects (the row and column marginals must be
        one-dimensional np.arrays). Sparse matrices are preferred.

    Parameters
    ----------
    M : np.Array or ss.csc_matrix of shape N_1 x N_1
        two-dimensional array containing the current values of the entries
        
    row_marginals : np.Array of shape N_1 or None
        one-dimensional array containing the row sum constraints
        
    col_marginals : np.Array of shape N_1 or None
        one-dimensional array containing the column sum constraints
        
    agg_marginals : np.Array or ss.csc_matrix of shape N_2 x N_2 or None
        two-dimensional array containing the aggregation constraints
        
    agg_matrix : np.Array or ss.csc_matrix of shape N_1 x N_2 or None
        two-dimensional array containing aggregation matrix

    tolerance : float, default 1e-5
        stopping criteria: update on M is smaller than given tolerance

    max_iterations : int, default 100_000
        stopping criteria: IPFP takes more than this much iterations

    log_violations : bool, default True
        return additional stats on violations of constraints per
        iteration. If False, tuple of constraint violations will contain
        two empty lists. If True, violations will be ordered by row, column,
        aggregated.

    ignore_zeros : Bool, default False
        whether to update values where the constraint is zero
        
    Returns
    ----------
    np.Array or ss.csc_matrix of shape N_1 x N_1
        updated version of the matrix M

    tuple of lists
        (max_update, avg_update)

    tuple of lists
        (max_constraint_violation, avg_constraint_violation)
    """

    max_updates, max_violations = [], []
    avg_updates, avg_violations = [], []
    for iter in (pbar := trange(max_iterations)):
        if iter == 0: current_M = M.copy()
    
        #Update with IPF:t
        updated_M = IPF_update(current_M,
                               row_marginals=row_marginals,
                               col_marginals=col_marginals,
                               agg_marginals=agg_marginals,
                               agg_matrix=agg_matrix,
                               ignore_zeros=ignore_zeros)
    
        #Log updates:
        update = abs(current_M - updated_M)
        max_updates.append(update.max())
        avg_updates.append(update.mean())
        pbar.set_description(f"Current max. update = {max_updates[-1]:.5f}")
    
        #Log violations:
        if log_violations:
            _max_violations, _avg_violations = [],[]
            if row_marginals is not None:
                row_violations = abs(np.array(updated_M.sum(axis=1)).ravel() - row_marginals)
                _max_violations.append(row_violations.max())
                _avg_violations.append(row_violations.mean())
            if col_marginals is not None:
                col_violations = abs(np.array(updated_M.sum(axis=0)).ravel() - col_marginals)
                _max_violations.append(col_violations.max())
                _avg_violations.append(col_violations.mean())
            if agg_marginals is not None:
                agg_violations = abs(agg_matrix.T @ updated_M @ agg_matrix - agg_marginals)
                _max_violations.append(agg_violations.max())
                _avg_violations.append(agg_violations.mean())

            max_violations.append(_max_violations)
            avg_violations.append(_avg_violations)
    
        #Stopping criteria:
        if max_updates[-1] < tolerance:
            print(f'Model converged in {iter+1} iterations for the tolerance of {tolerance}')
            break
            
        #Continue:
        current_M = updated_M
        if iter == max_iterations - 1:
            print(f'Model achieved a tolerance of {max_updates[-1]} in {iter+1} iterations')

    return updated_M, (max_updates, avg_updates), (max_violations, avg_violations)

def IPF_update(M,
               row_marginals=None,
               col_marginals=None,
               agg_marginals=None, agg_matrix=None,
               ignore_zeros=False):
    """
    Get a single IPFP (iterative proportional fitting procedure) update
        step to bring entries of a matrix M closer to satisfying row and
        column marginal constraints and (optionally) aggregation constraints
        attained by aggregating entries of M with an auxiliary matrix C.

    The full IPF constraints are:

        row_sum(M) = row_marginals
        col_sum(M) = col_marginals
        agg_matrix.T @ M @ agg_matrix = agg_constraints

    Not passing one of these constraints (or setting them to None) will skip
        the updates relative to the constraint. Not passing any of the three
        constraints will simply return the matrix M.

    The function handles numpy arrays and csc sparse matrices for all two
        dimensional objects (the row and column marginals must be
        one-dimensional np.arrays).

    Parameters
    ----------
    M : np.Array or ss.csc_matrix of shape N_1 x N_1
        two-dimensional array containing the current values of the entries
        
    row_marginals : np.Array of shape N_1 or None
        one-dimensional array containing the row sum constraints
        
    col_marginals : np.Array of shape N_1 or None
        one-dimensional array containing the column sum constraints
        
    agg_marginals : np.Array or ss.csc_matrix of shape N_2 x N_2 or None
        two-dimensional array containing the aggregation constraints
        
    agg_matrix : np.Array or ss.csc_matrix of shape N_1 x N_2 or None
        two-dimensional array containing aggregation matrix

    ignore_zeros : Bool, default False
        whether to update values where the constraint is zero
        
    Returns
    ----------
    np.Array or ss.csc_matrix of shape N_1 x N_1
        updated version of the matrix M
    """

    #Scale each row:
    if row_marginals is not None:
        #Get the row scalers and convert to a matrix:
        row_sums = np.array(M.sum(axis=1)).ravel()
        row_scalers = get_IPF_scalers(row_sums, row_marginals)
        R = ss.diags(row_scalers)
        #Multiply from the left:
        M = R @ M

    #Scale each column:
    if col_marginals is not None:
        #Get the column scalers and convert to a matrix:
        col_sums = np.array(M.sum(axis=0)).ravel()
        col_scalers = get_IPF_scalers(col_sums, col_marginals)
        C = ss.diags(col_scalers)
        #Multiply from the right:
        M = M @ C

    #Scale each coarse area pair
    if agg_marginals is not None:
        assert agg_matrix is not None, 'Must pass aggregation matrix'
        #Get the aggregation scalers and convert to a matrix:
        agg_sums = agg_matrix.T @ M @ agg_matrix
        agg_scalers = get_IPF_scalers(agg_sums, agg_marginals, ignore_zeros=ignore_zeros, verbose=True)
        A = agg_matrix @ agg_scalers @ agg_matrix.T
        #Multiply element-wise:
        M = M.multiply(A) if ss.issparse(M) else M * A

    return M

def get_IPF_scalers(current_values, target_values, ignore_zeros=False, verbose=False):
    """
    Get a series of IPF scalers determined by
    
        scalers = target_values/current_values

    Processess so that if current_values are 0,
        then scaler defaults to 1 (no change)
    """
    
    #Create default scalers:
    scalers = np.ones(target_values.shape)

    #Find indices where the scaler is defined:
    idx_to_update = np.nonzero(current_values)
    scalers[idx_to_update] = target_values[idx_to_update]/current_values[idx_to_update]

    #If we are ignoring zero scalers:
    if ignore_zeros: scalers[scalers==0] = 1
    
    return scalers

############################################################################################

def rescale_E(E, F, stayers=None, P=None, fix_diagonal=False):
    """
    Rescale the INFUTOR matrix
    """
    
    #Rescale E to match the sum of F:
    if not fix_diagonal:
        scale_factor = F.sum()/E.sum()
        E_rescaled = scale_factor*E
        
    #Fix the number of stayers first:
    else:
        scale_factor = (P - stayers)/(E.sum(axis=0) - E.diagonal())
        scale_factor = np.nan_to_num(scale_factor, posinf=1)
        E_rescaled = E.multiply(scale_factor.T).toarray()
        np.fill_diagonal(E_rescaled, stayers)

    return E_rescaled

def process_E(E, F=None, stayers=None, P0=None, rescale=False, fix_diagonal=False):
    """
    Process the INFUTOR matrix for the objective
        computation (scales and flattens).
    """
    #Rescale E to match the sum of F:
    if rescale and not fix_diagonal:
        assert F is not None, 'To rescale, must pass F'
        E = rescale_E(E, F=F, fix_diagonal=False)

    #Rescale E to be close to the sum of F, fixing diagonal:
    if rescale and fix_diagonal:
        print('Fixing the diagonal')
        assert stayers is not None, 'To fix diagonal must pass a vector of stayers'
        assert P0 is not None, 'To fix diagonal must pass a vector of pop. marginals'
        E = rescale_E(E, F=F, stayers=stayers, P=P0, fix_diagonal=True)
        
    #Reshape into vector:
    E_vector = E.reshape(1, -1, order='C')
    E_vector_flat = utils.flat(E_vector)

    return E_vector_flat

def process_P(P):
    """
    Process the population vector as a normalization
        vector for the flatten matrix.
    """
    #The number of repetitions is the length of the
    # vector (as the matrix is square):
    N = len(utils.flat(P))
    
    #Repeat the vector:
    denominator = np.repeat(utils.flat(P), N)

    #Replace zeros:
    denominator[denominator==0] = 1

    return denominator
    
############################################################################################
# FUNCTIONS TO COLLECT DATA:

def collect_data(year,
                 fine_geography='ZIP',
                 positive_population=False,
                 discard_empty_coarse_areas=True,
                 optimize_population=True,
                 optimize_population_scaling=False,
                 coarsify=False,
                 coarsify_params={'aggregate_P':False}):
    """
    NOTE: this function has been tested for when geometries
            don't change and population values are available
            through the ACS api. That is 2011-2019 (inclusive).
    
    Collects real data in the format required by the 
        optimization problem:

        C : binary 2D array of shape N_1 x N_2
            indicator matrix for area membership
    
        F : float 2D array of shape N_2 x N_2
            flow matrix between coarse-grained areas (counties) given
            by ACS 5-year flows
    
        E : float 2D array of shape N_1 x N_1
            noisy flow matrix between fine-grained areas (ZIPs, CBGs)
            given by INFUTOR data over the same time period
    
        P_0 : float 1D array of shape N_1
            fine-grained population at initial time, from ACS
    
        P_1 : float 1D array of shape N_1
            fine-grained population at final time, from ACS

    Parameters
    ----------
    year : int
        initial time 

    fine_geography : `ZIP` or `CBG`
        fine-grained unit of spatial analysis, for which
        INFUTOR flows are aggregated

    coarsify : Bool, default False
        if True, aggregates the INFUTOR matrix entries by county
        and the census county matrix by state. This will
        create a problem with N1 ~ 3,000 and N2 = 52 which is
        solveable with fewer resources.

    positive_population : Bool, default False
        if True, ensures every fine-grained area has population of at
        least 1 in both time periods.

    discard_empty_coarse_areas : Bool, default True
        if True, removes from consideration coarse-grained areas that
        contain no fine-grained area
        
    coarsify_params : dict
        parameters to define coarsification

        : aggregate_P : Bool, default False
            whether to compute coarse P from fine population vectors (True)
            or from rows/columns of flow matrix F (False)
        
    Returns
    ----------
    dict
        keys are the variables indicated on the description, and
        if coarsify is True, will also return a "ground-truth"
        matrix M : the original census flow by county.
    """
    #We will store our values in a data dictionary:
    data_dict = {}
    
    #Obtain the coarse flow matrix F:
    F_dataframe = prc.read_county_matrices(year)[year]
    data_dict['F'] = ss.csr_matrix(F_dataframe, dtype=np.float64)

    #We also read the counties from the index of F (keep the order!)
    counties = F_dataframe.index.values
    data_dict['F_idx'] = counties
    N_2 = len(counties)

    #The INFUTOR matrix is already pre-processed---select the geography:
    x = {'ZIP':'1', 'CT':'2', 'CBG':'3', 'CB':'4'}
    file_directory = f'{vars._infutor_migration_dir}d0{x[fine_geography]}_{fine_geography}/'
    file_name = f'INFUTOR_{fine_geography}_{year}-{year+4}'
    data_dict['E'] = ss.load_npz(f'{file_directory}{file_name}.npz').astype(np.float64)
    
    N_1 = data_dict['E'].shape[0]
    
    #Load from the GeoDataFrame of fine-grained areas (final time
    # version to standardize issues with geographies changing):
    gdf_t1 = gpd.read_file(f'{vars._census_spatial_dir}{fine_geography}.gpkg',
                           layer=str(year+4),
                           ignore_geometry=True).set_index(f'{fine_geography}_code')
    assert len(gdf_t1) == N_1

    #Get the index of each county (according to F):
    county_idx_map = {county:idx for idx,county in enumerate(counties)}
    counties_idx = gdf_t1['county_code'].map(county_idx_map).values
    data_dict['M_idx'] = gdf_t1.index.values
    
    #Create matrix C:
    coords = (np.arange(N_1), counties_idx)
    data_dict['C'] = ss.csr_matrix((np.ones(N_1), coords),
                                   shape=(N_1, N_2), dtype=int)

    #Remove counties with no fine-grained areas:
    if discard_empty_coarse_areas:
        areas_to_discard = data_dict['C'].sum(axis=0).A.flatten() == 0
        data_dict['F'] = data_dict['F'][~areas_to_discard,:][:,~areas_to_discard]
        data_dict['C'] = data_dict['C'][:,~areas_to_discard]
        _, N_2 = data_dict['C'].shape
        data_dict['F_idx'] = data_dict['F_idx'][~areas_to_discard]
        

    #Now we just need the population vectors. Read for t_1:
    data_dict['P_1'] = gdf_t1['population'].values.astype(np.float64)

    #And load the t_0 gdf (TODO: verify for when geometries change):
    gdf_t0 = gpd.read_file(f'{vars._census_spatial_dir}{fine_geography}.gpkg',
                           layer=str(year),
                           ignore_geometry=True).set_index(f'{fine_geography}_code')
    data_dict['P_0'] = gdf_t0.loc[gdf_t1.index.values, 'population'].values.astype(np.float64)
    
    #Ensure the population is at least 1:
    if positive_population:
        data_dict['P_1'][data_dict['P_1']==0] = 1
        data_dict['P_0'][data_dict['P_0']==0] = 1

    if optimize_population:
        assert discard_empty_coarse_areas, 'Must discard empty coarse areas to optimize'
        
        P_0_opt = get_optimized_population(data_dict['P_0'],
                                           data_dict['F'].sum(axis=1),
                                           data_dict['C'])
        if P_0_opt is not None: data_dict['P_0'] = P_0_opt
        
        P_1_opt = get_optimized_population(data_dict['P_1'],
                                           data_dict['F'].sum(axis=0),
                                           data_dict['C'])
        if P_1_opt is not None: data_dict['P_1'] = P_1_opt

    if optimize_population_scaling:

        optimized_tuple = get_reconciled_constraints(P0=data_dict['P_0'],
                                                     P1=data_dict['P_1'],
                                                     F=data_dict['F'],
                                                     C=data_dict['C'])
        data_dict['P_0'], data_dict['P_1'], data_dict['F'] = optimized_tuple

    #Check if we would like to coarsify the data---which we can do using the matrix C
    # and county codes (which contain the state codes):
    if coarsify:

        #Matrix F is our ground truth matrix M:
        data_dict['M'] = data_dict['F']
        data_dict['M_idx'] = data_dict['F_idx']

        #Get the states:
        states_per_county = [county_code[:2] for county_code in counties]
        states = list(set(states_per_county))
        states.sort()
        N_2_prime = len(states)
        assert N_2_prime == 52
        data_dict['F_idx'] = states
        
        #Map the state codes to the index:
        states_idx_map = {state:idx for idx,state in enumerate(states)}
        states_idx = [states_idx_map[state] for state in states_per_county]
        
        #Create a membership matrix N_2 x 52 of counties to states:
        coords = (np.arange(N_2), np.array(states_idx))
        C_prime = ss.csr_matrix((np.ones(N_2), coords),
                                shape=(N_2, N_2_prime), dtype=int)
        
        #Project matrices:
        data_dict['F']   =        C_prime.T @ data_dict['F'] @ C_prime
        data_dict['E']   = data_dict['C'].T @ data_dict['E'] @ data_dict['C']

        #Compute the population one of two ways:
        if coarsify_params['aggregate_P']:
            data_dict['P_0'] = data_dict['C'].T @ data_dict['P_0']
            data_dict['P_1'] = data_dict['C'].T @ data_dict['P_1']
        else:
            data_dict['P_0'] = np.array(data_dict['M'].sum(axis=1)).flatten()
            data_dict['P_1'] = np.array(data_dict['M'].sum(axis=0)).flatten()
            
        #Update matrix C:
        data_dict['C'] = C_prime
    
    return data_dict

############################################################################################

def constraint_check(M, F, C, P_0, P_1,
                     diag_constraints=None,
                     diag_constraints_type=None,
                     tol=10**(-5)):
    """
    Check that a given matrix M satisfy the problem constraints
    """
    
    #Verify the positivity constraint:
    assert np.all(M >= 0), 'Estimate contains negative entries'

    #Collect constraint violations:
    flows_constraint = C.T @ M @ C - F
    destp_constraint = M.sum(axis=0) - P_1
    origp_constraint = M.sum(axis=1) - P_0

    #Verify that maximum constraint violation is below tolerance:
    for constraint_violation in [flows_constraint, destp_constraint, origp_constraint]:
        max_constraint_violation = np.max(abs(constraint_violation))
        log_violation = np.log10(max_constraint_violation)
        assert max_constraint_violation < tol, f'Max violation on the order of 10**{log_violation:.2f}'

    #Assert the diagonal constraints:
    if diag_constraints_type is not None:
        
        D = M.diagonal() - diag_constraints
        if diag_constraints_type == 'exact': assert np.max(abs(D)) < tol, 'Estimate does not match diagonal'
        elif diag_constraints_type == 'LB': assert np.all(D >= 0), 'Estimate contains diag. entries below LB'

    return None    

############################################################################################

def get_optimized_population(P_vec, F_vec, C, verbose=False):
    """
    Product a population vector P for the fine-grained
        areas that yields the correct population for
        the coarse-grained areas when aggregated according
        to a membership matrix C. That is:

    P_vec = min_x || x - P_vec || s.t. C.T @ x = F_vec    
    """
    #Get the problem dimensions and flatten:
    N_1, N_2 = C.shape
    P_vec = np.ravel(P_vec)
    F_vec = np.ravel(F_vec)

    #Define a model:
    env = gp.Env(empty=True)
    if not verbose: env.setParam('OutputFlag', 0)
    env.start()
    model = gp.Model('optimizing pop.', env=env)

    # Add variables with non-negative constraint
    x = model.addMVar(shape=N_1, lb=0, name='P')
    
    # Set the objective
    obj = (x - P_vec) @ (x - P_vec)
    model.setObjective(obj, gp.GRB.MINIMIZE)

    # Add constraint that marginals match the coarse flows:
    model.addConstr(C.T @ x == F_vec)

    # Optimize model
    model.optimize()

    #Collect output:
    out = x.X if not model.status == gp.GRB.INFEASIBLE else None
    env.close()

    return out

def get_reconciled_constraints(P0, P1, F,
                               C=None,
                               true_population='F'):
    """
    Reconcile population and coarse-flow constraints so that
        the system has a constant population (or as close as
        possible without messing with zero entries).

    If no aggregation matirx C is passed, constraints are just
        naively re-scaled so they sum to the total population
        given in `true_population`. If C is passed, each county 
        is separately re-scaled so that it matches the origin or
        destination population of that county, according to F.

    Parameters
    ----------
    P0 : np.Array of shape N_1
        one-dimensional array containing original population
        
    P1 : np.Array of shape N_1
        one-dimensional array containing final population
        
    F : np.Array or ss.csc_matrix of shape N_2 x N_2
        two-dimensional array containing the flows between coarse areas
        
    C : np.Array or ss.csc_matrix of shape N_1 x N_2 or None
        two-dimensional array containing aggregation matrix

    true_population : int or `P0` or `P1` or `F`
        either a value, or the value corresponding to the sum of
        a given array
        
    Returns
    ----------
    np.Array of shape N_1
        updated version of P0
        
    np.Array of shape N_1
        updated version of P1
        
    np.Array of shape N_2 x N_2
        updated version of F
    """

    #Get the true population of the system:
    if true_population in ['P0', 'P_0', 'origin', '0']:
        true_population = P0.sum()
    elif true_population in ['P1', 'P_1', 'destination', '1']:
        true_population = P1.sum()
    elif true_population in ['F', 'flows', 'coarseflows']:
        true_population = F.sum()
    else:
        true_population = float(true_population)
    assert true_population > 0, 'Population must be positive'
        
    #First, re-scale the coarseflows naively:
    updated_F = float(true_population/F.sum())*F

    #If we did not pass an aggregation matrix, just rescale the
    # population vectors accordingly:
    if C is None:
        updated_P0 = float(true_population/P0.sum())*P0
        updated_P1 = float(true_population/P1.sum())*P1

    #If we did pass an aggregation matrix, rescale population
    # vectors according to their coarse areas:
    else:

        #We need that fine area belong to at most one coarse area:
        assert set(C.data) == {1.0}
        assert (C.sum(axis=1) > 1).sum() == 0

        #Update each population vector:
        updated_P = []
        for P, axis in zip([P0, P1], [1, 0]):
            #Figure out target coarse-are population and current coarse
            # area population for each fine-grained area entry:
            targ_P = (C @ updated_F.sum(axis=axis).flatten().T).flatten()
            if type(targ_P) == np.matrix: targ_P = np.array(targ_P).flatten()
            curr_P = (C @ C.T @ P).flatten()
            #Scale:
            updated_P.append((targ_P/curr_P)*P)
        
        updated_P0, updated_P1 = updated_P[0], updated_P[1]

    return updated_P0, updated_P1, updated_F

############################################################################################

def preprocess_E_diagonal(E, min_val=1):
    """
    Pre-process a csr sparse matrix E such that its
        diagonal has a minimum value specified
    """
    
    # Get the diagonal elements
    diagonal_indices = np.arange(min(E.shape))
    current_diagonal = E.diagonal()

    # Create a mask where the diagonal elements need to be replaced
    mask = current_diagonal < min_val
    E[diagonal_indices[mask], diagonal_indices[mask]] = min_val
    
    return E

def preprocess_E_population(E, total_population):
    """
    Pre-process a csr sparse matrix E such that its
        entries sum to a total popualtion value
    """
    
    scaler = total_population/E.sum()
    updated_E = scaler*E
    
    return updated_E

def preprocess_E_coarseflows(E, F, C):
    """
    Pre-process a csr sparse matrix E such that it
        sums to the coarse flow matrix F
    """

    #We need that fine area belong to at most one coarse area:
    assert set(C.data) == {1.0}
    assert (C.sum(axis=1) > 1).sum() == 0

    #We essentially do one IPF iteration:
    scalers_N2xN2 = get_IPF_scalers(C.T @ E @ C, F)
    scalers_N1xN1 = C @ scalers_N2xN2 @ C.T
    
    #Multiply element-wise:
    updated_E = E.multiply(scalers_N1xN1) if ss.issparse(E) else E * scalers_N1xN1

    return updated_E.tocsr()

def preprocess_E(E, F, C,
                 population=True, diagonal=True, coarseflows=True):
    """
    Pre-process E sequentially so that

    1. The coarseflows agree with F
    2. Every diagonal entry is at least 1
    3. The total population agree with F
    """
    #1. The coarseflows agree with F:
    if coarseflows: E = preprocess_E_coarseflows(E, F, C)
    #2. Every diagonal entry is at least 1:
    if diagonal: E = preprocess_E_diagonal(E)
    #3. The total population agree with F
    if population: E = preprocess_E_population(E, F.sum())

    return E
