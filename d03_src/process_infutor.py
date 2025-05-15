############################################################################################
# Functions to process Infutor data
#     1. Functions to read Infutor txt files into a csv
#     2. Functions to process an Infutor PID into an ACS response distribution
#     3. Functions to process Origin-Destination files
#     4. Functions to collect INFUTOR matrices

############################################################################################

import numpy as np
import pandas as pd
import geopandas as gpd
import scipy.sparse as ss
from datetime import datetime, timedelta

#Updating a csc matrix is costly, do it sparingly!!!
import warnings
warnings.filterwarnings("ignore", category=ss.SparseEfficiencyWarning)


import sys
sys.path.append('../d03_src/')
import vars
import process_census

############################################################################################
#1. Read INFUTOR txt files
def get_CRD4_columns(CRD4_layout_filepath=f'{vars._infutor_dir_processed}CRD4_fields.csv'):
    """
    Collects the column names from CRD4:
    """    
    
    layout = pd.read_csv(CRD4_layout_filepath).set_index('Name')
    columns = layout.index.values
    
    return columns

def process_df(df,
               cols_to_drop=None,
               nan_representations=['', 'nan']):
    """
    Cleans the INFUTOR dataframe: drops columns (probably INTERNAL ones),
        convert string representations of nan to np.nan, sets the type in
        some date and float columns.
    """

    #Remove internal columns:
    if cols_to_drop is not None:
        df = df.drop(cols_to_drop, axis=1)
        
    #Convert string representations to NaN:
    if nan_representations is not None:
        for nan in nan_representations:
            df = df.replace(nan, np.nan)

    return df

############################################################################################
# 2. Process INFUTOR individual
def process_individual(individual,
                       PO_box_window_in_days=365,
                       pad_ends_in_days=365,
                       pad_only_if_alive=False,
                       min_year=1902,
                       max_year=np.inf,
                       valid_address_types=['clean', 'rural_route', 'incomplete'],
                       date_columns=['IDATE', 'ODATE'],
                       effdate_columns=[f'EFFDATE{k}' for k in range(1, 11)],
                       address_columns=[f'ADDRID{k}' for k in range(1, 11)],
                       category_columns=[f'ADDRCAT{k}' for k in range(1, 11)],
                       verbose=False):
    """
    Function to process one row (individual) in the INFUTOR PID dataframe into
        an individual "distribution of ACS yearly responses" for every year in
        the range. That is, estimated yearly personal flows between ADDRIDs.
        
    Returns
    ----------
    list
        entries are 4-tuples of the format
        
        (YEAR, ORIGIN_ADDRID, DEST_ADDRID, PROBABILITY)

        Selecting all entries of the year and mapping IDs to indices
        will generate a COO sparse matrix. Note that for the first and
        last years when an individual is active the probabilities may not
        sum to 1.
    """
    #Resolve IDATE and ODATE inconsistencies:
    start_date, end_date = hygienize_IO_DATES(individual,
                                              date_columns=date_columns,
                                              effdate_columns=effdate_columns)
    individual.loc[date_columns] = start_date, end_date
    
    #In some cases there are no valid dates, so we skip these:
    if pd.isna(start_date):
        yearly_ACS_responses_list = []
    else:
        #Remove undated addresses:
        updated_values = remove_NaT_addresses(individual,
                                              date_columns=date_columns,
                                              effdate_columns=effdate_columns,
                                              address_columns=address_columns,
                                              category_columns=category_columns,
                                              verbose=verbose)
        if verbose: print('After removing undated values:', updated_values)
        individual.loc[effdate_columns+address_columns+category_columns] = updated_values
        
        #Remove PO boxes:
        updated_values = remove_PO_box(individual,
                                       window_in_days=PO_box_window_in_days,
                                       valid_address_types=valid_address_types,
                                       effdate_columns=effdate_columns,
                                       address_columns=address_columns,
                                       category_columns=category_columns)
        if verbose: print('\nAfter removing PO boxes:', updated_values)
        individual.loc[effdate_columns+address_columns+category_columns] = updated_values
        
        #Get the address history:
        address_history = get_monthly_addresses(individual,
                                                pad_ends_in_days=pad_ends_in_days,
                                                pad_only_if_alive=pad_only_if_alive,
                                                effdate_columns=effdate_columns,
                                                address_columns=address_columns,
                                                category_columns=category_columns)
        if verbose: print('\nAddress history:', address_history)
    
        #Get ACS response probability distribution:
        yearly_ACS_responses_list = get_yearly_ACS_responses(address_history,
                                                             min_year=min_year,
                                                             max_year=max_year)

    return yearly_ACS_responses_list
    
def hygienize_IO_DATES(individual,
                       date_columns=['IDATE', 'ODATE'],
                       effdate_columns=[f'EFFDATE{k}' for k in range(1, 11)]):
    """
    Resolve IDATE (first date a person is seen) and ODATE
        (last date a person is seen) inconsistencies.
    Assigns to IDATE the earliest across all dates and 
        to ODATE the latest across all dates.

    Parameters
    ----------
    individual : pd.Series
        row of an individual DataFrame containing EFFDATE and ADDRID
        columns, as well as IDATE, ODATE, DeceasedCD
        
    Returns
    ----------
    (new_start_date, new_end_date)       
    """
    #Create a hygienized date:
    start_date = individual[date_columns+effdate_columns].dropna().min()
    end_date = individual[date_columns+effdate_columns].dropna().max()

    return start_date, end_date

def remove_NaT_addresses(individual,
                         date_columns=['IDATE', 'ODATE'],
                         effdate_columns=[f'EFFDATE{k}' for k in range(1, 11)],
                         address_columns=[f'ADDRID{k}' for k in range(1, 11)],
                         category_columns=[f'ADDRCAT{k}' for k in range(1, 11)],
                         verbose=False):
    """
    Remove addresses without a date.

    Parameters
    ----------
    individual : pd.Series
        row of an individual DataFrame containing EFFDATE and ADDRID
        columns, as well as IDATE, ODATE, DeceasedCD
        
    Returns
    ----------
    pd.Series with EFFDATE ADDRID and ADDRCAT columns
    """

    #Collect values:
    IDATE = individual['IDATE']
    dates = individual[effdate_columns].values
    addresses = individual[address_columns].values
    categories = individual[category_columns].values

    # Nullify addresses where the corresponding date is NaT and vice-versa:
    for idx, (date, address) in enumerate(zip(dates, addresses)):
        #If there is no address, ensure there is no date or category:
        if pd.isna(address):
            dates[idx] = pd.NaT
            categories[idx] = pd.NA
        #If there is an address but no date, decide what to do based on
        # whether we have other options:
        else:
            if pd.isna(date):
                #If there is more than one address, remove:
                n_addresses = individual[address_columns].dropna().nunique()
                if n_addresses > 1:
                    if verbose: print(f'Multiple addresses: remove {address}')
                    addresses[idx] = pd.NA
                    categories[idx] = pd.NA
                #Otherwise, keep it and impute IDATE:
                elif n_addresses == 1:
                    if verbose: print(f'Only one address: impute {IDATE} as date of {address}')
                    dates[idx] = IDATE
    #Group into the new row:
    new_row = pd.Series(list(dates)+list(addresses)+list(categories),
                        index=effdate_columns+address_columns+category_columns)
    return new_row

def remove_PO_box(individual,
                  window_in_days=365,
                  valid_address_types=['clean', 'rural_route', 'incomplete'],
                  effdate_columns=[f'EFFDATE{k}' for k in range(1, 11)],
                  address_columns=[f'ADDRID{k}' for k in range(1, 11)],
                  category_columns=[f'ADDRCAT{k}' for k in range(1, 11)]):
    """
    Remove PO Box addresses within a reasonable window of a clean
        address

    Parameters
    ----------
    individual : pd.Series
        row of an individual DataFrame containing EFFDATE, ADDRID, and
        ADDRCAT columns
        
    Returns
    ----------
    pd.Series with EFFDATE, ADDRID, and ADDRCAT columns
    """
    #Note that, after step 2, all of these must have the same length:
    addresses = individual[address_columns]
    dates = individual[effdate_columns]
    categories = individual[category_columns]
    
    #Proceed only if we have a PO box, otherwise return the row:
    address_types_in_individual = set(categories.dropna().values)
    
    #If no PO box, just original row:
    if 'PO_box' not in address_types_in_individual:
        new_row = individual
    #Also nothing we can do if only PO box is present:
    elif len(address_types_in_individual)==1:
        new_row = individual
    #In case we have PO box and other types:
    else:
        #Separate the dates of non-PO box addresses:
        nonPO_indices = np.flatnonzero(categories.isin(valid_address_types))
        nonPO_dates = dates.iloc[nonPO_indices].dropna()
        PObox_indices = np.flatnonzero(categories=='PO_box')
        
        #Iterate through all PO Boxes
        for PO_box_idx in PObox_indices:
            PO_box_date = dates.iloc[PO_box_idx]
            remove_PO_box = False
            #For every non-PO date, check if it is within the window:
            for date in nonPO_dates:
                delta = abs(PO_box_date-date).days
                #If it is, null the PO box address (and stop)
                if delta < window_in_days:
                    addresses.iloc[PO_box_idx] = pd.NA
                    dates.iloc[PO_box_idx] = pd.NaT
                    categories.iloc[PO_box_idx] = pd.NA
                    break
        #The new row:
        new_row = pd.Series(list(dates)+list(addresses)+list(categories),
                            index=effdate_columns+address_columns+category_columns)
        
    return new_row

def get_monthly_addresses(individual, 
                          pad_ends_in_days = 365,
                          pad_only_if_alive = True,
                          effdate_columns=[f'EFFDATE{k}' for k in range(1, 11)],
                          address_columns=[f'ADDRID{k}' for k in range(1, 11)],
                          category_columns=[f'ADDRCAT{k}' for k in range(1, 11)]):
    """
    Get the monthly addresses (and associated probabilities) of each individual
        during their active INFUTOR time.
        
    Returns
    ----------
    dict
        keys are months, items are dictionaries with keys
        ADDRID (list) and p (float)
    """
    #Collect addresses, dates, and categories:
    addresses = individual[address_columns]
    dates = individual[effdate_columns]
    categories = individual[category_columns]
    
    #Determine start and end dates:
    pad = timedelta(days=pad_ends_in_days)
    start_date = individual['IDATE'] - pad
    end_date = individual['ODATE']
    
    #If the invidual is alive, pad end date:
    if pd.isna(individual['DeceasedCD']):
        end_date += pad
    elif not pad_only_if_alive:
        end_date += pad
    
    #Create an empty monthly dataframe:
    months = pd.date_range(start_date, end_date, freq='MS').strftime("%Y-%m").tolist()
    monthly_address = pd.DataFrame([], columns=months, index=['ADDRID_idx', 'p'])
    
    #Create an address history dataframe:
    address_history = pd.DataFrame([dates.values, categories.values, addresses.values],
                                   index=['EFFDATE', 'ADDRCAT', 'ADDRID']).T
    
    #We will use a mapping of idx > [ADDRID list] in order to call ffill and bfill
    # on address groups (we can't safely call on objects):
    addresses_mapping = {}
    
    #Iterate over dates to create a monthly snapshot of addresses and probabilities:
    addresses_per_date = address_history.groupby('EFFDATE', sort=True)
    for date_idx, (date, addresses) in enumerate(addresses_per_date):
        #First, check if we have non-PO box addresses in the date and select those,
        # otherwise select all [PO boxes]:
        clean_addresses = addresses.loc[addresses.ADDRCAT != 'PO_box']
        if len(clean_addresses) == 0: clean_addresses = addresses
        addresses_at_the_month = list(clean_addresses.ADDRID.values)
        #Log this address list with the date index:
        addresses_mapping[float(date_idx)] = addresses_at_the_month
        #Divide the probability evenly among clean addresses:
        p = 1/len(addresses_at_the_month)
        #Log into the monthly dataframe:
        month = date.strftime("%Y-%m")
        monthly_address[month] = (date_idx, p)
        
    #Fill the dates until the next address (and backwards at first):
    # OBS: Do this operation column-wise to preserve the dtypes and avoid
    #       a warning on the behavior of ffill.
    monthly_address_transpose = monthly_address.T.astype(float)
    monthly_address_filled = monthly_address_transpose.ffill(axis=0).bfill(axis=0)
    #Map date indices to addresses:
    monthly_address_filled['ADDRID'] = monthly_address_filled['ADDRID_idx'].map(addresses_mapping).astype('object')
    #Return a dictionary:
    monthly_address_history = monthly_address_filled.loc[:,['ADDRID', 'p']].to_dict(orient='index')
    
    return monthly_address_history

def get_yearly_ACS_responses(address_history,
                             min_year=0, max_year=np.inf):
    """
    For an address history, get the yearly ACS responses (estimated)
    """
    #Separate month and year:
    address_history_df = pd.DataFrame(address_history).T
    dates_as_str = address_history_df.index.str
    address_history_df['year'] = dates_as_str[:4]
    address_history_df['month'] = dates_as_str[-2:]
    
    #Find the active years of the individual:
    active_years = address_history_df['year'].drop_duplicates().sort_values().to_list()
    active_years = [y for y in active_years if int(y)>=min_year-1 and int(y)<=max_year]
    
    #We don't have any data if the list of active years contains a single
    # year (no ACS response period):
    yearly_responses = []
    if len(active_years) > 1:
        #For every year, compute probability distribution of ACS answers:
        for year in active_years[1:]:
            #Define the yearly monthly flow:
            yearly_addresses = address_history_df[address_history_df['year'] == year]
            previous_addresses = address_history_df[address_history_df['year'] == str(int(year)-1)]
            yearly_flow = yearly_addresses.merge(previous_addresses, on='month', how='outer',
                                                 suffixes=('_current', '_previous'), sort=True)
            #Only continue if we have at least one full year:
            if len(yearly_flow) == 12:
                #Map monthly weight:
                yearly_flow['weight'] = vars._monthly_weights #if (int(year)%4 != 0) else vars._leap_monthly_weights
                #We will apply the function row-wise:
                def _get_ACS_response(monthly_addresses):
                    #Define an empty list if not active at the month:
                    if pd.isna(monthly_addresses).any():
                        responses_list = []
                    #Otherwise, verify if person stayed or moved:
                    else:
                        current_addresses = monthly_addresses.ADDRID_current
                        previous_addresses = monthly_addresses.ADDRID_previous
                        stayed = set(current_addresses) == set(previous_addresses)
                        #For stayers, probability of response is probability they
                        # were at the address:
                        if stayed:
                            p = monthly_addresses.weight*monthly_addresses.p_current
                            responses_list = [(address,address,p) for address in current_addresses]
                        #For movers, cross-multiply:
                        else:
                            p = monthly_addresses.weight*monthly_addresses.p_current*monthly_addresses.p_previous
                            responses_list = [(address1,address2,p) for address1 in previous_addresses for address2 in current_addresses]
                    return responses_list
                ACS_responses = yearly_flow.apply(_get_ACS_response,axis=1)
                #Add up probabilities for the same pairs:
                ACS_responses_list = ACS_responses.explode().dropna().to_list()
                grouped_ACS_responses = pd.DataFrame(ACS_responses_list,
                                                     columns=['O', 'D', 'p']).groupby(['O', 'D']).sum().reset_index()
                #Include year and save the 4-tuple:
                grouped_ACS_responses.insert(0, 'year', int(year))
                yearly_responses.extend(list(grouped_ACS_responses.itertuples(index=False, name=None)))
    return yearly_responses
    
############################################################################################
# 3. Process INFUTOR flows:

def aggregate_individual_responses(individual_responses, verbose=False):
    """
    Takes a series of individual responses an aggregates into a dataframe
        of population ACS responses
    """

    #Get a full list of tuples and turn it into a dataframe:
    aggregated_responses_list = individual_responses.explode().dropna().reset_index(drop=True).to_list()
    aggregated_responses_df = pd.DataFrame(aggregated_responses_list,
                                           columns=['year', 'origin', 'destination', 'flow'],
                                           dtype=float)
    aggregated_responses_df = aggregated_responses_df.astype({'year':'Int32',
                                                              'origin':'Int64',
                                                              'destination':'Int64'})

    #Sum repeated values in the same year:
    clean_aggregated_responses_df = aggregated_responses_df.groupby(['year',
                                                                     'origin',
                                                                     'destination'], sort=True).sum().reset_index()
    N_sharing = len(aggregated_responses_df)-len(clean_aggregated_responses_df)
    if verbose: print(f'There were {N_sharing} individuals sharing a flow')

    return clean_aggregated_responses_df

def clean_OD_df(OD_df_raw, allowed_geographies):
    """
    Given the INFUTOR OD dataframe, removes invalid geographies
        and aggregates values based on repeated geographies
    """
    
    #The dataframe must contain origin and destination columns:
    assert 'origin'      in OD_df_raw.columns
    assert 'destination' in OD_df_raw.columns

    if 'flow' not in OD_df_raw.columns:
        OD_df_raw = OD_df_raw.assign(flow = 1)    
    
    #Group the OD pairs that repeat:
    OD_df_grouped = OD_df_raw.groupby(['origin', 'destination']).sum().reset_index()

    #Clean the df from non-existent geographies:
    ok_orig = OD_df_grouped['origin'     ].isin(allowed_geographies)
    ok_dest = OD_df_grouped['destination'].isin(allowed_geographies)
    OD_df_clean = OD_df_grouped[ok_orig&ok_dest].reset_index(drop=True)

    #Ensure we have no issues:
    assert OD_df_clean.isna().sum().sum() == 0

    return OD_df_clean
    
############################################################################################
#4. Collect INFUTOR:
def load_INFUTOR_matrix(year, geography='BLOCK_GROUP', ignore_PR=False, clip_diagonal_to_one=True,
                        E_matrix_dir=f'{vars._infutor_dir_processed}CRD4/OD_pairs/ADDRID/E_matrices/'):
    """
    Loads the INFUTOR matrix for the period from year-1 to year.
    """
    #Gets the matrix:
    E = ss.load_npz(f'{E_matrix_dir}{year}_blockgroup.npz')

    #Consider removing Puerto Rico:
    if ignore_PR:
        non_PR_indices = process_census.collect_non_PR_indices('blockgroup')
        E = E[non_PR_indices,:][:,non_PR_indices]

    #Consider aggregating:
    geography = geography.upper()
    if geography[-1] == 'S': geography = geography[:-1]
    if geography == 'BLOCKGROUP': geography = 'BLOCK_GROUP'
    if geography != 'BLOCK_GROUP':
        C = process_census.get_geography_matrices('BLOCK_GROUP', geography, ignore_PR=ignore_PR)
        E = C.T @ (E @ C)

    #Consider ensuring at least one person in the diagonal:
    if clip_diagonal_to_one:
        diagonal = E.diagonal()
        diagonal[diagonal < 1] = 1
        E.setdiag(diagonal) #This may give an efficiency warning---it's fine!
        
    return E
        

def verify_INFUTOR_matrix_integrity(year, test_codes=['10024', '10023', '10025', '10027']):
    """
    Checks that the sparse matrices reflect the OD dataframe.
    """

    OD_df = pd.read_csv(f'{vars._infutor_dir_processed}CRD4/ZIP_OD_pairs/{year}-{year+4}.csv', dtype=str)
    I_df = ss.load_npz(f'{vars._infutor_migration_dir}d01_ZIP/INFUTOR_ZIP_{year}-{year+4}.npz')
    ZIP_df = process_census.read_census_geography_shapefiles(years=year+4, geography='ZIP')[year+4]
    
    #Assert shapes match:
    assert len(ZIP_df) == I_df.shape[0]
    
    if test_codes is None: test_codes = ZIP_df.ZIP_codes.values
    
    for test_code in test_codes:
    
        flow_to_df = OD_df[OD_df.destination==test_code]
        flow_from_df = OD_df[OD_df.origin==test_code]
        test_idx = ZIP_df[ZIP_df.ZIP_code==test_code].index[0]
        
        #Assert indices match:
        assert set(flow_to_df.destination_idx.values) == {str(test_idx)}
        assert set(flow_from_df.origin_idx.values) == {str(test_idx)}
    
        #Get the origin and destination from the df:
        origin_series = flow_to_df[['flow', 'origin_idx']].set_index('origin_idx')['flow'].astype(int)
        destination_series = flow_from_df[['flow', 'destination_idx']].set_index('destination_idx')['flow'].astype(int)
    
        #Get the origin and destination from matrix:
        destination_I = I_df[test_idx].toarray().flatten()
        origin_I = I_df[:,test_idx].toarray().flatten()
            
        #Assert the sum of values matches:
        assert destination_I.sum() == destination_series.sum()
        assert origin_I.sum() == origin_series.sum()
        
        #Assert actual values match:
        for idx, flow in destination_series.items():
            assert destination_I[int(idx)] == flow
        
        for idx, flow in origin_series.items():
            assert origin_I[int(idx)] == flow
            
############################################################################################