############################################################################################
# Functions to process Census and ACS data
#     1. Functions to read Census spatial data
#     2. Functions to read Census yearly state to state migration data
#     3. Functions to read ACS 5-year county to county migration data

############################################################################################

import numpy as np
import pandas as pd
import geopandas as gpd
from tqdm import tqdm

from census import Census
from pygris import counties, zctas
import pygris
import scipy.sparse as ss
from scipy.optimize import nnls

import sys
sys.path.append('../d03_src/')
import vars
import os

############################################################################################
#1. Functions to read Census spatial data:
def get_census_geography(geography,
                         years=range(2010,2023), states=vars._states,
                         out_dir=f'{vars._census_spatial_dir}raw/'):
    """
    Collects a geodataframe of Census geographies into a .gpkg
        file with layers corresponding to the year these geographies
        were taken with respect to.
    """
    geographies = {'counties': pygris.counties,
                   'county_subdivisions': pygris.county_subdivisions,
                   'tracts': pygris.tracts,
                   'block_groups': pygris.block_groups,
                   'blocks': pygris.blocks}
    assert geography.lower() in geographies
    
    #Iterate over years:
    for year in tqdm(years):
        if year == 2010 and geography.lower() == 'counties':
            print('Library pygris contains a bug for counties in 2010')
        #Collect one gdf per state:
        gdfs = []
        for state in vars._states:
            gdf = geographies[geography](year=year, state=state)
            gdfs.append(gdf)
        #Merge all into a single GeoDataFrame and save:
        merged_gdf = pd.concat(gdfs)
        merged_gdf.to_file(f"{out_dir}{geography}.gpkg",
                           driver='GPKG', crs=merged_gdf.crs, layer=str(year))
    return None    

def collect_non_PR_indices(geography):
    """
    Collect the indices (for a given geography) of the
       geographies in the 50 states + DC.
    """
    #Verify geography:
    geography = geography.upper()
    if geography[-1] == 'S': geography = geography[:-1]
    if geography == 'BLOCKGROUP': geography = 'BLOCK_GROUP'
    assert geography in ['STATE', 'COUNTY', 'TRACT', 'BLOCK_GROUP', 'BLOCK'], 'Invalid geography'
    
    #Read the indices:
    idx_df = pd.read_csv(f'{vars._census_spatial_dir}processed/indices.csv',
                         usecols=[geography],
                         dtype='Int64')
    #Collect valid indices:
    indices = idx_df[geography].dropna()
    
    #PR indices have 72 afront the GEOID (assuming max length i.e.
    # no leading zeros). We find the length of each GEOID first and
    # then those with max lenght and leading 72:
    geoid_length = indices.apply(lambda x: len(str(x)))
    geoid_maxlength = geoid_length == geoid_length.max()
    leading_72 = indices.apply(lambda x: str(x)[:2]) == '72'
    PR_rows = geoid_maxlength&leading_72
    
    #Select them out:
    indices = indices.loc[~PR_rows].index.values

    return indices

def get_geography_indices(geography,
                          ignore_PR=False,
                          index_to_geography=False):
    """
    Retrieves a dictionary that maps between
        indices and geography names for a particular
        aggregation level.
    The default return is a map from geography (as int)
        to index. Set `index_to_geography` to True
        to obtain the reverse mapping.
    """
    #Verify geography:
    geography = geography.upper()
    if geography[-1] == 'S': geography = geography[:-1]
    if geography == 'BLOCKGROUP': geography = 'BLOCK_GROUP'
    assert geography in ['STATE', 'COUNTY', 'TRACT', 'BLOCK_GROUP', 'BLOCK'], 'Invalid geography'
    
    #Read the DataFrame:
    idx_df = pd.read_csv(f'{vars._census_spatial_dir}processed/indices.csv',
                         usecols=[geography],
                         dtype='Int64')
    
    #Collect valid indices:
    indices = idx_df[geography].dropna()
    
    #Potentially ignore Puerto Rico:
    if ignore_PR:
        indices_to_keep = collect_non_PR_indices(geography)
        indices = indices.iloc[indices_to_keep]
    
    #Convert to dictionary:
    idx_to_geography_dict = indices.to_dict()

    #Invert if we request:
    if index_to_geography:
        return idx_to_geography_dict
    else:
        geography_to_idx_dict = {v:k for k,v in idx_to_geography_dict.items()}
        return geography_to_idx_dict

def get_geography_matrices(fine_geography, coarse_geography,
                           as_dict=False,
                           ignore_PR=False,
                           matrix_dir=f'{vars._census_spatial_dir}processed/matrices/'):
    """
    Retrieves a matrix that maps between fine and coarse
        geographies i.e. an N x M binary matrix.
    """
    #Verify and clean geographies:
    geography_order = ['STATE', 'COUNTY', 'TRACT', 'BLOCK_GROUP', 'BLOCK']
    def clean_geography(geography):
        geography = geography.upper()
        if geography[-1] == 'S': geography = geography[:-1]
        if geography == 'BLOCKGROUP': geography = 'BLOCK_GROUP'
        return geography
    fine_geography   = clean_geography(  fine_geography)
    coarse_geography = clean_geography(coarse_geography)
    for geography in [fine_geography, coarse_geography]:
        assert geography in geography_order, 'Invalid geography'
    fine_idx = geography_order.index(fine_geography)
    coarse_idx = geography_order.index(coarse_geography)
    assert fine_idx > coarse_idx, 'Invalid relationship between geographies'
    
    #We iterate through the geography order until we find the target fine
    # geography, multiplying the matrices in a chained fashion:
    _coarse_geography = coarse_geography
    for _fine_idx in range(coarse_idx+1, fine_idx+1):
        #Get the firt lap geography:
        _fine_geography = geography_order[_fine_idx]
        #Get the matrix:
        _C = ss.load_npz(f'{matrix_dir}{_fine_geography}-{_coarse_geography}.npz')
        #If we have stored geographies, we multiply, otherwise update:
        C = _C @ C if _fine_idx != coarse_idx+1 else _C
        _coarse_geography = _fine_geography
    
    #Potentially ignore Puerto Rico:
    if ignore_PR:
        row_indices = collect_non_PR_indices(  fine_geography)
        col_indices = collect_non_PR_indices(coarse_geography)
        C = C[row_indices, :][:, col_indices]
        
    #Potentially return as a dictionary of fine_geography>coarse_geography:
    if as_dict:
        C_coo = C.tocoo()
        C = {row: col for row, col in zip(C_coo.row, C_coo.col)}        

    return C

#########################
#2. Functions to get demographic data:
normalization = {'Population':['Race', 'Sex', 'Age', 'Age (Coarse)', 'Urbanization'],
                 'Population (for poverty level)':['Poverty Level', 'Poverty Level (Binary)'],
                 'Population (for marital status)':['Marital Status'],
                 'Population (for education)':['Education', 'Education (Coarse)'],
                 'Households':['Household Income', 'Children'],
                 'Households (occupied)':['Household Tenure', 'Household Size']}
reversed_normalization = {value: key for key, values in normalization.items() for value in values}

def get_demographics(geography, features=None,
                     years=None,ignore_PR=False, 
                     source='ACS',
                     moe=False, pct=False,
                     directory=f'{vars._census_demographics_dir}processed/'):
    """
    Function that collects the demographics from Census or ACS
    """
    #Process the inputs:
    if years is not None and type(years) != list: years = [int(years)]
    geography = geography.upper()
    if geography in ['CBG', 'BLOCK_GROUP']: geography = 'BLOCKGROUP'
    assert source == 'ACS', 'Only implemented for ACS so far'
    
    #First, collect the raw, multi-index dataframe and make it into a dictionary:
    raw_df = pd.read_csv(f"{directory}ACS5/{geography.lower()}_{'MoE' if moe else 'ESTIMATE'}.csv", header=[0,1,2])
    raw_dict = {year: raw_df.loc[:,year] for year in raw_df.columns.levels[0] if years is None or int(year) in years}
    
    #In case we are getting moes but want percent, we still need the estimates:
    if moe and pct:
        raw_est_df = pd.read_csv(f"{directory}ACS5/{geography.lower()}_ESTIMATE.csv", header=[0,1,2])
        est_dict = {year: raw_est_df.loc[:,year] for year in raw_est_df.columns.levels[0] if years is None or int(year) in years}
    
    #Collect the Puerto Rico indices if we will need them:
    if ignore_PR: non_PR_indices = collect_non_PR_indices(geography)
    
    #Now process each of these dataframes to select the columns:
    return_dictionary = {}
    for year, df in raw_dict.items():
        
        #First, check the index:
        index_col = ('Geography', f"{geography if geography != 'BLOCKGROUP' else 'CBG'}_idx")
        yearly_df = df.sort_values(index_col)
        assert np.all(yearly_df.index == yearly_df[index_col]), f'Indexing problem on {year} dataframe'
        if moe and pct: est_df = est_dict[year].sort_values(index_col)
    
        #Create the percentages:
        if pct:
            for col in yearly_df.columns:
                #Ignore the geography, total, and median columns:
                if col[0] not in ['Geography', 'Total'] and 'Median' not in col[1]:
                    #Find the correct denominator:
                    denominator = ('Total', reversed_normalization[col[0]])
                    #If we are getting the MoE, combine as ratio. otherwise just divide:
                    ratio = yearly_df[col]/yearly_df[denominator] if not moe else combine_moe(vals=[est_df[col], est_df[denominator]],
                                                                                              moes=[yearly_df[col], yearly_df[denominator]],
                                                                                              how='ratio-subset')
                    #Update:
                    yearly_df[col] = ratio.astype(float).fillna(1)
                    
        #Consider removing Puerto Rico:
        if ignore_PR: yearly_df = yearly_df.iloc[non_PR_indices]
            
        #If we did not pass features, return the full dictionary:
        return_dictionary[int(year)] = yearly_df if features is None else yearly_df[features].values.astype(float)
        
    return return_dictionary
    
############

def get_county_population(year=None, return_df=False, filename=f'{vars._census_demographics_dir}processed/county_pop.csv'):
    """
    Read the county population (as a vector of indices if `return_df`=False) for a year
    """
    df = pd.read_csv(filename)
    if return_df:
        return df
    else:
        assert year is not None, 'must pass year if requesting vector'
        P = df[str(year)].astype(int).values
        return P

def get_ACS1_state_to_state(year, ignore_PR=False,
                            dir=f'{vars._census_migration_dir}d02_state/processed/'):
    """
    Collects the estimate and the MoE matrix for
        ACS 1-year migration in the year-1 to year
        period between states.
    """
    #Collect the ACS data:
    est = pd.read_csv(f'{dir}{year-1}-{year}/estimates.csv', index_col='State').astype(float).values
    moe = pd.read_csv(f'{dir}{year-1}-{year}/moes.csv', index_col='State').astype(float).values
    
    #Clean:
    if ignore_PR:
        non_PR_indices = collect_non_PR_indices('state')
        est = est[non_PR_indices][:,non_PR_indices]
        moe = moe[non_PR_indices][:,non_PR_indices]

    return est, moe

def get_ACS1_state_inmigration(year, ignore_PR=False, rate=True,
                               dir=f'{vars._census_migration_dir}d02_state/processed/'):
    """
    Collects the estimate and the MoE for
        ACS 1-year in-migration in the year-1 to year
        period per state.
    """
    #Collect the ACS data:
    inrates = pd.read_csv(f"{dir}{year-1}-{year}/in-migration{'_rate' if rate else ''}.csv", index_col='State').astype(float)
    #Clean:
    if ignore_PR:
        non_PR_indices = collect_non_PR_indices('state')
        inrates = inrates.iloc[non_PR_indices]

    return inrates


def get_ACS1_state_nonmovers(year, ignore_PR=False,
                               dir=f'{vars._census_migration_dir}d02_state/processed/'):
    """
    Collects the estimate and the MoE for
        ACS 1-year non-movers in the year-1 to year
        period per state.
    """
    #Collect the ACS data:
    file_dir = f'{dir}{year-1}-{year}/'
    nonmovers = pd.read_csv(f"{file_dir}same-house.csv", index_col='State').astype(float)
    
    #Clean:
    if ignore_PR:
        non_PR_indices = collect_non_PR_indices('state')
        nonmovers = nonmovers.iloc[non_PR_indices]

    return nonmovers

def get_ACS5_county_to_county(year, ignore_PR=False,
                              dir=f'{vars._census_migration_dir}d01_county/processed/'):
    """
    Collects the estimate and the MoE matrix for
        ACS 5-year migration in the year-4 to year
        period.
    """
    #Collect the ACS data:
    file_dir = f'{dir}{year-4}-{year}/'
    est = ss.load_npz(f'{file_dir}estimates.npz')
    moe = ss.load_npz(f'{file_dir}moes.npz')
    
    #Clean:
    if ignore_PR:
        non_PR_indices = collect_non_PR_indices('county')
        est = est[non_PR_indices,:][:,non_PR_indices]
        moe = moe[non_PR_indices,:][:,non_PR_indices]
        
    return est, moe

def get_ACS5_county_inmigration(year, ignore_PR=False, rate=True,
                               dir=f'{vars._census_migration_dir}d01_county/processed/'):
    """
    Collects the estimate and the MoE for
        ACS 5-year in-migration in the year-4 to year
        period per county.
    """
    #Collect the ACS data:
    file_dir = f'{dir}{year-4}-{year}/'
    inrates = pd.read_csv(f"{file_dir}in-migration{'_rate' if rate else ''}.csv", index_col='idx').astype(float)
    inrates.index = inrates.index.astype(int)
    
    #Clean:
    if ignore_PR:
        non_PR_indices = collect_non_PR_indices('county')
        inrates = inrates.iloc[non_PR_indices]

    return inrates

def get_ACS5_county_nonmovers(year, ignore_PR=False,
                               dir=f'{vars._census_migration_dir}d01_county/processed/'):
    """
    Collects the estimate and the MoE for
        ACS 5-year non-movers in the year-4 to year
        period per county.
    """
    #Collect the ACS data:
    file_dir = f'{dir}{year-4}-{year}/'
    nonmovers = pd.read_csv(f"{file_dir}same-house.csv", index_col='idx').astype(float)
    nonmovers.index = nonmovers.index.astype(int)
    
    #Clean:
    if ignore_PR:
        non_PR_indices = collect_non_PR_indices('county')
        nonmovers = nonmovers.iloc[non_PR_indices]

    return nonmovers

def get_IRS_county_to_county(year, ignore_PR=False,
                             directory=f'{vars._IRS_migration_dir}processed/'):
    """
    Collects the estimate of IRS county-to-county migration
        from previous year (year-1) to current year (year), as
        well as estimate of inbound flows to each county [those
        may not add up to the column sum of the matrix due to 
        censoring procedures].
    """
    matrix = ss.load_npz(f'{directory}{year-1}-{year}.npz')
    inbound = pd.read_csv(f'{directory}/inflow.csv', usecols=[str(year), 'idx'], dtype='float')
    inbound_arr = inbound.sort_values('idx')[str(year)].values
    if ignore_PR:
        non_PR_indices = collect_non_PR_indices('county')
        matrix = matrix[non_PR_indices,:][:,non_PR_indices]
        inbound_arr = inbound_arr[non_PR_indices]
    return matrix, inbound_arr

def get_PEP_state_components(year, ignore_PR=True, PEP_file='PEP.csv'):
    """
    Load PEP populations components for states in the period year-1 to year
    """
    
    #Load the dataframe:
    PEP_state_df = pd.read_csv(f'{vars._census_demographics_dir}processed/state_{PEP_file}', index_col='STATE')
    immigrant_df = pd.read_csv(f'{vars._census_migration_dir}d02_state/processed/{year-1}-{year}/immigrants.csv', index_col='State')
    
    #Consider cleaning:
    if ignore_PR:
        non_PR_indices = collect_non_PR_indices('state')
        PEP_state_df = PEP_state_df.iloc[non_PR_indices]
        immigrant_df = immigrant_df.iloc[non_PR_indices]
        
    #Select the yearly columns of components **leading to that year**:
    #P0 = PEP_state_df.loc[:,f'POPESTIMATE{year-1}'].astype(float).fillna(0).values
    P1 = PEP_state_df.loc[:,f'POPESTIMATE{year}'].astype(float).fillna(0).values
    births = PEP_state_df.loc[:,f'BIRTHS{year}'].astype(float).fillna(0).values
    deaths = PEP_state_df.loc[:,f'DEATHS{year}'].astype(float).fillna(0).values
    intmig = PEP_state_df.loc[:,f'INTERNATIONALMIG{year}'].astype(float).fillna(0).values
    #dommig = PEP_state_df.loc[:,f'DOMESTICMIG{year}'].astype(float).fillna(0).values
    #i.e. P0 + births - deaths + domestic_migration + international_migration + (residual) = P1
    
    #Get the estimate for number of immigrants (ACS) and deduce emmigrants:
    immigr = immigrant_df['Estimate'].astype(float).fillna(0).values
    emmigr = immigr - intmig
    
    #Consolidate:
    components = {'P1':P1, 'births':births, 'deaths':deaths, 'immigr':immigr, 'emmigr':emmigr}
    
    return components

def get_census2010_CBG_demographics(col=None, ignore_PR=False,
                                    directory=f'{vars._census_demographics_dir}processed/'):
    """
    Collects the demographic data for the 2010 Census,
        per CBG. If a single column is requested, the result
        is retrieved as a numpy array (indexed per CBG). Otherwise,
        return the full dataframe.
    The `ignore_PR` parameter will only work for when a column is passed.
    """
    #Collect the full dataframe:
    demographics_df = pd.read_csv(f'{directory}CBG_Census2010.csv', index_col='CBG_idx')
    demographics_df = demographics_df.sort_index()
    #Get the column:
    if col is not None:
        demographics_arr = demographics_df[col].values
        if ignore_PR:
            non_PR_indices = collect_non_PR_indices('blockgroup')
            demographics_arr = demographics_arr[non_PR_indices]
        return demographics_arr
    else:
        return demographics_df

def get_ACS5_TRACT_demographics(year, col=None, ignore_PR=False,
                                directory=f'{vars._census_demographics_dir}processed/',
                                source='ACS'):
    """
    Collects the demographic data for the ACS 5-year estimates,
        per trtac. If a single column is requested, the result
        is retrieved as a numpy array (indexed per tract). Otherwise,
        return the full dataframe.
    The `ignore_PR` parameter will only work for when a column is passed.
    """
    #Collect the full dataframe:
    demographics_df = pd.read_csv(f'{directory}ACS5/TRACT_{year}.csv', index_col='TRACT_idx')
    demographics_df = demographics_df.sort_index()
    #Get the column:
    if col is not None:
        demographics_arr = demographics_df[col].values
        if ignore_PR:
            non_PR_indices = collect_non_PR_indices('tract')
            demographics_arr = demographics_arr[non_PR_indices]
        return demographics_arr
    else:
        return demographics_df

def get_ACS5_CBG_demographics(year, col=None,ignore_PR=False,
                                directory=f'{vars._census_demographics_dir}processed/',
                              source='ACS'):
    """
    Collects the demographic data for the ACS 5-year estimates,
        per CBG. If a single column is requested, the result
        is retrieved as a numpy array (indexed per tract). Otherwise,
        return the full dataframe.
    """
    print('DEPRECATING THIS, USE `get_CBG_demographics_ACS5` instead')
    assert year >= 2013 and source != 'NHGIS', "No ACS 5-year data at the CBG level before 2013 if not using source NHGIS"
    
    #Collect the full dataframe:
    if source == 'ACS':
        demographics_df = pd.read_csv(f'{directory}ACS5/BLOCKGROUP_{year}.csv', index_col='CBG_idx')
        demographics_df = demographics_df.sort_index()
        #Get the column:
        if col is not None:
            demographics_arr = demographics_df[col].values
            if ignore_PR:
                non_PR_indices = collect_non_PR_indices('blockgroup')
                demographics_arr = demographics_arr[non_PR_indices]
            return demographics_arr
        else:
            if ignore_PR:
                non_PR_indices = collect_non_PR_indices('blockgroup')
                demographics_df = demographics_df.iloc[non_PR_indices]
            return demographics_df

def get_CBG_demographics_ACS5(features=None, years=None, ignore_PR=False,
                              directory=f'{vars._census_demographics_dir}processed/'):
    
    demographics_df = pd.read_csv(f'{directory}ACS5-NHGIS/BLOCKGROUP.csv').sort_values('CBG_idx')
    assert np.all(demographics_df.index == demographics_df['CBG_idx']), 'Indexing problem, review saved dataframe'
    
    if features is not None:
        if type(features) == str: features = [features]
        demographics_df = demographics_df.loc[:,[col for col in demographics_df.columns if col[5:] in features]]
    if ignore_PR:
        non_PR_indices = collect_non_PR_indices('blockgroup')
        demographics_df = demographics_df.iloc[non_PR_indices]

    #If we didn't pass any features, return the full dataframe:
    #TODO: allow years
    if features is None:
        return demographics_df
        
    #Select per year:
    years_in_df = [int(col[:4]) for col in demographics_df.columns]
    if years is not None:
        if type(years) == int or type(years)==float: years=[years]

    #Return as a features dictionary:
    feature_dict = {}
    for feature in features:
        selected_df = demographics_df.loc[:,[col for col in demographics_df.columns if col[5:]==feature]]
        if years is not None:
            feature_arr = selected_df.loc[:,[f'{y}_{feature}' for y in years if y in years_in_df]].values
            feature_dict[feature] = dict(zip(years, feature_arr.T))
        else:
            feature_dict[feature] = selected_df

    #Simplify the return if possible:
    if features is not None and len(features) == 1:
        if years is not None and len(years) == 1:
            return feature_dict[features[0]][years[0]]
        else:
            return feature_dict[features[0]]
    else:
        return feature_dict

def get_urbanfrac_CBG(include_clusters=True, ignore_PR=False,
                      directory=f'{vars._census_demographics_dir}processed/'):
    """
    Collects the urbanized fraction of the CBG. If `include_clusters`, urbanized means
        within an Urbanized Area (50,000 people or more) as well as an Urban Cluster
        (2,500 people or more). If not, only urbanized areas.
    """
    urbanization_df = pd.read_csv(f'{directory}CBG_Urbanization.csv', index_col='CBG_idx')
    urbanization_df = urbanization_df.sort_index()
    #Get the column:
    col = 'urban' if include_clusters else 'urban_exclusively'
    urbanization_arr = urbanization_df[f'{col}_fraction'].values
    #Remove PR:
    if ignore_PR:
        non_PR_indices = collect_non_PR_indices('blockgroup')
        urbanization_arr = urbanization_arr[non_PR_indices]

    return urbanization_arr


############################################################################################

def crosswalk_geographies(raw_df, GEOID_column='GEOID', geography='CBG'):
    """
    Crosswalk Census Block Group or Tract GEOIDs to match our naming
    """
    df = raw_df.copy()
    geography = geography.lower()
    geoid_lengths = {'tract':11, 'cbg':12}
    assert GEOID_column in df.columns, 'DataFrame must contain column GEOID'
    assert np.all(df[GEOID_column].str.len() == geoid_lengths[geography]), f'GEOID column must be a string of length {geoid_lengths[geography]} for {geography} level'
    
    #Fix renames inherited from county:
    df[GEOID_column] = df[GEOID_column].apply(lambda x: x.replace('46102', '46113', 1) if x.startswith('46102') else x)
    df[GEOID_column] = df[GEOID_column].apply(lambda x: x.replace('02158', '02270', 1) if x.startswith('02158') else x)
    df[GEOID_column] = df[GEOID_column].apply(lambda x: x.replace('51019050100', '51515050100', 1) if x.startswith('51019050100') else x)
    #if geography == 'county': df[GEOID_column] = df[GEOID_column].apply(lambda x: x.replace('51019', '51515', 1))

    #Fix renames in Madison county (after 2011):
    df[GEOID_column] = df[GEOID_column].apply(lambda x: x.replace('36053030101', '36053940101', 1) if x.startswith('36053030101') else x)
    df[GEOID_column] = df[GEOID_column].apply(lambda x: x.replace('36053030102', '36053940102', 1) if x.startswith('36053030102') else x)
    df[GEOID_column] = df[GEOID_column].apply(lambda x: x.replace('36053030103', '36053940103', 1) if x.startswith('36053030103') else x)
    df[GEOID_column] = df[GEOID_column].apply(lambda x: x.replace('36053030200', '36053940200', 1) if x.startswith('36053030200') else x)
    df[GEOID_column] = df[GEOID_column].apply(lambda x: x.replace('36053030300', '36053940300', 1) if x.startswith('36053030300') else x)
    df[GEOID_column] = df[GEOID_column].apply(lambda x: x.replace('36053030401', '36053940401', 1) if x.startswith('36053030401') else x)
    df[GEOID_column] = df[GEOID_column].apply(lambda x: x.replace('36053030402', '36053940700', 1) if x.startswith('36053030402') else x)
    df[GEOID_column] = df[GEOID_column].apply(lambda x: x.replace('36053030403', '36053940403', 1) if x.startswith('36053030403') else x)
    df[GEOID_column] = df[GEOID_column].apply(lambda x: x.replace('36053030600', '36053940600', 1) if x.startswith('36053030600') else x)

    #Fix renames and geography changes in Oneida county (after 2011):
    df[GEOID_column] = df[GEOID_column].apply(lambda x: x.replace('36065024800', '36065940000', 1) if x.startswith('36065024800') else x)
    df[GEOID_column] = df[GEOID_column].apply(lambda x: x.replace('36065024700', '36065940100', 1) if x.startswith('36065024700') else x)
    df[GEOID_column] = df[GEOID_column].apply(lambda x: x.replace('36065024900', '36065940200', 1) if x.startswith('36065024900') else x)

    #Fix renames in Pima County AZ (after 2012):
    df[GEOID_column] = df[GEOID_column].apply(lambda x: x.replace('04019002704', '04019002701', 1) if x.startswith('04019002704') else x)
    df[GEOID_column] = df[GEOID_column].apply(lambda x: x.replace('04019002906', '04019002903', 1) if x.startswith('04019002906') else x)
    df[GEOID_column] = df[GEOID_column].apply(lambda x: x.replace('04019004118', '04019410501', 1) if x.startswith('04019004118') else x)
    df[GEOID_column] = df[GEOID_column].apply(lambda x: x.replace('04019004121', '04019410502', 1) if x.startswith('04019004121') else x)
    df[GEOID_column] = df[GEOID_column].apply(lambda x: x.replace('04019004125', '04019410503', 1) if x.startswith('04019004125') else x)
    df[GEOID_column] = df[GEOID_column].apply(lambda x: x.replace('04019005200', '04019470400', 1) if x.startswith('04019005200') else x)
    df[GEOID_column] = df[GEOID_column].apply(lambda x: x.replace('04019005300', '04019470500', 1) if x.startswith('04019005300') else x)

    #Fix LA County (after 2012):
    df[GEOID_column] = df[GEOID_column].replace({'060371370001':'060379304011'}) #CBG
    df[GEOID_column] = df[GEOID_column].replace({'060371370002':'060378002043'}) #CBG
    df[GEOID_column] = df[GEOID_column].replace({'06037137000':'06037930401'})   #Tract---majority

    #Include a Census Tract in Richmond county:
    new_geoid = '360850089000'[:geoid_lengths[geography]]
    if new_geoid not in df[GEOID_column].values:
        df.loc[len(df), GEOID_column] = new_geoid
    
    return df

############################################################################################

def resolve_ACS(ACS_CBG_populations,
                Census_2010=None, use_ACS_2010=False, use_NNLS=False):
    """
    From ACS 5-year estimates, we construct yearly population estimates
    """
    #The response variable are the ACS rolling averages:
    Y = np.vstack(list(ACS_CBG_populations.values()))
    Y_B = np.vstack([Census_2010]+[Y]) if Census_2010 is not None else Y
    Y[0] = Y_B[0]
    M = Y.shape[0] #Number of ACS years we have

    #The design matrix creates 5-year rolling averages:
    A = np.zeros((M, M + 4))
    for i in range(M):
        A[i, i:i+5] = 1/5  # Set five consecutive 1's starting at position i

    #For the design matrix with Census AND 2010 ACS:
    B = A[:,3:]/A[:,3:].sum(axis=1)[:,np.newaxis]
    B = np.vstack([np.zeros(B.shape[1]), B])
    B[0,1] = 1

    #For the design matrix with just Census:
    A = A[:,4:]/A[:,4:].sum(axis=1)[:,np.newaxis]

    #Compute the solution:
    x, y = (B, Y_B) if use_ACS_2010 else (A, Y)
    Xhat = np.array([nnls(x, _y)[0] for _y in y.T]).T if use_NNLS else np.linalg.solve(x, y)

    #Return as a dictionary:
    ACS_hat_CBG = dict(zip(range(2020-Xhat.shape[0], 2020), Xhat))

    return ACS_hat_CBG


############################################################################################

def process_state_to_state_migration(year):
    """
    Process the state-to-state migation file downloaded from
    
    https://www.census.gov/data/tables/time-series/demo/geographic-mobility/state-to-state-migration.html
    
    into an OD matrix containing data on flows from year-1 to year (row to column). 
    """
    
    #Read:
    df = pd.read_excel(f'{vars._census_migration_dir}d02_state/raw/state_to_state_migrations_table_{year}.xls')
    
    #Find the rows and columns containing state indices:
    states = vars._states_ACS_keys
    col_header = df.iloc[:,vars._states_ACS_col_header[year]]
    row_header = df.iloc[vars._states_ACS_row_header[year],:]
    for s in states:
        assert (col_header == s).sum() == 1, f'{s} included more than once (or not included) in header column'
        assert (row_header == s).sum() == 1, f'{s} included more than once (or not included) in header row'
        
    #Select rows and columns to keep:
    rows_to_keep = col_header.isin(states) 
    cols_to_keep = row_header.isin(states)
    moes_to_keep = cols_to_keep.shift(1, fill_value=False)
    
    #Create dataframes and give them indices (transpose to ensure flows t0 -> t1):
    est_df = df.iloc[np.flatnonzero(rows_to_keep), np.flatnonzero(cols_to_keep)].transpose()
    moe_df = df.iloc[np.flatnonzero(rows_to_keep), np.flatnonzero(moes_to_keep)].transpose()
    for _df in est_df, moe_df:
        _df.index = [state if state != "District of Columbia " else "District of Columbia" for state in states]
        _df.columns = _df.index
    
    #For years after 2010, we must collect people who moved within state/stayed within state separately:
    if year >= 2010:
        #People who stayed in the same house:
        same_house_idx = (df.iloc[vars._states_ACS_row_header[year]-1,:].isin(vars._same_house_cols)).reset_index(drop=True).idxmax()
        same_house_est = df.iloc[np.flatnonzero(rows_to_keep),same_house_idx].values
        same_house_moe = df.iloc[np.flatnonzero(rows_to_keep),same_house_idx+1].values
        #People who moved within the same state:
        same_state_idx = (df.iloc[vars._states_ACS_row_header[year]-1,:].isin(vars._same_state_cols)).reset_index(drop=True).idxmax()
        same_state_est = df.iloc[np.flatnonzero(rows_to_keep),same_state_idx].values
        same_state_moe = df.iloc[np.flatnonzero(rows_to_keep),same_state_idx+1].values
        #Diagonal:
        diag_est = same_state_est + same_house_est
        diag_moe = np.sqrt((same_state_est**2 + same_house_est**2).astype(float))
        #Place in the diagonal:
        np.fill_diagonal(est_df.values, diag_est)
        np.fill_diagonal(moe_df.values, diag_moe)
        #We can also save this into a df:
        same_house_df = pd.DataFrame([same_house_est, same_house_moe],
                                     columns=vars._states_ACS_keys,
                                     index=['Estimate', 'MoE']).T
        #We also save the immigrants:
        abroad_df = df.loc[rows_to_keep].iloc[:,[122, 123]]
        abroad_df.index = est_df.index
        abroad_df.index.name = 'State'
        abroad_df.columns = ['Estimate', 'MoE']
    
    #We should not have any nans now:
    for _df in est_df, moe_df:
        assert _df.shape == (52, 52), "Coarse areas mismatch"
        assert not _df.isnull().values.any(), f"Null values in the {year} dataframe"
    
    #Save:
    dir = f'{vars._census_migration_dir}d02_state/processed/{year-1}-{year}/'
    if not os.path.exists(dir): os.makedirs(dir)
    est_df.astype(float).to_csv(f'{dir}estimates.csv')
    moe_df.astype(float).to_csv(f'{dir}moes.csv')
    if year >= 2010: same_house_df.astype(float).to_csv(f'{dir}same_house.csv')
    if year >= 2010: abroad_df.astype(float).to_csv(f'{dir}immigrants.csv')

    return est_df, moe_df
    
############################################################################################
#1. Spatial data:

def read_census_geography_shapefiles(years=None, geography='counties'):
    """
    Collects geodataframes of census geographies for the years 2000, 2010-20 into
        a dictionary where keys are years.
    
    Parameters
    ----------
    years : list, None, or int
        migration year in question, data for 2000 and 2010 to 2020 is available
        If None, get a dictionary of all years.
        
    geography : `counties` or `ZIP`
        name of the file 
        
    Returns
    ----------
    dict
    """
    assert geography in ['counties', 'ZIP']
    
    if years is None: years = [2000] + list(range(2010, 2021))
    if type(years) == int: years = [years]
    
    gdf_dict = {}
    for year in years:
        gdf_dict[year] = gpd.read_file(f'{vars._census_spatial_dir}{geography}.gpkg', layer=str(year))
    
    return gdf_dict

def clean_column_names(gdf):
    col_renames = {col:col[:-2] for col in gdf.columns if col != 'geometry'}
    renamed_gdf = gdf.rename(col_renames,axis=1)
    if 'CNTYIDFP' in renamed_gdf.columns: renamed_gdf = renamed_gdf.rename({'CNTYIDFP':'GEOID'}, axis=1)
    return renamed_gdf

def save_census_shapefiles(geography='counties', verbose=False):
    """
    Creates a geodataframe whose layers are US geographies from years 2010-20.
        Should be called only once.
    
    Parameters
    ----------        
    geography : `counties` or `ZIP`
        geography to collect, which will be the name of the file 
        
    Returns
    ----------
    dict
        dictionary where each entry is the corresponding year (layer name in the
        geopackage file)
    """
    assert geography in ['counties', 'ZIP']

    #Counties:
    if geography == 'counties':
        geographies_dict = save_census_counties()
        
    #ZIP Code tabulation areas:
    if geography == 'ZIP':
        geographies_dict = save_census_ZIP(verbose)     
        
    return geographies_dict

def save_census_counties():
    """
    Creates a dictionary whose keys are years 2000, 2010-2020 and
        entries are county shapefiles in the US.
    """
    #Collect the geographies with pygris.counties:
    years = [2000, 2010] + list(range(2011, 2023))
    geographies_dict = {year: counties(year=year) for year in years}
    
    #2000 and 2010 counties have 00 and 10 on column names that we must remove:
    for year in [2000, 2010]: geographies_dict[year] = clean_column_names(geographies_dict[year])
    
    #Remove territories (Except PR) from these gdfs:
    geographies_dict = {year: gdf[(gdf.STATEFP.astype(int)<=56)|(gdf.STATEFP.astype(int)==72)] for year,gdf in geographies_dict.items()}

    #Filter the columns we will use and save:
    for year, gdf in geographies_dict.items():
        filtered_gdf = gdf.loc[:,['STATEFP', 'COUNTYFP', 'GEOID', 'NAMELSAD', 'geometry']]
        renames = {'STATEFP':'state_code', 'COUNTYFP':'county_code', 'NAMELSAD':'county'}
        geographies_dict[year] = filtered_gdf.rename(renames, axis=1)
        geographies_dict[year].to_file(f'{vars._census_spatial_dir}counties.gpkg', driver='GPKG', layer=str(year))

    return geographies_dict
    
def match_county_to_ZIP(gdf, year, area_crs=vars._area_crs):
    """
    Creates a column `county_code` on a ZIP tabulation areas gdf with the
        majority county in the file.
    """
    
    #Load the corresponding county gdf and project:
    county_gdf = gpd.read_file(f'{vars._census_spatial_dir}counties.gpkg', layer=str(year))
    county_gdf_proj = county_gdf.to_crs(area_crs)
    gdf_proj = gdf.to_crs(area_crs)
    
    #Clip the ZIP code gdf to the county gdf (keeps only 50 states + PR):
    gdf_clipped = gpd.clip(gdf_proj, county_gdf_proj, keep_geom_type=True).reset_index(drop=True)
    
    #Compute the areas of intersection between the ZIP code gdf and the county gdf:
    intersections = gpd.overlay(county_gdf_proj[['GEOID', 'geometry']], gdf_clipped, keep_geom_type=True)
    intersections['area'] = intersections.area
    
    #Select the majoritary county:
    majority_county_idx = intersections.groupby(['ZIP_code'])['area'].idxmax()
    ZIP_to_county = intersections.loc[majority_county_idx].set_index('ZIP_code')['GEOID']
    gdf['county_code'] = gdf['ZIP_code'].map(ZIP_to_county)

    return gdf
    
def save_census_ZIP(verbose=False):
    """
    Creates a dictionary whose keys are years 2000, 2010-2020 and
        entries are ZIP tabulation areas shapefiles in the US.
    """
    #Collect the geographies with pygris.zctas:
    years = [2000, 2010] + list(range(2012, 2021))
    geographies_dict = {year: zctas(year=year) for year in years}

    #Fill in unavailable items with previously avaiable year:
    geographies_dict[2011] = geographies_dict[2010] #zctas not available for 2011
    #for year in range(2006, 2010): geographies_dict[year] = geographies_dict[2000]

    #Sort dicitonary keys:
    geographies_dict = dict(sorted(geographies_dict.items()))

    #Remove year 00 10 and 20 from columns:
    for year, gdf in geographies_dict.items(): geographies_dict[year] = clean_column_names(gdf)

    #Initialize the API (for population):
    c = Census(vars._census_api_key)
    
    #Filter the columns we will use and save:
    for year, gdf in tqdm(geographies_dict.items()) if verbose else geographies_dict.items():
        
        #Filter:
        filtered_gdf = gdf.loc[:,['ZCTA5CE', 'geometry']]
        filtered_gdf = filtered_gdf.rename({'ZCTA5CE':'ZIP_code'}, axis=1)

        #Include the majority county code:
        matched_gdf = match_county_to_ZIP(filtered_gdf, year)
        
        #Clean the gdf from areas we found no county:
        gdf_clean = matched_gdf.dropna(how='any', axis=0)

        #Assert our ZIP codes are all correct:
        assert len(set(gdf_clean.ZIP_code.apply(len)))==1

        #Include the population using ACS data:
        if year > 2010:
            #Get data from API:
            ACS_data = c.acs5.state_zipcode(fields = ('B01003_001E'),
                                            state_fips = '*', zcta = '*', year=year)
            ACS_df = pd.DataFrame(ACS_data).rename({'zip code tabulation area':'ZIP_code',
                                                    'B01003_001E':'population',
                                                    'state':'state_code'}, axis=1)
            ACS_df['population'] = ACS_df['population'].astype(int)
            
            #Assert our ZIP codes are all correct:
            assert set(ACS_df.ZIP_code.apply(len))=={5}
    
            #Merge through the ZIP code column:
            gdf_clean = gdf_clean.merge(ACS_df, on='ZIP_code', how='inner')
            gdf_clean = gdf_clean.dropna(how='any', axis=0)

        #Sort values by ZIP code:
        gdf_sorted = gdf_clean.sort_values(by='ZIP_code').reset_index(drop=True)

        #Save and update:
        geographies_dict[year] = gdf_sorted
        gdf_sorted.to_file(f'{vars._census_spatial_dir}ZIP.gpkg', driver='GPKG', layer=str(year))
        
    return geographies_dict
    
############################################################################################
# 2.State migrations

def process_state_by_state_migration(year=2016,
                                     file_directory=f'{vars._census_migration_dir}d02_state/',
                                     file_name='state_to_state_migrations_table_',
                                     file_ext='xls'):
    """
    ATTENTION: Not verified for years besides 2016-2020
    
    Reads the state to state migrations table into a 52 x 52 matrix from
    https://www.census.gov/data/tables/time-series/demo/geographic-mobility/state-to-state-migration.html
    
    Parameters
    ----------
    year : int
        migration year in question, data for 2005 to 2022 is available
        
    file_directory : str
        name of the file 
        
    Returns
    ----------
    pd.DataFrame
        dataframe with 52 rows and 52 columns corresponding to the number of
        people who moved from state A (row) to state B (column) on a given year
    """
    #Collect the raw data:
    df = pd.read_excel(f'{file_directory}{file_name}{year}.{file_ext}')

    #We collect the rows relative to each state migration (must exclude rows in the middle where the header is repeated
    #  and drop nan rows)
    state_rows=[x for x in range(10, 77) if x not in [43, 44, 45]]
    df_states = df.iloc[state_rows].dropna(how='all', axis=0).reset_index(drop=True)
    
    #The first column corresponds to people's current state of residence:
    df_states.columns.values[0] = 'Current Residence'
    
    #The other columns can be interpreted from rows 4,5,6 and correspond to Estimates and MOE for regular stats and then
    # for the matrix. For example let's collect the diagonal entries and the population:
    diagonal_column = df_states.iloc[:,5].values
    total_population = df_states.iloc[:,1].values
    
    #Now we filter out the columns corresponding to people mocing between states:
    previous_residence_cols = list(range(9, 121))
    df_states.columns.values[previous_residence_cols] = df.iloc[5, 9:121].values
    
    #For abroad information, select out PR:
    df_states.columns.values[124] = 'Puerto Rico'
    inflow_from_abroad = df_states.iloc[:,122] - df_states.iloc[:,124]
    
    #Select only those columns that represent state estimates:
    state_cols = df_states['Current Residence'].values
    cols_to_keep = [c for c in df_states.columns if c=='Current Residence' or c in state_cols]
    df_states_no_diag = df_states[cols_to_keep].set_index('Current Residence')
    
    #Small issue of removing space after DC:
    df_states_no_diag.rename({'District of Columbia ':'District of Columbia'}, axis=0, inplace=True)
    df_states_no_diag.rename({'District of Columbia ':'District of Columbia'}, axis=1, inplace=True)
    
    #Get the matrix values:
    F = df_states_no_diag.values.astype(float)
    np.fill_diagonal(F, diagonal_column)
    df_states_processed = pd.DataFrame(F, columns=df_states_no_diag.columns, index=df_states_no_diag.index)

    return df_states_processed

############################################################################################
# 3.County migrations

def get_county_to_county_matrix(t0,
                                file_directory=f'{vars._census_migration_dir}d01_county/',
                                extension=None):
    """
    Collects a county to county migration matrix from ACS data where rows and columns are
        both county FIPS codes (GEOIDs). Dataframe is always square, and diagonal entries
        represent either movement within county or no movement.

    Parameters
    ----------
    t0 : int
        initial year, ACS does county to county migration in 5-year intervals so
        will collect data from t0 to t0+4 inclusive.
        
    Returns
    ----------
    pd.DataFrame
        ~3220 x 3220 dataframe (actual number depends of year)
    """

    #First we read the ACS data into a non-zero flow list:
    if extension is None: extension='xls' if t0 <= 2008 else 'xlsx' 
    nonzero_flow_df = collect_non_zero_county_flows(t0, file_directory, extension)

    #Now we get the FIPs of all counties in the United States at that year:
    counties_gdf = read_county_shapefiles(years=t0+4)[t0+4]
    all_counties = counties_gdf.GEOID.astype(str).values
    N = len(all_counties)
    all_counties.sort()

    #Create an empty square dataframe:
    flow_df_US = pd.DataFrame(np.zeros((N, N)), index=all_counties, columns=all_counties)

    #Update non-zero values:
    def update_flow_df_US(origin, destination, flow):
        if str(origin) in all_counties and str(destination) in all_counties:
            flow_df_US.loc[origin, destination] = flow
        return None
    _ = nonzero_flow_df.apply(lambda row: update_flow_df_US(row.origin, row.destination, row.flow), axis=1)

    #Ensure we did not add any county:
    assert N == len(flow_df_US)
    assert N == len(flow_df_US.T)

    flow_df_US.index.rename('origin', inplace=True)

    return flow_df_US

def process_census_county_sheet(full_df):
    """
    Process the ACS county-to-county migration xlxs file into a 
       pandas dataframe
    """
    #Find the rows where we have data:
    data = full_df[0].str.isdigit().fillna(False).values
    #Collect the data values:
    data_df = full_df[data].reset_index(drop=True)
    #Rename the columns:
    data_df.columns = vars._county_to_county_columns
    return data_df

def get_flow_df(df):
    """
    Turns an ACS county to county migration file into an O-D-Flow dataframe
    """
        
    #Get OD codes and flow. We cut the first zero of state codes and call abroad 99_999:
    origin      = (df['previous_state_code'  ].apply(lambda x: x[1:])   + df['previous_county_code']).fillna('99999')
    destination =  df['current_state_code'   ].apply(lambda x: x[1:])   + df['current_county_code']
    flow        =  df['county_to_county_flow']
    
    #Build a dataframe:
    flow_df = pd.DataFrame([origin, destination, flow], index=['origin', 'destination', 'flow']).T
    
    #We need to add entries that come from abroad:
    flow_df_grouped = flow_df.groupby(['origin', 'destination']).sum()['flow'].reset_index()

    #We also need to add the diagonal entries as non-movers + movers within county
    diagonal_cols = ['current_county_nonmovers', 'current_county_movers_within_county']
    diagonal_values = df.groupby('current_county_code').first()[diagonal_cols].sum(axis=1)
    GEOID = df['current_state_code'].iloc[0][1:] + diagonal_values.index
    flow_df_diagonal = pd.DataFrame([GEOID, GEOID, diagonal_values.values], index=['origin', 'destination', 'flow']).T

    return pd.concat([flow_df_grouped, flow_df_diagonal],ignore_index=True)

def read_county_matrices(years,
                         filedir=vars._census_migration_dir):
    """
    Collects 4 year county to county migration csvs into
        a dictionary where keys are years.
    """
    
    if years is None: years = list(range(2006, 2017))
    if type(years) == int: years = [years]
    
    migration_df_dict = {}
    for year in years:
        df = pd.read_csv(f'{filedir}d01_county/{year}-{year+4}/flow_county_{year}-{year+4}.csv', index_col='origin')
        df.index = df.index.astype(str).str.zfill(5)
        migration_df_dict[year] = df
    
    return migration_df_dict


def collect_non_zero_county_flows(t0,
                                  file_directory=f'{vars._census_migration_dir}d01_county/',
                                  extension='xlsx',
                                  discarded_rows=vars._census_migration_discarded_rows):
    """
    Reads the migration files from t_0 to t_0+5 to make a dataframe
        with origin county, destination county, and number of migrants
    """
    #We read the excel file from the census as a dictionary, where the keys are current
    # states of residence:
    filepath = f'{file_directory}{t0}-{t0+4}/county-to-county-{t0}-{t0+4}-current-residence-sort.{extension}'
    df_dict = pd.read_excel(filepath, header=None, sheet_name=None)
    assert len(df_dict.keys()) == 52

    #Let's remove header, foootnotes, and rename columns:
    data_dict = {state:process_census_county_sheet(df) for state,df in df_dict.items()}
    assert len(data_dict.keys()) == 52

    #Assert that we have the correct number of removed rows (census data is not consistent across years)
    for state, df in df_dict.items():
        assert len(data_dict[state]) > 0
        diff = len(df) - len(data_dict[state])
        if state == 'Puerto Rico':  #it's inconsistent for some years
            assert diff == discarded_rows['PR'][t0], f"{state}, {t0}-{t0+4}: {diff} discarded rows"
        elif state == 'Pennsylvania':  #it's inconsistent for some years
            assert diff == discarded_rows['PA'][t0], f"{state}, {t0}-{t0+4}: {diff} discarded rows"
        else:
            assert diff == discarded_rows['States'][t0], f"{state}, {t0}-{t0+4}: {diff} discarded rows"

    #To make the matrix, we want to get origin and destination county codes as well as flows:
    flow_dict = {state: get_flow_df(df) for state, df in data_dict.items()}
    assert len(flow_dict.keys()) == 52

    #Concatenate all of the states:
    flow_df_nonzero_full = pd.concat(flow_dict.values(),ignore_index=True)
    flow_df_nonzero_US = flow_df_nonzero_full[flow_df_nonzero_full.origin != '999999'].reset_index(drop=True)

    return flow_df_nonzero_US

################################################################################################################
# Functions to process ACS and NHGIS files:
def combine_moe(vals=None, moes=None, how='sum', sparse=None):
    """
    Function to combine MoEs of ACS estimates
    """
    if sparse is None:
        sparsity = [ss.issparse(x) for x in moes]
        assert len(set(sparsity)) == 1, 'Either all moes are sparse matrices or all are array-like'
        sparse = sparsity[0]
    
    if how=='sum':
        assert moes is not None, 'For sum, MoEs must be passed'
        moe = combine_moe_sum(moes) if not sparse else combine_moe_sum_sparse(moes)

    if how=='ratio-subset':
        assert len(vals) == 2 and len(moes) == 2, 'For ratios, exactly two values + moes must be passed'
        assert np.all(vals[0] <= vals[1]), 'For subset ratios, numerator must be passed first'
        moe = combine_moe_ratio_subset(*vals, *moes)
        #Check for failures and use the general formula for those cases
        failure_idx = np.isnan(moe)
        if np.any(failure_idx):
            general_moe = combine_moe_ratio(*vals, *moes)
            moe[failure_idx] = general_moe[failure_idx]

    if how=='ratio':
        assert len(vals) == 2 and len(moes) == 2, 'For ratios, exactly two values + moes must be passed'
        moe = combine_moe_ratio(*vals, *moes)

    return moe

def combine_moe_sum(moes):
    """
    Combine the MoEs for a sum of estimates
    """
    moe = np.sqrt(sum([x**2 for x in moes]))
    return moe

def combine_moe_sum_sparse(moes):
    """
    Combine the MoEs for a sum of estimates on sparse matrices
    """
    moe = np.sqrt(sum([x.power(2) for x in moes]))
    return moe

def combine_moe_ratio_subset(numerator, denominator, numerator_moe, denominator_moe):
    """
    Combine the MoEs for a ratio where the numerator is a subset of the denominator
    """
    p = numerator/denominator
    with np.errstate(invalid='ignore'):
        moe = np.sqrt(numerator_moe**2 - (p*denominator_moe)**2)/denominator

    return moe

def combine_moe_ratio(numerator, denominator, numerator_moe, denominator_moe):
    """
    Combine the MoEs for a general ratio
    """
    p = numerator/denominator
    moe = np.sqrt(numerator_moe**2 + (p**2)*(denominator_moe**2))/denominator

    return moe

################################################################################################################

def parse_NHGIS_codebook(raw_dir, geography, year, source='ACS5', n_tables=12):
    """
    Parse the NHGIS Codebook to get column names
    """
    filepath = f'{raw_dir}Codebooks/{geography}-{source}-{year}.txt'
    with open(filepath, 'r') as file:
        lines = file.readlines()
    
    #First, determine the region of the codebook to read the table codes:
    table_idx0 = lines.index('Tables:\n')+2
    table_idx1 = table_idx0+lines[table_idx0:].index('Data Dictionary\n')-1
    table_codes = lines[table_idx0:table_idx1]

    #Extract names, Census codes, and NHGIS codes:
    table_names       = [name.strip('\n').split('.', 1)[-1].lstrip() for name in table_codes[0::len(table_codes)//n_tables]]
    table_sourcecodes = [name.strip('\n').split(':', 1)[-1].lstrip() for name in table_codes[2::len(table_codes)//n_tables]]
    table_NHGIScodes  = [name.strip('\n').split(':', 1)[-1].lstrip() for name in table_codes[3::len(table_codes)//n_tables]]
    table_dict = {f'{name} ({source_code})': NHGIS_code for name, source_code, NHGIS_code in zip(table_names, table_sourcecodes, table_NHGIScodes)}

    #Second, determine the region of the codebook to read the column codes:
    columns_idx0 = lines.index('Data Type (E):\n')
    columns_idx1 = lines.index('Data Type (M):\n')
    column_codes = np.array(lines[columns_idx0:columns_idx1]) #Will be useful for splitting
    
    #Split whenever we have a line break and create one array per table:
    splits = np.where(column_codes == ' \n')[0]
    column_codes_per_table = np.split(column_codes, splits+1)[-n_tables-1:-1]
    
    #Populate the column dictionary:
    column_dict = {}
    for table_columns in column_codes_per_table:
        code_name_pairs = [code_name_pair.strip('\n').strip(' ').split(':', 1) for code_name_pair in table_columns[4:-1]]
        code_name_dict = {pair[1].lstrip(): pair[0][-3:] for pair in code_name_pairs}
        table_code = table_columns[3].strip('\n').split(':', 1)[-1].lstrip()
        column_dict[table_code] = code_name_dict

    return table_dict, column_dict