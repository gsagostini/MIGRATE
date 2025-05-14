############################################################################################
# Functions to process wildfire data
############################################################################################

import numpy as np
import pandas as pd
import geopandas as gpd

import warnings

import sys
sys.path.append('../d03_src/')
import vars
import os

#######################################################################################################################

def get_fires_gdf(years=None,
                  fires_path=f'{vars._burn_dir}california_fire.gdb'):

    #Load the GDF (which will trigger some warnings due to old Geopandas behavior):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning)
        raw_fire_gdf = gpd.read_file(fires_path, driver='fileGDB', layer='firep23_1')
    
    #Select years:
    if years is not None:
        raw_fire_gdf = raw_fire_gdf[raw_fire_gdf.YEAR_.isin(years)].reset_index(drop=True)
    
    #Include duration:
    date_end = pd.to_datetime(raw_fire_gdf.CONT_DATE.str[:10],  format='%Y-%m-%d', errors='coerce')
    date_ala = pd.to_datetime(raw_fire_gdf.ALARM_DATE.str[:10], format='%Y-%m-%d', errors='coerce')
    raw_fire_gdf['days'] = np.clip((date_end-date_ala).dt.total_seconds()/(24*60*60), 1, a_max=None).fillna(1)
    
    #Clean:
    fire_gdf = raw_fire_gdf.loc[:,['YEAR_', 'FIRE_NAME', 'CAUSE', 'GIS_ACRES', 'days', 'geometry']]
    fire_gdf.columns = ['year', 'name', 'cause', 'acres', 'days', 'geometry']

    return fire_gdf
    
#######################################################################################################################

def collect_fire_CBG_gdf(firename, firemo, fireyear, firecounties,
                         all_fires_gdf,
                         california_gdf_dict,
                         matrix_dictionary,
                         area_threshold=0.1,
                         include_distance_controls=False):

    #Find the fire:
    all_fire = all_fires_gdf[all_fires_gdf.name.str.contains(firename.upper())&(all_fires_gdf.year == fireyear)]
    all_fire = all_fire.sort_values('acres', ascending=False).reset_index(drop=True)
    if len(all_fire) > 1:
        print(f'WARNING: more than one possible {firename} fire perimeter. Picking the largest.')
        #display(all_fire)
    fire = all_fire.iloc[[0]]
    fire_bounds = fire.bounds.values[0]

    #Fill in:
    fire_CBGs = california_gdf_dict['blockgroup'].loc[:,['index', 'GEOID', 'geometry']]
    fire_CBGs['CBG_area'] = fire_CBGs.area
    fire_CBGs['burned_area'] = fire_CBGs.intersection(fire.geometry.iloc[0]).area
    fire_CBGs['burned_frac'] = fire_CBGs['burned_area']/fire_CBGs['CBG_area']
    fire_CBGs['distance_to_fire_km'] = fire_CBGs.centroid.distance(fire.geometry.iloc[0])/1_000 #km

    #We collect the indices of CBGs:
    # 1. Directly affected by the fire (area burned > 1%)
    # 2. Not affected by the fire
    # 3. Not affected by the fire, and at least 50km from affected CBGs
    # 4. Not affected by the fire, and at most 50km from affected CBGs
    # 5. Not affected by the fire, yet within one of the two counties affected
    # 6. Not affected by the fire, yet adjacent to an affected CBG
    affected_CBGs = fire_CBGs.burned_frac >= area_threshold
    affected_union = fire_CBGs.loc[affected_CBGs].dissolve().geometry[0]
    disjoint_from_affected = fire_CBGs.geometry.disjoint(affected_union)
    affected_counties_GEOID = california_gdf_dict['county'][california_gdf_dict['county'].NAME.isin(firecounties)].GEOID.values
    CBG_idx_per_group = {}
    
    CBG_idx_per_group['CBGs within fire perimeter'] = fire_CBGs.loc[affected_CBGs, 'index'].values
    CBG_idx_per_group['CBGs neighboring perimeter'] = fire_CBGs.loc[(~disjoint_from_affected)&(~affected_CBGs), 'index'].values
    CBG_idx_per_group['CBGs within affected counties'] = fire_CBGs.loc[(~affected_CBGs)&(fire_CBGs.GEOID.str.startswith(tuple(affected_counties_GEOID))), 'index'].values
    CBG_idx_per_group['CBGs in California'] = fire_CBGs.loc[~affected_CBGs, 'index'].values
    if include_distance_controls:
        CBG_idx_per_group['CBGs farther than 50km away from fire perimeter'] = fire_CBGs.loc[(~affected_CBGs)&(fire_CBGs.distance_to_fire_km >= 50), 'index'].values
        CBG_idx_per_group['CBGs within 50km from fire perimeter'] = fire_CBGs.loc[(~affected_CBGs)&(fire_CBGs.distance_to_fire_km < 50), 'index'].values
    
    #Get movers, stayers, populations, and ratios:
    population_t0 = np.array(matrix_dictionary[fireyear+1].sum(axis=1)).flatten()
    population_t1 = np.array(matrix_dictionary[fireyear+1].sum(axis=0)).flatten()
    stayers = matrix_dictionary[fireyear+1].diagonal()
    
    outmovers = population_t0 - stayers
    percentage_that_moved = 100*outmovers/population_t0
    fire_CBGs['movers_pct'] = percentage_that_moved[fire_CBGs['index'].values]
    fire_CBGs['movers_pth'] = 10*percentage_that_moved[fire_CBGs['index'].values]
    
    incoming = population_t1 - stayers
    percentage_that_moved = 100*incoming/population_t1
    fire_CBGs['incoming_pct'] = percentage_that_moved[fire_CBGs['index'].values]
    fire_CBGs['incoming_pth'] = 10*percentage_that_moved[fire_CBGs['index'].values]

    return fire, fire_CBGs, CBG_idx_per_group

def collect_outmigration(flows, population=None, indices=None):
    """
    Collects out-migration and population
    """
    
    #Get the populations:
    if population is None:
        population = np.array(flows.sum(axis=1)).flatten()
        if indices is not None: population = population[indices]

    #Get the movers:
    stayers = flows.diagonal()
    if indices is not None: stayers = stayers[indices]
    movers  = population - stayers

    #Get rates:
    outmigration = movers/population

    return outmigration, population

def include_outmigration(gdf, flows, population=None):

    #Collect rates:
    indices = gdf['index'].values
    outmigration, population = collect_outmigration(flows, population=population, indices=indices)

    #Include rates in gdf:
    gdf['movers_pct'] = outmigration*100
    gdf['movers_pth'] = outmigration*1_000
    gdf['population'] = population

    return gdf