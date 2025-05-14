import os
os.environ['DC_STATEHOOD'] = '1'

from collections import defaultdict
import us

#######################################################################################################################
# VARIABLES FILE
# This file contains most of the variables used in the other modules. Make
#   sure to update the paths below with tha path to the repository in your
#   machine.

_path_to_repo = '/share/pierson/gs665/migration_flows/'

#######################################################################################################################
# PATHS:

#Paths to INFUTOR data directories:
_infutor_dir = '/share/pierson/infutor_data/'
_infutor_dir_processed = '/share/pierson/infutor_data/_processed/'

#Paths to census directories:
_census_spatial_dir = f'{_path_to_repo}d01_data/d01_census-spatial/'
_census_demographics_dir = f'{_path_to_repo}d01_data/d02_census-demographics/'
_census_migration_dir = f'{_path_to_repo}d01_data/d03_census-migration/'
_infutor_migration_dir = f'{_path_to_repo}d01_data/d04_infutor-migration/'
_IRS_migration_dir = f'{_path_to_repo}d01_data/d06_IRS-migration/'

#Paths to wildfire data directory:
_burn_dir = f'{_path_to_repo}d01_data/d08_california-wildfires/'

#Paths to NYCHA data directory:
_NYCHA_dir = f'{_path_to_repo}d01_data/d11_housing/'

#Paths to state txt file:
_states_filepath = f'{_path_to_repo}d01_data/states.txt'

#Paths to output directories:
_outputs_dir = f'{_path_to_repo}d05_outputs/'

#Paths to MIGRATE data:
MIGRATE_dir = f'{_outputs_dir}d02_IPF/Main/'
MIGRATE_str = "M_match-CBGP0_match-state_match-state-nonmovers_NNLS_wACS2010" #in case there are several versions
validations_dir = f'{_outputs_dir}d03_Validations/'

#######################################################################################################################
#API keys for the Census and the HUD crosswalk---please use your keys:
_census_api_key = None
_census_crosswalk_key = None #https://www.huduser.gov/portal/dataset/uspszip-api.html

#######################################################################################################################
#Albert Equal Areas Conic Projection (which allows for geometry computations):
_area_crs = 'EPSG: 5070' #whole US
_CA_crs = 'EPSG: 3310'   #CA Albers
_NY_crs = 'EPSG: 2263'   #NY

#######################################################################################################################
#Number of PIDs in our INFUTOR snapshot:
_N_INFUTOR = 614_949_844

#######################################################################################################################
#Dates:
_monthly_weights=[31/365, 28/365, 31/365, 30/365, 31/365, 30/365, 31/365, 31/365, 30/365, 31/365, 30/365, 31/365]

#######################################################################################################################
#STATE-TO-STATE FILES:
#https://www.census.gov/data/tables/time-series/demo/geographic-mobility/state-to-state-migration.html

_states = ['Alabama', 'Alaska', 'Arizona', 'Arkansas', 'California', 'Colorado', 'Connecticut', 'Delaware', 'District of Columbia', 'Florida', 'Georgia', 'Hawaii', 'Idaho', 'Illinois', 'Indiana', 'Iowa', 'Kansas', 'Kentucky', 'Louisiana', 'Maine', 'Maryland', 'Massachusetts', 'Michigan', 'Minnesota', 'Mississippi', 'Missouri', 'Montana', 'Nebraska', 'Nevada', 'New Hampshire', 'New Jersey', 'New Mexico', 'New York', 'North Carolina', 'North Dakota', 'Ohio', 'Oklahoma', 'Oregon', 'Pennsylvania', 'Rhode Island', 'South Carolina', 'South Dakota', 'Tennessee', 'Texas', 'Utah', 'Vermont', 'Virginia', 'Washington', 'West Virginia', 'Wisconsin', 'Wyoming', 'Puerto Rico']
_abbreviations = [us.states.lookup(state).abbr for state in _states]
_fips = [us.states.lookup(state).fips for state in _states]

_states_ACS_keys = [state if state != "District of Columbia" else "District of Columbia " for state in _states]
_states_ACS_row_header = defaultdict(lambda:5) #Row that contains state indices each year---do assert to verify
_states_ACS_col_header = defaultdict(lambda:0) #First column always contains indices
_same_house_cols = ['Same house 1 year ago', 'Same residence 1 year ago']
_same_state_cols = ['Same state of residence 1 year ago', 'Different residence, same state 1 year ago']

#######################################################################################################################
_ACS_county_to_county_field_positions = {
                                        'STATE_1': (0, 3),
                                        'COUNTY_1': (3, 6),
                                        'STATE_0': (6, 9),
                                        'COUNTY_0': (9, 12),
                                        'STATE_1_NAME': (13, 43),
                                        'COUNTY_1_NAME': (43, 78),
                                        'STATE_0_NAME': (193, 223),
                                        'COUNTY_0_NAME': (223, 258),
                                        'POP_1_EST': (79, 87),
                                        'POP_1_MOE': (88, 96),
                                        'ALL-MOVERS_1_EST': (113, 120),
                                        'ALL-MOVERS_1_MOE': (121, 128),
                                        'DIFFCOUNTY-WITHINSTATE-MOVERS_1_EST': (145, 152),
                                        'DIFFCOUNTY-WITHINSTATE-MOVERS_1_MOE': (153, 160),
                                        'DIFFSTATE-MOVERS_1_EST': (161, 168),
                                        'DIFFSTATE-MOVERS_1_MOE': (169, 176),
                                        'ABROAD_1_EST': (177, 184),
                                        'ABROAD_1_MOE': (185, 192),
                                        'FLOW_EST': (373, 380),
                                        'FLOW_MOE': (381, 388),
                                        'NON-MOVERS_1_EST': (97, 104),
                                        'NON-MOVERS_1_MOE': (105, 112),
                                        'IN-COUNTY-MOVERS_1_EST': (129, 136),
                                        'IN-COUNTY-MOVERS_1_MOE': (137, 144),
                                        'NON-MOVERS_0_EST': (277, 284),
                                        'NON-MOVERS_0_MOE': (285, 292),
                                        'IN-COUNTY-MOVERS_0_EST': (309, 316),
                                        'IN-COUNTY-MOVERS_0_MOE': (317, 324),
                                        }
_ACS_county_cotuny_numeric_cols = ['POP_1', 'ALL-MOVERS_1', 'DIFFCOUNTY-WITHINSTATE-MOVERS_1', 'DIFFSTATE-MOVERS_1', 'ABROAD_1',  'NON-MOVERS_1', 'IN-COUNTY-MOVERS_1', 'FLOW']
_ACS_county_to_county_field_positions_raw = {
    "Current Residence FIPS State Code": (0, 3),
    "Current Residence FIPS County Code": (3, 6),
    "Residence 1 Year Ago FIPS State Code/U.S. Island Areas Code/Foreign Region Code": (6, 9),
    "Residence 1 Year Ago FIPS County Code": (9, 12),
    "Current Residence State Name": (13, 43),
    "Current Residence County Name": (43, 78),
    "Population 1 Year and Over Current County – Estimate": (79, 87),
    "Population 1 Year and Over Current County – MOE": (88, 96),
    "Nonmovers Current County – Estimate": (97, 104),
    "Nonmovers Current County – MOE": (105, 112),
    "Movers within the U.S. for Current County – Estimate": (113, 120),
    "Movers within the U.S. for Current County – MOE": (121, 128),
    "Movers within the Same County for Current County – Estimates": (129, 136),
    "Movers within the Same County for Current County – MOE": (137, 144),
    "Movers from a Different County in the Same State for Current County – Estimate": (145, 152),
    "Movers from a Different County in the Same State for Current County – MOE": (153, 160),
    "Movers from a Different State for Current County – Estimate": (161, 168),
    "Movers from a Different State for Current County – MOE": (169, 176),
    "Movers from Abroad – Estimate": (177, 184),
    "Movers from Abroad – MOE": (185, 192),
    "Residence 1 Year Ago State Name/U.S. Island Areas/Foreign Region": (193, 223),
    "Residence 1 Year Ago County Name": (223, 258),
    "Population That Lived in County 1 Year Ago – Estimate": (259, 267),
    "Population That Lived in County 1 Year Ago – MOE": (268, 276),
    "Nonmovers County of Residence 1 Year Ago – Estimate": (277, 284),
    "Nonmovers County of Residence 1 Year Ago– MOE": (285, 292),
    "Movers within the U.S. for County of Residence 1 Year Ago – Estimate": (293, 300),
    "Movers within the U.S. for County of Residence 1 Year Ago – MOE": (301, 308),
    "Movers within the Same County for County of Residence 1 Year Ago – Estimates": (309, 316),
    "Movers within the Same County for County of Residence 1 Year Ago – MOE": (317, 324),
    "Movers to a Different County in the Same State for County of Residence 1 Year Ago – Estimate": (325, 332),
    "Movers to a Different County in the Same State for County of Residence 1 Year Ago – MOE": (333, 340),
    "Movers to a Different State for County of Residence 1 Year Ago – Estimate": (341, 348),
    "Movers to a Different State for County of Residence 1 Year Ago – MOE": (349, 356),
    "Movers to Puerto Rico – Estimate": (357, 364),
    "Movers to Puerto Rico – MOE": (365, 372),
    "Movers within Flow – Estimate": (373, 380),
    "Movers within Flow – MOE": (381, 388),
}

#######################################################################################################################

#STATE GROUPINGS:

#1. Following census regions and divisions (or microrregions)
# https://www2.census.gov/geo/pdfs/maps-data/maps/reference/us_regdiv.pdf

_states_to_microrregions = {'Alabama':'East South Central',
                           'Alaska':'Pacific',
                           'Arizona':'Mountain',
                           'Arkansas':'West South Central',
                           'California':'Pacific',
                           'Colorado':'Mountain',
                           'Connecticut':'New England',
                           'Delaware':'South Atlantic',
                           'District of Columbia':'South Atlantic',
                           'Florida':'South Atlantic',
                           'Georgia':'South Atlantic',
                           'Hawaii':'Pacific',
                           'Idaho':'Mountain',
                           'Illinois':'East North Central',
                           'Indiana':'East North Central',
                           'Iowa':'West North Central',
                           'Kansas':'West North Central',
                           'Kentucky':'East South Central',
                           'Louisiana':'West South Central',
                           'Maine':'New England',
                           'Maryland':'South Atlantic',
                           'Massachusetts':'New England',
                           'Michigan':'East North Central',
                           'Minnesota':'West North Central',
                           'Mississippi':'East South Central',
                           'Missouri':'West North Central',
                           'Montana':'Mountain',
                           'Nebraska':'West North Central',
                           'Nevada':'Mountain',
                           'New Hampshire':'New England',
                           'New Jersey':'Middle Atlantic',
                           'New Mexico':'Mountain',
                           'New York':'Middle Atlantic',
                           'North Carolina':'South Atlantic',
                           'North Dakota':'West North Central',
                           'Ohio':'East North Central',
                           'Oklahoma':'West South Central',
                           'Oregon':'Pacific',
                           'Pennsylvania':'Middle Atlantic',
                           'Rhode Island':'New England',
                           'South Carolina':'South Atlantic',
                           'South Dakota':'West North Central',
                           'Tennessee':'East South Central',
                           'Texas':'West South Central',
                           'Utah':'Mountain',
                           'Vermont':'New England',
                           'Virginia':'South Atlantic',
                           'Washington':'Pacific',
                           'West Virginia':'South Atlantic',
                           'Wisconsin':'East North Central',
                           'Wyoming':'Mountain',
                           'Puerto Rico':'South Atlantic'}

_microrregions_to_regions = {'New England':'Northeast',
                            'Middle Atlantic':'Northeast',
                            'East North Central':'Midwest',
                            'West North Central':'Midwest',
                            'South Atlantic':'South',
                            'East South Central':'South',
                            'West South Central':'South',
                            'Mountain':'West',
                            'Pacific':'West'}

#######################################################################################################################
#MIGRATION COUNTY TO COUNTY PARAMETERS:

#column names for data from https://www.census.gov/topics/population/migration/guidance/county-to-county-migration-flows.html
_county_to_county_columns = ['current_state_code',
                             'current_county_code',
                             'previous_state_code',
                             'previous_county_code',
                             
                             'current_state',
                             'current_county',
                             
                             'current_county_population',
                             'current_county_population_moe',
                             
                             'current_county_nonmovers',
                             'current_county_nonmovers_moe',
                             
                             'current_county_movers_within_US',
                             'current_county_movers_within_US_moe',
                             
                             'current_county_movers_within_county',
                             'current_county_movers_within_county_moe',
                             
                             'current_county_movers_within_state_diff_county',
                             'current_county_movers_within_state_diff_country_moe',
                             
                             'current_county_movers_diff_state',
                             'current_county_movers_diff_state_moe',
                             
                             'current_county_movers_from_abroad',
                             'current_county_movers_from_abroad_moe',
                            
                             'previous_state',
                             'previous_county',
                             
                             'previous_county_population',
                             'previous_county_population_moe',
                             
                             'previous_county_nonmovers',
                             'previous_county_nonmovers_moe',
                             
                             'previous_county_movers_within_US',
                             'previous_county_movers_within_US_moe',
                             
                             'previous_county_movers_within_county',
                             'previous_county_movers_within_county_moe',
                             
                             'previous_county_movers_within_state_diff_county',
                             'previous_county_movers_within_state_diff_country_moe',
                             
                             'previous_county_movers_diff_state',
                             'previous_county_movers_diff_state_moe',
                             
                             'previous_county_movers_to_PR',
                             'previous_county_movers_to_PR_moe',
                             
                             'county_to_county_flow',
                             'county_to_county_flow_moe']

#Number of rows discarded from each spreadsheet based on t0. different some years for states and PR and a glitch on PA 2012:
_census_migration_discarded_rows = {'States':{2006:8,
                                              2007:8,
                                              2008:9,
                                              2009:10,
                                              2010:10,
                                              2011:10,
                                              2012:10,
                                              2013:10,
                                              2014:10,
                                              2015:11,
                                              2016:11,
                                              2017:11},
                                    'PA':{2006:8,
                                          2007:8,
                                          2008:9,
                                          2009:10,
                                          2010:10,
                                          2011:10,
                                          2012:116,
                                          2013:10,
                                          2014:10,
                                          2015:11,
                                          2016:11,
                                          2017:11},
                                    'PR':{2006:10,
                                          2007:10,
                                          2008:9,
                                          2009:10,
                                          2010:10,
                                          2011:10,
                                          2012:10,
                                          2013:10,
                                          2014:10,
                                          2015:11,
                                          2016:11,
                                          2017:11}}
