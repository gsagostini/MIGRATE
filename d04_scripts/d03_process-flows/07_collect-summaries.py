# Create a table summarizing INFUTOR
#####################################################################################################################################
import sys
sys.path.append('../../d03_src/')
import vars

import numpy as np
import pandas as pd
#####################################################################################################################################

#Get the dates:
df = pd.read_csv(f'{vars._infutor_dir_processed}CRD4/individuals/full.csv',
                 usecols=['IDATE', 'ODATE']+[f'EFFDATE{k}' for k in range(1, 11)], dtype='str')

#Select only years:
year_df = pd.DataFrame(dtype=float)
for col in df.columns: year_df.loc[:, col] = df[col].str[:4].astype(float)

#Find the maximum and minimum year:
year_df['IDATE'] = year_df.min(axis=1)
year_df['ODATE'] = year_df.max(axis=1)
year_df[['IDATE']].to_csv('IDATE.csv', index=False)


#####################################################################################################################################

#Construct a table:
all_table_series = pd.Series()

#First check overall the number of individuals active 2010-2019:
active = (year_df['IDATE'] <= 2019) & (year_df['ODATE'] >= 2010)
all_table_series['Active individuals in 2010-2019'] = active.sum()

#Get the number of addresses per person:
addresses_per_person = 10-year_df.loc[active].iloc[:,2:].isna().sum(axis=1)
all_table_series['Total addresses for active individuals'] = addresses_per_person.sum()
all_table_series['Mean addresses for active individuals'] = addresses_per_person.mean()
all_table_series['Median addresses for active individuals'] = addresses_per_person.median()

#Construct a table for years:
years = np.arange(2010, 2020)
table_df = pd.DataFrame(index=years)

#We want to check the number of individuals active in 2010-2020:
active_after_2010 = year_df['IDATE'].values[:, np.newaxis] <= years
active_bfore_2020 = years <= year_df['ODATE'].values[:, np.newaxis]
activity_per_individual = active_after_2010&active_bfore_2020
activity_per_year = activity_per_individual.sum(axis=0)
table_df['Active Individuals'] = pd.Series(activity_per_year, index=years)

#We also want to check the number of address pings per year
new_address_years = year_df.iloc[:,2:].values.flatten()
table_df['Address Pings'] = pd.Series(new_address_years).dropna().value_counts().loc[years]
all_table_series['Address pings in 2010-2019'] = table_df['Address Pings'].sum()

#Save
all_table_series.to_csv(f'{vars._outputs_dir}period_table.csv')
table_df.index.name='Year'
table_df.to_csv(f'{vars._outputs_dir}yearly_table.csv')