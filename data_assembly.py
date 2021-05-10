# Policy Pandas: Radhika Ramakrishnan, Salma Katri, Dalya Elmalt
# The following scrapes, cleans, and merges 8 variables of data
# Variables: unemployment, labor force participation, population, food stamp
# dependence, poverty, income, voting outcomes, taxes paid, insurance coverage
# References: SQL joins, strip leading zeros

import bs4
import re
import pandas as pd
import requests
import numpy as np
import math

def string_to_int(column):
    '''
    Removes commas from values in a dataframe's column and transforms the
    values in that column into integers

    Inputs: column - the actual column to be modified

    Returns: modified_column - new and improved column
    '''

    modified_column = column.str.replace(',', '')
    modified_column = modified_column.astype(int)

    return modified_column


# We chose to account for Alaska as a whole unit because its counties 
# change quite a bit from year to year, preventing consistent analysis
def Alaska_data(dataframe, num_cols):
    '''
    Sums Alaska data and modifies dataframe to include totals rather than
    values for individual counties

    Inputs: dataframe - the dataframe to be cleaned, num_cols - a list of the 
    column names of the dataframe that have numerical data

    Returns: new_values
    '''

    alaska = dataframe[dataframe['statefips'] == '2']

    new_values = []
    for nc in num_cols:
        new_value = sum(alaska[nc])
        new_values.append(new_value)
    
    return new_values


# Unemployment, Labor Force Participation Rates

def lfpr_reader(year):
    '''
    Reads in labor force and unemployment data

    Inputs: year - string with the last two digits of the year whose
    dataframe is being processed

    Returns: df - cleaner version of the data for that year in a dataframe
    '''
    df = pd.read_csv('LFPR{}.csv'.format(year),
        usecols=[1, 2, 3, 4, 6, 8], names=['statefips', 'countyfips',
        'county_name', 'year', 'labor_force', 'unemployment'])
    
    if year == '08' or year == '09':
        df = df.iloc[6:3223]
    else:
        df = df.iloc[6:3225]

    df['labor_force'] = string_to_int(df['labor_force'])
    df['unemployment'] = string_to_int(df['unemployment'])

    df['statefips'] = df['statefips'].str.lstrip('0')
    
    alaska = Alaska_data(df, ['labor_force', 'unemployment'])
    df = df[df['statefips'] != '2']
    ak_dict = {'statefips': '2', 'countyfips': '0', 'county_name': 'Alaska',
        'year': '20{}'.format(year), 'labor_force': alaska[0], 
        'unemployment': alaska[1]}
    alaska_df = pd.DataFrame(ak_dict, index=[0])
    df = df.append(alaska_df)
    
    return df


def make_lfpr():
    '''
    Constructs the labor force participation and unemployment rate dataframe

    Returns: dataframe with two variables
    '''
    lfpr08 = lfpr_reader('08')
    lfpr09 = lfpr_reader('09')
    lfpr10 = lfpr_reader('10')
    lfpr11 = lfpr_reader('11')
    lfpr12 = lfpr_reader('12')
    lfpr13 = lfpr_reader('13')
    lfpr14 = lfpr_reader('14')
    lfpr15 = lfpr_reader('15')

    dfs = [lfpr08, lfpr09, lfpr10, lfpr11, lfpr12, lfpr13, lfpr14, lfpr15]
    lfpr_df = pd.DataFrame(columns=['statefips', 'countyfips',
        'county_name', 'year', 'labor_force'])
    for df in dfs:
        lfpr_df = lfpr_df.append(df)

    lfpr_df['countyfips'] = lfpr_df['countyfips'].str.lstrip('0')
    lfpr_df['fips'] = lfpr_df['statefips'] + ',' + lfpr_df['countyfips']
    lfpr_df['fips'] = lfpr_df['fips'].replace(to_replace='2,', value='2,0')
    lfpr_df.ix[lfpr_df.fips == '2,0', 'countyfips'] = '0'
    
    # We only want data for the 50 states
    lfpr_df = lfpr_df[lfpr_df['statefips'] != '72'] 

    return lfpr_df


# Population

def make_population():
    '''
    Reads in and cleans the population data

    Returns: final_pop_df a dataframe containing population data
    '''
    pop = pd.read_csv('population.csv', names=['fips', 'year', 'population', 
        'fips_numerical'])
    pop = pop[9:]

    new_df = pop['fips'].str.extract(r'([0-9]+)([0-9][0-9][0-9])', expand=True)
    pop = pd.concat([pop, new_df], axis=1)
    pop = pop.rename(columns={0: 'statefips', 1: 'countyfips'})
    pop = pop.ix[pop['statefips'].iloc[:].notnull()]

    pop['statefips'] = pop['statefips'].str.lstrip('0')
    pop['countyfips'] = pop['countyfips'].str.lstrip('0')
    pop['fips'] = pop['statefips'] + ',' + pop['countyfips']

    pop = pop[pop['population'].notnull()]

    pop['population'] = pop['population'].astype(int)

    final_pop_df = pd.DataFrame(columns=['fips', 'year', 'population', 
        'fips_numerical', 'statefips', 'countyfips'])
    
    years = ['2008', '2009', '2010', '2011', '2012', '2013', '2014', '2015']
    for year in years:
        new_pop = pop[pop['year'] == year]
        alaska = Alaska_data(new_pop, ['population'])
        new_pop = new_pop[new_pop['statefips'] != '2']
        ak_dict = {'fips': '2,0', 'year': year, 'population': alaska[0],
        'fips_numerical': '2000', 'statefips': '2', 'countyfips': '0'}
        ak_df = pd.DataFrame(ak_dict, index=[0])
        new_pop = new_pop.append(ak_df)
        final_pop_df = final_pop_df.append(new_pop) 

    final_pop_df['fips'] = (final_pop_df['fips'].replace(to_replace='46,113', 
        value='46,102'))
    final_pop_df.ix[final_pop_df.fips == '46,102', 'countyfips'] = '102'
    final_pop_df.ix[final_pop_df.fips == '46,102', 'fips_numerical'] = '46102'

    # an independent city, our model is only interested in 
    # consistently measured counties
    final_pop_df = final_pop_df[final_pop_df['fips'] != '51,515'] 

    return final_pop_df


# Food Stamp Data

def food_stamps():
    '''
    Constructs and cleans food stamp dependence rate dataframe

    Returns: FS - dataframe of food stamp data
    '''
    fstamp = pd.read_csv("cntysnap.csv")
    fstamp.columns = ['statefips', 'countyfips', 'name', '2014', '2013', 
    '2012', '2011', '2010', '2009', '2008']
    fstamp = fstamp[fstamp['countyfips'] != 0]

    year = []
    foodstamps = []

    for i in ['2008', '2009', '2010', '2011', '2012', '2013', '2014']:
        year += [i]*len(fstamp['name'])
        foodstamps += fstamp[i].tolist()

    statefips = fstamp['statefips'].tolist()*7
    countyfips = fstamp['countyfips'].tolist()*7

    county = []
    state = []

    for i in fstamp['name'].tolist():
        state.append(i[-2:])
        county.append(i[:-4])

    county  = county*7
    state = state*7
    year = pd.DataFrame(year, columns = ['year']) 
    statefips = pd.DataFrame(statefips, columns= ['statefips'])    
    countyfips = pd.DataFrame(countyfips, columns = ['countyfips'])   
    state = pd.DataFrame(state, columns= ['state'])
    county = pd.DataFrame(county, columns= ['county'])   
    foodstamps = pd.DataFrame(foodstamps, columns= ['foodstamps'])         

    frames = [year, statefips, countyfips, state, county, foodstamps]
    FS = pd.concat(frames, axis=1)
    FS['statefips'] = FS.statefips.astype(str)
    FS['countyfips'] = FS.countyfips.astype(str)
    FS['fips'] = FS['statefips'] + ',' + FS['countyfips']

    
    alaska_dfs = pd.DataFrame(columns=['county', 'countyfips', 'fips',
        'foodstamps', 'state', 'statefips', 'year'])
    for year in ['2008', '2009', '2010', '2011', '2012', '2013', '2014']:
        ak_df = FS[FS['year'] == year]
        ak_df = ak_df[ak_df['foodstamps'].notnull()]
        alaska = Alaska_data(ak_df, ['foodstamps'])
        alaska_dict = {'county': 'Alaska', 'countyfips': '0', 'fips': '2,0',
        'foodstamps': alaska[0], 'state': 'AK', 'statefips': '2', 'year': year
        }
        alaska_df = pd.DataFrame(alaska_dict, index=[0])
        alaska_dfs = alaska_dfs.append(alaska_df)

    FS = FS[FS['statefips'] != '2']
    FS = FS.append(alaska_dfs)
    FS = FS[FS['foodstamps'].notnull()]

    FS['fips'] = FS['fips'].replace(to_replace='46,113', value='46,102')
    FS.ix[FS.fips == '46,102', 'countyfips'] = '102'

    FS = FS[FS['fips'] != '51,515']

    return FS


# Poverty & Income Data

def poverty_income():
    '''
    Creates DataFrame for poverty and income data

    Returns - pov_inc, a dataframe of poverty rate and income data
    '''
    files = ["est08ALL.csv",
        "est09ALL.csv",
        "est10ALL.csv",
        "est11ALL.csv",
        "est12ALL.csv",
        "est13ALL.csv",
        "est14ALL.csv",
        "est15ALL.csv"]
    # Columns' names for 2008 to 2011
    col_list = ['State FIPS', 'County FIPS', 'Postal', 'Name',
            'Poverty Percent All Ages', 'Median Household Income']
    # Columns' names for 2012 to 2015
    alt_col_list = ['State FIPS Code', 'County FIPS Code', 'Postal Code', 'Name',
                'Poverty Percent, All Ages', 'Median Household Income']
     
    # Poverty Percent, All Ages, the percent of total population in poverty
    pov_inc = []
    for file_1 in files:
        if file_1[-9:-7] in {'08', '09', '10', '11'}:
            pov = pd.read_csv(file_1)[col_list]
        else:
            pov = pd.read_csv(file_1)[alt_col_list]
        pov.columns = ['statefips', 'countyfips', 'state', 'county', 'poverty',
                'hhinc']
        #Changing code from string to integer
        pov.loc[:, 'year'] = pd.Series(['20'+file_1[-9:-7]]*len(pov['state']))
        pov = pov[pov['countyfips'] != 0]
        pov = pov[pov.county.notnull()]

        pov['statefips'] = pov['statefips'].astype(str)
        pov['statefips'] = pov['statefips'].str.lstrip('0')
        pov['countyfips'] = pov['countyfips'].astype(str)
        pov['countyfips'] = pov['countyfips'].str.lstrip('0')
        pov['countyfips'] = pov['countyfips'].str.rstrip('.0')
        pov['fips'] = pov['statefips'] + ',' +  pov['countyfips']

        pov['poverty'] = pov['poverty'].astype(str)
        pov = pov[pov['hhinc'].notnull()]
        pov['hhinc'] = pov['hhinc'].astype(str)
        pov.loc[pov.poverty == '.', 'poverty'] = 0
        pov.loc[pov.hhinc == '.', 'hhinc'] = '0'
        pov['hhinc'] = string_to_int(pov['hhinc'])
        pov['poverty'] = pov['poverty'].astype(float)

        alaska = Alaska_data(pov, ['poverty', 'hhinc'])
        pov = pov[pov['statefips'] != '2']
        alaska_dict = {'year': pov['year'].iloc[1], 'fips' : '2,0', 'statefips': 
        '2', 'countyfips' : '0', 'state': 'AK', 'county': 'Alaska',
        'poverty': alaska[0]/8, 'hhinc': alaska[1]}
        alaska_df = pd.DataFrame(alaska_dict, index=[0])
        pov = pov.append(alaska_df)

        cond = ((pov['county'] != 'St. Louis city') & 
            (pov['county'] != 'Alexandria city') & 
            (pov['county'] != 'Buena Vista city') & 
            (pov['county'] != 'Colonial Heights city') & 
            (pov['county'] != 'Danville city') & 
            (pov['county'] != 'Falls Church city') & 
            (pov['county'] != 'Fredericksburg city') & 
            (pov['county'] != 'Hampton city') &  
            (pov['county'] != 'Hopewell city') & 
            (pov['county'] != 'Martinsville city') & 
            (pov['county'] != 'Newport News city') & 
            (pov['county'] != 'Norfolk city') & 
            (pov['county'] != 'Petersburg city') & 
            (pov['county'] != 'Radford city') & 
            (pov['county'] != 'Roanoke city') & 
            (pov['county'] != 'Staunton city') & 
            (pov['county'] != 'Virginia Beach city') & 
            (pov['county'] != 'Williamsburg city'))
        pov = pov[cond]

        if len(pov_inc) == 0:
            pov_inc = pd.DataFrame(pov)
        else:
            pov_inc = pov_inc.append(pd.DataFrame(pov))

    pov_inc = (pov_inc[['year', 'fips', 'statefips', 'countyfips', 'state', 
    'county', 'poverty', 'hhinc']])
    pov_inc.loc[pov_inc.poverty == 0, 'poverty'] = np.nan
    pov_inc.loc[pov_inc.hhinc == 0, 'hhinc'] = np.nan

    pov_inc['fips'] = (pov_inc['fips'].replace(to_replace='46,113', 
        value='46,102'))
    pov_inc.ix[pov_inc.fips == '46,102', 'countyfips'] = '102'
    pov_inc['county'] = (pov_inc['county'].replace(to_replace='Shannon County', 
        value='Oglala Lakota County'))

    return pov_inc


# Voting data:

def voting_outcomes_data():
    '''
    Constructs and cleans voting outcomes dataframe

    Returns: voting_outcomes_data - a dataframe of voting data
    '''
    vot = pd.read_csv("US_County_Level_Presidential_Results_08-16.csv")
    vot = vot.sort_values(by='fips_code')

    # 2008
    voting_outcomes = vot[['fips_code', 'county', 'total_2008', 'dem_2008',
     'gop_2008', 'oth_2008']]
    voting_outcomes['year'] = '2008'
    voting_outcomes.columns = ['fips_code', 'county', 'total_votes', 'dem',
     'gop', 'other', 'year']

    # 2012
    vot_12 = vot[['fips_code', 'county']]
    vot_12['total_2012'] = vot[['total_2012']]
    vot_12['dem_2012'] = vot[['dem_2012']]
    vot_12['gop_2012'] = vot[['gop_2012']]
    vot_12['other_2012'] = vot[['oth_2012']]
    vot_12['year'] = '2012'
    vot_12.columns = ['fips_code', 'county', 'total_votes', 'dem', 'gop',
      'other', 'year']

    # 2016
    vot_16 = vot[['fips_code', 'county']]
    vot_16['total_2016'] = vot[['total_2016']]
    vot_16['dem_2016'] = vot[['dem_2016']]
    vot_16['gop_2016'] = vot[['gop_2016']]
    vot_16['other_2016'] = vot[['oth_2016']]
    vot_16['year'] = '2016'
    vot_16.columns = ['fips_code', 'county', 'total_votes', 'dem', 'gop',
      'other', 'year']

    # Alaska
    ak_dict = {'fips_code': ['2000']*3, 'county': ['Alaska']*3, 'total_votes': 
    [322100, 247483, 299883], 'dem': [122485, 102138, 116454], 
    'gop': [192631, 136848, 163387], 'other': [6984, 8497, 20042],
    'year': ['2008', '2012', '2016']}
    ak_df = pd.DataFrame(ak_dict, index=[0, 1, 2])
    frames = [voting_outcomes, vot_12, vot_16]
    voting_outcomes_data = pd.concat(frames)
    voting_outcomes_data = voting_outcomes_data.append(ak_df)

    voting_outcomes_data = (voting_outcomes_data.rename(columns={'fips_code': 
        'fips_numerical'}))
    voting_outcomes_data['fips_numerical'] = (
        voting_outcomes_data['fips_numerical'].astype(str))
    
    voting_outcomes_data['fips_numerical'] = (
        voting_outcomes_data['fips_numerical'].replace(to_replace='46,113', 
            value='46,102'))
    voting_outcomes_data['county'] = (
        voting_outcomes_data['county'].replace(to_replace='Shannon County', 
            value='Oglala Lakota County'))

    return voting_outcomes_data


# Tax data
# The most recent data available from the IRS on the county level was from 
# 2010, which is why our data set does not go farther back for this variable

def tax_data():
    '''
    Constructs and cleans dataframe for tax data

    Returns: tax - a dataframe of tax data
    '''
    files = ["10incyallagi.csv",
            "11incyallagi.csv",
            "12incyallagi.csv",
            "13incyallagi.csv",
            "14incyallagi.csv"]

    col_list = ['STATEFIPS', 'COUNTYFIPS', 'STATE', 'COUNTYNAME',
                'agi_stub', 'N18300', 'A18300']
    tax = []
    for file in files:
        data = pd.read_csv(file, encoding="ISO-8859-1")
        data = data[col_list]
        data.columns = ['statefips', 'countyfips', 'state', 'county',
                        'income_bracket', 'number_returns', 'tax_collected']
        data['statefips'] = data['statefips'].astype(str)
        data['countyfips'] = data['countyfips'].astype(str)
        data['fips'] = data['statefips'] + ',' +  data['countyfips']
        data.loc[:, 'year'] = pd.Series(['20'+file[-16:-14]]*len(data['state']))
        
        df = (data.groupby(by=['state', 
            'county', 'year'], )['tax_collected'].sum())
        df = df.reset_index()
        df.columns = ['state', 'county', 'year', 'total_tax']
        
        data = pd.merge(data, df, on=['state', 'county', 'year'])

        # keep only first row for total_tax
        data = (data.groupby(
            by=['countyfips', 'statefips'], ).head(1).reset_index(drop=True))
        data = data[data['countyfips'] != '0']

        alaska = Alaska_data(data, ['total_tax'])
        data = data[data['statefips'] != '2']
        alaska_dict = {'year': data['year'].iloc[1], 'fips': '2,0', 
        'statefips': '2', 'county': '0', 'state': 'AK', 'county': 'Alaska',
        'total_tax': alaska[0], 'countyfips': '0'}
        alaska_df = pd.DataFrame(alaska_dict, index=[0])
        data = data.append(alaska_df)

        if len(tax) == 0:
            tax = pd.DataFrame(data)
        else:
            tax = tax.append(pd.DataFrame(data))

        tax = tax[['year', 'fips', 'statefips', 'countyfips',
                   'state','county', 'total_tax']]

    tax['year'] = tax['year'].astype(str)

    tax['fips'] = tax['fips'].replace(to_replace='46,113', value='46,102')
    tax['county'] = (tax['county'].replace(to_replace='Shannon County', 
        value='Oglala Lakota County'))
    tax.ix[tax.fips == '46,102', 'countyfips'] = '102'

    return tax

def insurance_data():
    '''
    Constructs and cleans dataframe for insurance data

    Returns: premiums, a dataframe of insurance data
    '''
    files = ["sahie_2008.csv",
             "sahie_2009.csv",
             "sahie_2010.csv",
             "sahie_2011.csv",
             "sahie_2012.csv",
             "sahie_2013.csv",
             "sahie_2014.csv"]

    names = ['year', 'statefips', 'countyfips', 'agecat', 'racecat', 'sexcat',
            'iprcat', 'insured']

    premiums = []
    
    for file_1 in files:
        df = pd.read_csv(file_1, header=80, usecols=[0, 2, 3, 5, 6, 7, 8, 13], 
            names=names, dtype={13:str})
        cond = ((df.iprcat == 0) & (df.sexcat == 0) &
            (df.racecat == 0) & (df.agecat == 0) &
            (df.countyfips != 0))
        df = df[cond]
        df['statefips'] = df['statefips'].astype(str)
        df['countyfips'] = df['countyfips'].astype(str)
        df['fips'] = df['statefips'] + ',' + df['countyfips']
        df = df[df['countyfips'] != '0']
        df = df[['year', 'fips', 'statefips', 'insured']]
        df[df.isnull().any(axis=1)]
        
        df = df[df['fips'] != '15,5']
        df['insured'] = df.insured.map(int)
        alaska = Alaska_data(df, ['insured'])      
        df = df[df['statefips'] != '2']
        alaska_dict = {'year': df['year'].iloc[1], 'fips': '2,0',
        'statefips': '2', 'insured': alaska[0]}
        alaska_df = pd.DataFrame(alaska_dict, index=[0])
        df = df.append(alaska_df)

        if file_1 == "sahie_2008.csv":
            premiums = pd.DataFrame(df)
        else:
            premiums = premiums.append(pd.DataFrame(df))

    premiums['year'] = premiums['year'].astype(str)

    premiums['fips'] = (premiums['fips'].replace(to_replace='46,113', 
        value='46,102'))

    return premiums


# Merge, Finish Cleaning
labor_force_data = make_lfpr()
population_data = make_population()
food_stamp_data = food_stamps()
poverty_income_data = poverty_income()
votes_data = voting_outcomes_data()
taxes_data = tax_data()
insurance_coverage_data = insurance_data()

# Merge Labor Force Participation Rate/Unemployment & Population
complete_data = labor_force_data.merge(population_data, on=['year', 'fips', 
    'statefips', 'countyfips'], how='inner')

# Add Food Stamp data
complete_data = complete_data.merge(food_stamp_data, 
    on=['year', 'fips', 'statefips', 'countyfips'], how='left')

# Add Tax data
# The left join here eliminates Bedford city, an independent city
# We choose not to use it to focus our analysis on counties or 
# cities with complete data/that are not ambiguously part of 
# counties
complete_data = complete_data.merge(taxes_data, on=['year', 'fips', 
    'statefips', 'countyfips'], how='left')
complete_data = complete_data[['countyfips', 'labor_force', 'statefips',
'unemployment', 'year', 'fips', 'population', 'county_name', 'foodstamps', 
'state_x', 'county_x', 'total_tax', 'fips_numerical']]
complete_data = complete_data.rename(columns={'state_x': 'state', 
    'county_x': 'county'})

# Add Insurance data
complete_data = complete_data.merge(insurance_coverage_data, on=['year', 'fips',
    'statefips'], how='left')

# Add Poverty/Income data
complete_data = complete_data.merge(poverty_income_data, on=['year', 'fips',
    'statefips', 'countyfips'], how='inner')
complete_data = complete_data[['countyfips', 'labor_force', 'statefips',
'unemployment', 'year', 'fips', 'population', 'county_name', 'foodstamps',
'total_tax', 'insured', 'poverty', 'hhinc', 'fips_numerical']]

complete_data = complete_data[complete_data['countyfips'].notnull()]

# Modify to obtain rates and save
complete_data['unemployment'] = complete_data['unemployment'].astype(float)
complete_data['unemployment'] = (
    (complete_data['unemployment']/complete_data['labor_force']) * 100)
varlist = ['labor_force', 'insured', 'foodstamps']
for var in varlist:
    complete_data[var] = complete_data[var].astype(float)
    complete_data[var] = (complete_data[var]/complete_data['population']) * 100

ids = ['statefips', 'countyfips', 'year']
for id_name in ids:
    complete_data[id_name] = complete_data[id_name].astype(int)
complete_data.sort_values(by=['year', 'statefips', 'countyfips'])

complete_data.to_csv('complete_data.csv')
votes_data.to_csv('voting_outcomes.csv')


