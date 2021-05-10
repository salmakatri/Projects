# Policy Pandas: Radhika Ramakrishnan, Salma Katri, Dalya Elmalt
# Develop model to predict election outcomes (republican or democrat)
# Train model on 2008 data
# Test on 2016 data

import pandas as pd 
import numpy as np 
import itertools
import math
from sklearn import linear_model


def regress(X_vars, Y_vars):
    '''
    Estimates simple or multiple linear regression model using the ordinary
    least squares method

    Inputs: X_vars - a pandas dataframe containing the X variables 
    (regressors), Y_vars - a pandas dataframe containing election 
    results data

    Returns: results - tuple of parameters of regression  model, the
    model itself
    '''
    regr = linear_model.LinearRegression()

    Y_v = Y_vars.values
    X_v = X_vars.values

    regr.fit(X_v, Y_v)
    coeff = regr.coef_
    b_0 = regr.intercept_[0]

    list_of_coefficients = [b_0]
    for i in range(len(coeff)):
        list_of_coefficients += [coeff[i]]

    return list_of_coefficients, regr


def r_squared(Y_vars, predicted_values, k):
    '''
    Calculates the r-squared (simple linear regression) or adjusted r-squared
    (multiple linear regression) value for a model

    Inputs: X_vars - a pandas dataframe containing the X variables 
    (regressors), Y_vars - a pandas dataframe containing election 
    results data

    Returns: r2 - the relevant form of r-squared 
    '''

    SST, SSR = sum_squares(Y_vars, predicted_values)
    n = len(predicted_values)

    if k == 1:
        r_squared = 1 - (SSR/SST)
        return r_squared
    elif k > 1:
        adj_r_squared = 1 - ((SSR/(n - k - 1))/(SST/(n - 1)))
        return adj_r_squared


def sum_squares(Y_vars, predicted):
    '''
    Calculates the total sum of squares and residual sum of squares for a 
    linear regression model 

    Inputs: predicted_values - a dataframe of values predicted by the linear
    regression model, Y_vars - the actual values used for the regression

    Returns: sums_squared - tuple containing the total sum of squares and the 
    residual sum of squares
    '''

    Y_vars1 = Y_vars.reset_index(drop=True)
    predicted1 = predicted.reset_index(drop=True)
    y_new = pd.concat([Y_vars1, predicted1], axis=1)
    
    y_bar = Y_vars.mean()
    y_new['y_diff_squared'] = (Y_vars1 - y_bar) ** 2
    SST = sum(y_new['y_diff_squared'])
    
    y_array = np.subtract(Y_vars1.values, y_new['predictions'].reshape(-1,1))
    y_new['residuals'] = y_array
    y_new['u_diff_squared'] = y_new['residuals'] ** 2
    SSR = sum(y_new['u_diff_squared'])

    sums_squared = (SST, SSR)

    return sums_squared


def apply_model(X_vars, regr):
    '''
    Given the parameters of a linear regression model, applies the model and
    returns the predicted values in a dataframe

    Inputs: X_vars - dataframe of x-values, Y_vars - dataframe of y-values,
    model_params - list of parameter estimates for the linear regression model

    Returns: predicted_df - dataframe of predicted values
    '''
    X_v = X_vars.values
    predicted = regr.predict(X_v)
    predicted_df = pd.DataFrame(predicted, columns=['predictions'])

    return predicted_df


def model_selector(X_vars, Y_vars):
    '''
    Selects the best model based on adjusted r-squared or r-squared value

    Inputs: X_vars - dataframe of regressors, Y_vars - dataframe of the 
    single regressand

    Returns: params - list of the best regressors to include in the regression
    '''
    potential_vars = ['labor_force', 'unemployment', 'poverty', 'hhinc',
    'foodstamps', 'insured', 'total_tax']
    r_squared1 = -1
    variables = []
    regr1 = None

    for i in range(len(potential_vars)):
        test_models = combinations(i+1, potential_vars)
        for model in test_models:
            model_list = [item for item in model]
            list1, regr = regress(X_vars[model_list], Y_vars)
            predicts = apply_model(X_vars[model_list], regr)
            r_sq = r_squared(Y_vars, predicts, i+1)
            if r_sq > r_squared1:
                r_squared1 = r_sq
                variables = model_list
                regr1 = regr
    
    return variables, regr1, r_squared1

def combinations(n, uniq_vals):
    '''
    Generates all possible combinations of length n of a given set of values

    Inputs: n - length of combinations, uniq_vals - list of unique values from 
    which combinations are being made

    Returns: combs - a list of lists, where each internal list contains a 
    combination, and the overall list contains all of the combinations of
    interest
    '''
    combs = itertools.combinations(uniq_vals, n)
    combs_list = list(combs)

    return combs_list


def get_outcome(training, testing):
    '''
    Creates a dataframe of predicted election results for counties
    This function also prints out 2 tuples - each containing the list 
    of variables used in the model and the r-squared of the best model
    for each party

    Inputs: training - dataframe of training data, testing - dataframe
    of testing data

    Returns: predictions - dataframe of predicted outcomes
    '''
    X_train = training[['labor_force', 'unemployment', 'poverty', 'hhinc',
    'foodstamps', 'insured', 'total_tax']]
    Y_train_dem = training[['dem']]
    Y_train_gop = training[['gop']]
    X_test = testing[['labor_force', 'unemployment', 'poverty', 'hhinc',
    'foodstamps', 'insured', 'total_tax']]
    Y_test_dem = testing[['dem']]
    Y_test_gop = testing[['gop']]
    fips = testing[['fips']]
    fips = fips.reset_index(drop=True)

    # dem (two different estimations because different variables may be
    # important when estimating vote percentages for each party)
    opt_model_dem, regr_dem, rsq_dem = model_selector(X_train, Y_train_dem)
    dem_predictions = apply_model(X_test[opt_model_dem], regr_dem)
    dem_predictions = dem_predictions.rename(
        columns={'predictions': 'democratic'})
    dem_predictions = dem_predictions.reset_index(drop=True)

    # gop
    opt_model_gop, regr_gop, rsq_gop = model_selector(X_train, Y_train_gop)
    gop_predictions = apply_model(X_test[opt_model_gop], regr_gop)
    gop_predictions = gop_predictions.rename(
        columns={'predictions': 'republican'})
    gop_predictions = gop_predictions.reset_index(drop=True)

    predictions = pd.concat([fips, dem_predictions, gop_predictions], axis=1)

    print((opt_model_gop, rsq_gop))
    print((opt_model_dem, rsq_dem))
    
    return predictions

def dem_indicator(predicted_outcomes):
    '''
    Creates an indicator variable equal to 0 if a county voted republican,
    1 if the county voted democrat

    Inputs: predicted_outcomes - a dataframe of predicted outcomes

    Returns: predicted_outcomes - the same dataframe, now including the
    indicator for voting democrat
    '''
    predicted_outcomes['indicator'] = 0
    predicted_outcomes.loc[(predicted_outcomes.democratic > 
    predicted_outcomes.republican, 'indicator')] = 1

    return predicted_outcomes


def evaluate_model(testing, predicted_outcomes):
    '''
    Determines probability of succesfully predicting the election outcome in 
    any given county

    Inputs: actual - dataframe of real election results, predicted_outcomes - 
    dataframe of predicted election results
    
    Returns: predicted_outcomes - the dataframe, now containing a column with 
    the probability of being correct the model has
    '''
    actual_outcomes = testing[['fips', 'gop', 'dem', 'dem_ind_actual']]
    
    predicted_outcomes = pd.concat([actual_outcomes, predicted_outcomes], 
        axis=1)
    predicted_outcomes['right'] = 0
    predicted_outcomes.loc[(
        predicted_outcomes['dem_ind_actual'] == predicted_outcomes['indicator'], 'right')] = 1
    
    total_correct = sum(predicted_outcomes['right'])
    total = len(predicted_outcomes)
    prob = total_correct/total_correct

    predicted_outcomes['probability_correct'] = prob
    predicted_outcomes = (
        predicted_outcomes[['fips', 'indicator', 'probability_correct']])

    return predicted_outcomes

# Pre-process data for analysis
# Use logs of votes, some variables, to examine percentages and percent change

X = pd.read_csv('complete_data.csv')
Y = pd.read_csv('voting_outcomes.csv')

Y['dem_ind_actual'] = 0
Y.loc[Y.dem > Y.gop, 'dem_ind_actual'] = 1

Y_train = Y[Y['year'] == 2008]
Y_train['fips_numerical'] = Y_train['fips_numerical'].astype(float)
Y_train = Y_train[['fips_numerical', 'dem', 'gop', 'dem_ind_actual']]

taxes = X[X['year'] == 2010]
taxes = taxes[['total_tax', 'fips']]
X_train = X[X['year'] == 2008]
X_train = X_train.drop('total_tax', axis=1)
X_train = X_train.merge(taxes, on=['fips'], how='inner')
X_train = X_train[X_train['fips_numerical'].notnull()]
X_train = X_train.merge(Y_train, on=['fips_numerical'], how='inner')
X_train = X_train.dropna()
train = X_train.drop(['countyfips', 'statefips', 'year', 'county_name', 
    'fips_numerical', 'Unnamed: 0'], axis=1)

Y_test = Y[Y['year'] == 2016]
Y_test['fips_numerical'] = Y_test['fips_numerical'].astype(float)
Y_test = Y_test[['fips_numerical', 'dem', 'gop', 'dem_ind_actual']]

other_test = X[['fips', 'year', 'foodstamps', 'insured', 'total_tax']]
other_test = other_test[other_test['year'] == 2014]

X_test = X[X['year'] == 2015]
X_test = X_test.drop(['foodstamps', 'insured', 'total_tax'], axis=1)
X_test = X_test.merge(other_test, on=['fips'], how='inner')
X_test = X_test[X_test['fips_numerical'].notnull()]
X_test = X_test.merge(Y_test, on=['fips_numerical'], how='inner')
X_test = X_test.drop(['countyfips', 'statefips', 'year_x', 'year_y', 
    'county_name', 'fips_numerical', 'Unnamed: 0'], axis=1)
X_test = X_test.dropna()
test = X_test.iloc[0:3076]

test = test.reset_index(drop=True)
train = train.reset_index(drop=True)

varlist = ['labor_force', 'poverty', 'insured']
for var in varlist:
    train[var] = train[var].multiply(train['population'])/100
    test[var] = test[var].multiply(test['population'])/100
    train[var] = train[var].map(math.log)
    test[var] = test[var].map(math.log)

train['foodstamps'] = (train['foodstamps'].multiply(train['population']))/100
train['unemployment'] = ((train['unemployment'].multiply(
    train['labor_force']))/100).map(math.exp)
train['unemployment'] = train['unemployment'].map(math.log)
train['dem'] = train['dem'].map(math.log)
train['gop'] = train['gop'].map(math.log)

test['foodstamps'] = (test['foodstamps'].multiply(train['population']))/100
test['unemployment'] = ((test['unemployment'].multiply(
    test['labor_force']))/100).map(math.exp)
test['unemployment'] = test['unemployment'].map(math.log)
test['dem'] = test['dem'].map(math.log)
test['gop'] = test['gop'].map(math.log)

predictions = get_outcome(train, test)
predictions = dem_indicator(predictions)
# This returns a probability of 1, which means our model is pretty good
# This may be because most counties vote republican, which our model
# captures, but also makes it easier to have a high likelihood of 
# being right, as long as your model mostly predicts republican outcomes
predictions = evaluate_model(test, predictions) 
predictions.to_csv('model.csv')
