import pandas as pd
import numpy as np


import wrangle as w
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

import seaborn as sns
import matplotlib.pyplot as plt

import os
import acquire as a

def split_data(df):
    '''
    take in a DataFrame and target variable. return train, validate, and test DataFrames.
    return train, validate, test DataFrames.
    '''
    train_validate, test = train_test_split(df, test_size=.2, random_state=123)
    train, validate = train_test_split(train_validate, 
                                       test_size=.25, 
                                       random_state=123)
    return train, validate, test

def scaled_df(train, validate, test):
    """
    This function scales the train, validate, and test data using the MinMaxScaler.

    Parameters:
    train (pandas DataFrame): The training data.
    validate (pandas DataFrame): The validation data.
    test (pandas DataFrame): The test data.

    Returns:
    Tuple of:
        X_train_scaled (pandas DataFrame): The scaled training data.
        X_validate_scaled (pandas DataFrame): The scaled validation data.
        X_test_scaled (pandas DataFrame): The scaled test data.
        y_train (pandas Series): The target variable for the training data.
        y_validate (pandas Series): The target variable for the validation data.
        y_test (pandas Series): The target variable for the test data.
    """

    X_train = train[['fixed_acidity','volatile_acidity','citric_acid','residual_sugar','chlorides','free_sulfur_dioxide','total_sulfur_dioxide', 'density','ph','sulphates','alcohol','bound_sulfur_dioxide']]
    X_validate = validate[['fixed_acidity','volatile_acidity','citric_acid','residual_sugar','chlorides','free_sulfur_dioxide','total_sulfur_dioxide', 'density','ph','sulphates','alcohol','bound_sulfur_dioxide']]
    X_test = test[['fixed_acidity','volatile_acidity','citric_acid','residual_sugar','chlorides','free_sulfur_dioxide','total_sulfur_dioxide', 'density','ph','sulphates','alcohol','bound_sulfur_dioxide']]

    y_train = train.quality
    y_validate = validate.quality
    y_test = test.quality

    #making our scaler
    scaler = MinMaxScaler()
    #fitting our scaler 
    # AND!!!!
    #using the scaler on train
    X_train_scaled = scaler.fit_transform(X_train)
    #using our scaler on validate
    X_validate_scaled = scaler.transform(X_validate)
    #using our scaler on test
    X_test_scaled = scaler.transform(X_test)

    # Convert the array to a DataFrame
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)

    # Convert the array to a DataFrame
    X_validate_scaled = pd.DataFrame(X_validate_scaled, columns=X_validate.columns, index=X_validate.index)
    
    # Convert the array to a DataFrame
    X_test_scaled = pd.DataFrame(X_test_scaled,columns=X_test.columns, index=X_test.index)
    
    return X_train_scaled, X_validate_scaled, X_test_scaled, y_train, y_validate, y_test

#set summarize function
def summarize(df):
    '''
    summarize will take in a single argument (a pandas dataframe) 
    and output to console various statistics on said dataframe, including:
    # .head()
    # .info()
    # .describe()
    # .value_counts()
    # observation of nulls in the dataframe
    #distrinution of numerical attributes
    '''
    print(f"""SUMMARY REPORT
=====================================================
          
          
Dataframe head: 
{df.head()}
          
=====================================================
          
          
Dataframe info: """)
    df.info()

    print(f"""=====================================================
          
          
Dataframe Description: 
{df.describe().T}
          
=====================================================


nulls in dataframe by column: 
{nulls_by_col(df)}
=====================================================


nulls in dataframe by row: 
{nulls_by_row(df)}
=====================================================
    
    
DataFrame value counts: 
 """)         
    for col in (get_object_cols(df)): 
        print(f"""******** {col.upper()} - Value Counts:
{df[col].value_counts()}
    _______________________________________""")                   
        
# fig, axes = plt.subplots(1, len(get_numeric_cols(df)), figsize=(15, 5))
    
    for col in get_numeric_cols(df):
        sns.histplot(df[col])
        plt.title(f'Histogram of {col}')
        plt.show()  

def nulls_by_col(df):
    """
    This function will:
        - take in a dataframe
        - assign a variable to a Series of total row nulls for ea/column
        - assign a variable to find the percent of rows w/nulls
        - output a df of the two variables.
    """
    num_missing = df.isnull().sum()
    pct_miss = (num_missing / df.shape[0]) * 100
    cols_missing = pd.DataFrame({
                    'num_rows_missing': num_missing,
                    'percent_rows_missing': pct_miss
                    })
    
    return  cols_missing

def nulls_by_row(df):
    """
    This function will:
        - take in a dataframe
        - assign a variable to a Series of total row nulls for ea/column
        - assign a variable to find the percent of rows w/nulls
        - output a df of the two variables.
    """
    num_missing = df.isnull().sum(axis=1)
    pct_miss = (num_missing / df.shape[1]) * 100
    
    rows_missing = pd.DataFrame({
                    'num_cols_missing': num_missing,
                    'percent_cols_missing': pct_miss
                    })
    
    return rows_missing

def get_object_cols(df):
    '''
    This function takes in a dataframe and identifies the columns that are object types
    and returns a list of those column names. 
    '''
    # get a list of the column names that are objects (from the mask)
    object_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    return object_cols

def get_numeric_cols(df):
    '''
    This function takes in a dataframe and identifies the columns that are object types
    and returns a list of those column names. 
    '''
    # get a list of the column names that are objects (from the mask)
    num_cols = df.select_dtypes(exclude=['object', 'category']).columns.tolist()
    
    return num_cols

#set summarize function
def summarize2(df_clean):
    '''
    summarize will take in a single argument (a pandas dataframe) 
    and output to console various statistics on said dataframe, including:
    # .head()
    # .info()
    # .describe()
    # .value_counts()
    # observation of nulls in the dataframe
    #distrinution of numerical attributes
    '''
    print(f"""SUMMARY REPORT
=====================================================
          
          
Dataframe head: 
{df_clean.head(3)}
          
=====================================================
          
          
Dataframe info: """)
    df_clean.info()

    print(f"""=====================================================
          
          
Dataframe Description: 
{df_clean.describe().T}
          
=====================================================


nulls in dataframe by column: 
{nulls_by_col(df_clean)}
=====================================================


nulls in dataframe by row: 
{nulls_by_row(df_clean)}
=====================================================
    
    
DataFrame value counts: 
 """)         
    for col in (get_object_cols(df_clean)): 
        print(f"""******** {col.upper()} - Value Counts:
{df_clean[col].value_counts()}
    _______________________________________""")                   
        
# fig, axes = plt.subplots(1, len(get_numeric_cols(df)), figsize=(15, 5))
    
    for col in get_numeric_cols(df_clean):
        sns.histplot(df_clean[col])
        plt.title(f'Histogram of {col}')
        plt.show()  

def get_superstore():
    # Check if the CSV file exists
    if os.path.exists('superstore_prepped.csv'):
        # If the file exists, load it into a data frame
        df = pd.read_csv('superstore_prepped.csv')
         # set datetime
        df.sale_date = pd.to_datetime(df.sale_date)
         # set index as date
        df = df.set_index('sale_date')
        # sort index
        df = df.sort_index()
    else:
        
        #get and read csv
        df = pd.read_csv('ts_superstore.csv', index_col=0)

        # format sale date
        df.sale_date = df.sale_date.str.replace('00:00:00 GMT','')

        # remove wihite space
        df.sale_date = df.sale_date.str.strip()

        # change to date and time
        df.sale_date = pd.to_datetime(df.sale_date, format='%a, %d %b %Y')

        #set index as sale_date
        df = df.set_index('sale_date')

        #sort index values
        df = df.sort_index()

        #add day of week column
        df['day_of_week'] = df.index.day_name()

        #add month column
        df['month'] = df.index.month_name()

        #add sales total column
        df['sales_total']= (df.sale_amount * df.item_price)

        # Write the data frame to a CSV file
        df.to_csv('superstore_prepped.csv', index=True)

    return df

def get_germany():
    # check if csv exists
    if os.path.exists('germany_power_prepped.csv'):
        # If the file exists, load it into a data frame
        df = pd.read_csv('germany_power_prepped.csv')
         # set datetime
        df.date = pd.to_datetime(df.date)
         # set index as date
        df = df.set_index('date')
        # sort index
        df = df.sort_index()
    else:
        # define data source
        url = 'https://raw.githubusercontent.com/jenfly/opsd/master/opsd_germany_daily.csv'
        
        # turn into df from acquire function
        df = a.get_power(url)

        # rename columns
        df = df.rename(columns={'Date':'date','Consumption':'consumption','Wind':'wind','Solar':'solar','Wind+Solar':'wind_solar'})

        # set datetime
        df.date = pd.to_datetime(df.date)

        # set index as date
        df = df.set_index('date')

        # sort index
        df = df.sort_index()

        # add month column
        df['month'] = df.index.month_name() 

        # add year column
        df['year'] = df.index.strftime('%Y')

        # drop nulls
        df = df.dropna()

        # create other sources of energy column
        df['other_energy_sources'] = df.consumption - df.wind_solar

        # Write the data frame to a CSV file
        df.to_csv('germany_power_prepped.csv', index=True)
    
    return df