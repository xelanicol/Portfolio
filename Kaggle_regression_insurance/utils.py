import pandas as pd
import numpy as np
def preprocess_data(df):
    # convert text features to categorical dtype
    df.Gender = df.Gender.astype('category')
    df['Marital Status'] = df['Marital Status'].astype('category')
    df['Education Level'] = df['Education Level'].astype('category')
    df['Occupation'] = df['Occupation'].astype('category')
    df['Location'] = df['Location'].astype('category')
    df['Policy Type'] = df['Policy Type'].astype('category')
    df['Customer Feedback'] = df['Customer Feedback'].astype('category')
    df['Smoking Status'] = df['Smoking Status'].astype('category')
    df['Exercise Frequency'] = df['Exercise Frequency'].astype('category')
    df['Property Type'] = df['Property Type'].astype('category')

    # previous claims has outliers - mostly N/A, 0, 1, 2. Convert to categorical
    df['Previous Claims'] = df['Previous Claims'].clip(0,2).replace({np.nan:'unknown',0:'none',1:'one',2:'two_or_more'})

    cat_cols = ['Gender','Marital Status', 'Education Level', 'Occupation',
        'Location', 'Policy Type',
        'Customer Feedback', 'Smoking Status', 'Exercise Frequency',
        'Property Type','Previous Claims']

    # convert categorical features to binary features, remove redundant features
    df = pd.get_dummies(df,columns=cat_cols,drop_first=True)

    df['Policy Start Date'] = pd.to_datetime(df['Policy Start Date']).apply(lambda x: x.timestamp())
    df.fillna(df.median(),inplace=True)

    return df