import pandas as pd
import numpy as np
import datetime

from joblib import load
from sklearn.pipeline import Pipeline
import holidays
#from sklearn.ensemble import RandomForestRegressor


def load_model():
    model = load('./rf_pipeline.joblib')
    return model
    
def create_feat(df, win=6):
    # Inspired by
    # https://stackoverflow.com/questions/47482009/pandas-rolling-window-to-return-an-array
    as_strided = np.lib.stride_tricks.as_strided
    
    cols = ['amount', 'Zero_amount', 'outgoing', 'incoming', 
            'balance', 'balance_start', 'month', 'year', 'work']
    res = pd.DataFrame(columns=cols)
    for i in cols:
        v = as_strided(df[i], (len(df[i]) - (win - 1), win), (df[i].values.strides * 2))
        res.loc[:, i] = pd.Series(v.tolist(), index=range(0, len(v)))
    return res
    
def clean(transactions, account):
    df = pd.DataFrame(map(dict, transactions))
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(by=['date'], ascending=[False]).reset_index(drop=True)

    # We create a column to keep track of the zero amount transactions
    df.loc[:, 'Zero_amount'] = (df['amount']==0).astype(int)

    outgoing = df[df['amount']<0].groupby('date')['amount'].sum().rename('outgoing')
    incoming = df[df['amount']>0].groupby('date')['amount'].sum().rename('incoming')

    df = df.groupby('date').sum()\
                           .merge(outgoing, how='left', left_index=True, right_index=True)\
                           .merge(incoming, how='left', left_index=True, right_index=True)\
                           .fillna(0).reset_index()
               
    # Calculate the sum of the transactions on 30 days intervals
    df = df.set_index('date')\
           .resample('30D').sum().reset_index()


    # Find where the accounts change and set the latest account balance
    df.loc[df['date'] == df['date'].max(), 'balance'] = dict(account)['balance']

    # Where the balance is null, initialize the balance value to the inverse of the transaction
    df.loc[df['balance'].isnull(), 'balance'] = -df['amount'].shift()
    # then use a cumulative sum divided by account to find the value of the balance after each operation
    df.loc[:, 'balance'] = df['balance'].cumsum()

    # Subtract balance and amount to get the balance before the transaction
    df.loc[:, 'balance_start'] = df['balance']-df['amount']
    
    # The year and month at the beginning of the 30 days
    df.loc[:, 'month'] = df['date'].dt.month
    df.loc[:, 'year'] = df['date'].dt.year

    # we count how many working days there were in the 30 days period
    fr_hol = holidays.France(years=df['date'].dt.year).keys()
    df.loc[:, 'work'] = df['date'].apply(lambda x: np.busday_count(x.date(), x.date() + datetime.timedelta(30),
                                                                   holidays=[np.datetime64(x) for x in fr_hol]))
                                                                   
    # For the moment the model uses only the last 6 months
    df = df[df['date']>df['date'].max() - datetime.timedelta(180)]

    features = create_feat(df, win=6)
    # Cancel series using spurious accounts id
    for i in features.columns:
        colnames = ['{}_{}'.format(i,j) for j in range(6,0,-1)]
        features[colnames] = pd.DataFrame(features[i].values.tolist(), index= features.index)
        features = features.drop(i, axis=1)
        
    return features
    
def predict_outgoing(transactions, account):
    pipe=load_model()
    X = clean(transactions, account)

    return pipe.predict(X)[0]
 


