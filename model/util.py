import sqlalchemy
import os
import numpy as np
import datetime
import pandas as pd
from sklearn import preprocessing
from scipy import stats

def create_engine():
    return sqlalchemy.create_engine('postgresql://{user}:{pwd}@{host}:5432/{db}'.format(
            host=os.environ['PGHOST'], db=os.environ['PGDATABASE'], user=os.environ['PGUSER'], pwd=os.environ['PGPASSWORD']))

def get_class(name):
    i = name.rfind('.')
    cls = name[i+1:]
    module = name[:i]
    
    mod = __import__(module, fromlist=[cls])
    return getattr(mod,cls)

def init_object(name, **kwargs):
    return get_class(name)(**kwargs)

def randtimedelta(low, high, size):
    d = np.empty(shape=size, dtype=datetime.timedelta)
    r = np.random.randint(low, high, size=size)
    for i in range(size):
        d[i] = datetime.timedelta(r[i])
    return d

def randdates(start,end, size):
    d = np.empty(shape=size, dtype=datetime.datetime)
    r = randtimedelta(0, (end-start).days, size)
    for i in range(size):
        d[i] = start + r[i]
    return d

# normalize a dataframes columns
# method = 'normalize': use standard score i.e. (X - \mu) / \sigma
# method = 'percentile': replace with percentile. SLOW
def normalize(df, method):
    if method == 'standard':
        return pd.DataFrame(preprocessing.scale(df), index=df.index, columns=df.columns)
    elif method == 'percentile':
        return df.rank(pct=True)

def get_collinear(df, tol=.1, verbose=False):
    q, r = np.linalg.qr(df)
    diag = r.diagonal()
    if verbose:
        for i in range(len(diag)):
            if np.abs(diag[i]) < tol:
                print r[:,i] # TODO print equation with column names!
    return [df.columns[i] for i in range(len(diag)) if np.abs(diag[i]) < tol]

def drop_collinear(df, tol=.1, verbose=True):
    columns = get_collinear(df, tol=tol)
    if (len(columns) > 0) and verbose:
        print 'Dropping collinear columns: ' + str(columns)
    df.drop(columns, axis=1, inplace=True)
    return df
