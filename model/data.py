import pandas as pd
from sklearn import preprocessing
from functools import partial
from scipy import stats
import re
import numpy as np
import collections

import random
import datetime
import model
from lead.output.aggregate import aggregate
import util
import warnings

CATEGORY_CLASSES = {
#    'address_building_condition': ['SOUND', 'NEEDS MINOR REPAIR',
#        'NEEDS MAJOR REPAIR', 'UNINHABITABLE'],
    'kid_sex' : ['M', 'F'],
    'test_type' : ['V', 'C'],
    'surname_ethnicity': ['black', 'white', 'api', 'aian', 'p2race'],
    'kid_ethnicity': ['black', 'white', 'hispanic', 'asian'],
    'tract_ethnicity': ['asian', 'black', 'white', 'latino'],
    'ward_ethnicity': ['black', 'white', 'latino'], # there are no majority asian wards
    'ward_id': range(1,51),
    'community_area_id': range(1,78)
}


class ModelData(object):
    def read(self, **args):
        raise NotImplementedError
    
    def write(self, **args):
        raise NotImplementedError
        
    def transform(self):
        raise NotImplementedError

class LeadData(ModelData):
    # default exclusions set
    # TODO: organize and explain these
    EXCLUDE = { 
        'kid_id', 'test_id', 'address_id', 'complex_id', 'census_tract_id', # ids
        'kid_first_name', 'kid_last_name', 'address_method', # strings
        'test_bll', 'test_minmax', 'kid_minmax_date', # leakage
        'test_date', 'kid_date_of_birth', #date 
        'test_year', 'join_year', # used for join
        'kid_birth_days_to_test',  'address_inspection_init_days_to_test' # variables that confuse the model?
    }

    DATE_COLUMNS = {
        'inspections' : ['init_date', 'comply_date']
    }
    
    KIDS_DATE_COLUMNS = ['kid_date_of_birth', 'test_date', 
                    'address_inspection_init_date', 'address_inspection_comply_date']
    
    def __init__(self, source, directory=None, 
                 tables=['inspections', 'tracts', 'wards', 'addresses', 'complexes', 'acs']):
        self.source = source
        self.directory = directory
        
        self.tables = {t:None for t in tables}
    
    def read(self, **args):
        if self.source == 'pkl':
            self.read_pkl()
        elif self.source == 'sql':
            self.read_sql()
        elif self.source == 'csv':
            self.read_csv()
        else:
            raise ValueError('Unexpected data source: ' + str(self.source))
    
    def read_sql(self):
        engine = util.create_engine()
        
        self.tests = pd.read_sql('select * from output.tests', engine)
        
        for table in self.tables:
            date_columns = self.DATE_COLUMNS[table] if table in self.DATE_COLUMNS else None
            self.tables[table] = pd.read_sql('select * from output.' + table, engine, parse_dates=date_columns)
            
    def write(self):
        self.tests.to_pickle(self.directory + '/tests.pkl')
        
        for table in self.tables:
            self.tables[table].to_pickle(self.directory + "/" + table + '.pkl')
        
    def read_pkl(self):
        self.tests = pd.read_pickle(self.directory + "/tests.pkl")
        
        for table in self.tables:
            self.tables[table] = pd.read_pickle(self.directory + "/" + table + '.pkl')
    
    def read_csv(self):
        self.tests = pd.read_csv(self.directory + "/tests.csv", index_col='test_id')
        
        for c in self.KIDS_DATE_COLUMNS:
            self.tests[c] = self.tests[c].apply(lambda x: 
                    x if pd.isnull(x) else datetime.datetime.strptime(x, '%Y-%m-%d').date())
        
        for table in self.tables:
            self.tables[table] = pd.read_csv(self.directory + "/" + table + '.csv')
            
    def transform(self, year, train_years,
                kid, tract, address, ward, address_history, tract_history, # boolean true or false
                bll_threshold=5, max_age=3*365, min_age=90, require_address=True, 

                # these parameters have defaults that were established by testing
                spacetime_normalize_method = None, # whether or not to normalize each year of tract data
                address_test_periods = [None],
                training='all', # all, preminmax
                training_max_age=None,
                testing='all',  # all, never_tested
                testing_max_age=None,
                community_area = False, # don't include community area binaries
                exclude={}, 
                undersample=None,
                impute=True, normalize=True, drop_collinear=False,
                multiaddress=False,
                census_tract_binarize=False,
                ward_id = None,
                building_year_decade=True,
                test_date_season=True,
                exclude_addresses=[296888, # Aunt Martha's Health Center
                                   70798,  # Union Health Service Inc. 
                                   447803]): # Former Maryville Hospital

        exclude = self.EXCLUDE.union(exclude)
        df = self.tests.merge(self.tables['addresses'], on='address_id', how='left', copy=False)
        df = df.merge(self.tables['complexes'], on='complex_id', how='left', copy=False)
        df['complex_assessor_null'].fillna(True, inplace=True)
        df['complex_building_null'].fillna(True, inplace=True)

        df['test_year'] = df['test_date'].apply(lambda d: d.year)
        age_mask = (df.test_kid_age_days >=  min_age) & (df.test_kid_age_days <= max_age)
        df = df[age_mask]
        df.set_index('test_id', inplace=True)
        if ward_id is not None:
            df = df[df.ward_id==ward_id]
            
        if  require_address:
            df = df[df.address_id.notnull() & df.ward_id.notnull() & df.census_tract_id.notnull() & df.community_area_id.notnull()]
            exclude.update({'address_null', 'ward_null', 'census_tract_null', 'community_area_null'})
        if exclude_addresses is not None:
            df = df[~df.address_id.isin(exclude_addresses)]

        # get tests relevant to date
        today = datetime.date(year, 1, 1)
        past_tests = df[df.test_date < today] # for test aggregation

        # censor minmax when it's in the future. for simplicity replaced with current test rather than minmax up to that date.
        past_minmax = df['kid_minmax_date'] < today 
        df['kid_minmax_bll'] = df['kid_minmax_bll'].where(past_minmax, df.test_bll)
        df['kid_minmax_date'] = df['kid_minmax_date'].where(past_minmax, df.test_date)

        date_from = datetime.date(year - train_years, 1, 1)
        date_mask = (df.test_date >= date_from)
        df = df[date_mask]

        # cross validation
        train = (df.test_date < today) 
        if training == 'preminmax':
            train = train & (df.test_date <= df.kid_minmax_date)
        if training_max_age is not None:
            train = train & (df.test_kid_age_days <= training_max_age)
      
        if testing == 'all':
            test = (df.test_date >= today) & (df.kid_date_of_birth < today) 
        elif testing == 'untested':
            test = (df.test_date >= today) & (df.kid_date_of_birth < today) & (df.min_sample_date >= today)
        else:
            print 'Warning: testing option \'{}\' not supported'.format(testing)

        if testing_max_age is not None:
            test = test & (df.test_kid_age_days <= testing_max_age)

        # want to get a single test for each future kid
        # if they get poisoned, take their first poisoned test
        # if they don't, take their first test
        df2 = df[test & ((df.test_bll > 5) == (df.kid_minmax_bll > 5))]
        testix = df2.groupby('kid_id')['test_kid_age_days'].idxmin()
        first_test = pd.Series(df.index.isin(testix), index=df.index)
        test = first_test & ((df.test_minmax & (df.test_bll > 5)) | (df.test_bll <= 5))

        # store past tests at all locations in train+test set for aggregation
        test_tract = past_tests.census_tract_id.isin(df.census_tract_id)
        test_address = past_tests.complex_id.isin(df.complex_id)
        past_tests_tract = past_tests[test_tract]
        past_tests_address = past_tests[test_address]

        train_or_test = train | test
        train = train.loc[train_or_test]
        test = test.loc[train_or_test]
        self.cv = (train,test)
        df = df[train_or_test].copy()

        # set test details for (future) test set to nan to eliminate leakage!
        # can't know about minmax bll,date for future poisonings!
        test_columns = [c for c in df.columns if c.startswith('test_')]
        df.loc[ test, test_columns ] = np.nan

        df.reset_index(inplace=True)

        # generate a fake test date for future tests
        epoch = datetime.date.fromtimestamp(0)
        past_test = df.test_date < today # aka training
        mean_age = datetime.timedelta(df['test_kid_age_days'].mean())
        future_test_date = (df['kid_date_of_birth'] + mean_age).apply(lambda d: max(d, today))
        df['test_date'] = df['test_date'].where(past_test, future_test_date)

        df['kid_date_of_birth_month'] = df['kid_date_of_birth'].apply(lambda d: d.month)
        df['kid_birth_date'] = df['kid_date_of_birth']
    
        for c in ['kid_birth']: #['address_inspection_init', 'address_inspection_comply', 'kid_birth']:
            df[c + '_days'] = df[c + '_date'].apply(lambda d: None if pd.isnull(d) else (d - epoch).days)
            df[c + '_days_to_test'] = pd.to_timedelta((df['test_date'] - df[c + '_date']), 'D').astype(int)
            df.drop(c + '_date', axis=1, inplace=True)
        
        df['join_year'] = df.test_date.apply(lambda d: min(d.year-1, year-1)) 
        
        if test_date_season:
            df['test_date_month'] = df['test_date'].apply(lambda d: d.month).where(past_test)
            CATEGORY_CLASSES['test_date_month'] = range(1,13)

        # spatial 
        if not kid:
            exclude.add('kid_.*')
        if not address:
            #TODO update this for complexes
            exclude.update(['address_building_.*', 'address_assessor_.*', 'address_lat', 'address_lng'])
        if tract:
            prefix_columns(self.tables['tracts'], 'tract_', 'census_tract_id')
            df = df.merge(self.tables['tracts'], on=['census_tract_id'], how='left', copy=False)
        if not ward:
            exclude.add('ward_id')
        if not community_area:
            exclude.add('community_area_id')
        
        # spatio-temporal
        years = range(year-train_years, year)
        inspections_tract_ag,inspections_address_ag = self.aggregate_inspections(years, levels=['census_tract_id', 'complex_id'])
        tests_tract_ag = self.aggregate_tests(levels=['census_tract_id'], df=past_tests_tract, multiaddress=multiaddress)[0]


        prefix_columns(inspections_tract_ag, 'tract_inspections_all_')
        prefix_columns(tests_tract_ag, 'tract_tests_1y_')
        spacetime_tract = inspections_tract_ag.join(tests_tract_ag, how='outer')

        prefix_columns(inspections_address_ag, 'address_inspections_cumulative_')
        spacetime_address = inspections_address_ag
        for period in address_test_periods:
            ta = self.aggregate_tests(levels=['complex_id'], years=years, period=period, df=past_tests_address, multiaddress=multiaddress)[0]
            prefix = str(period) + 'y' if period is not None else 'all'
            prefix_columns(ta, 'address_tests_' + prefix + '_')
            spacetime_address = spacetime_address.merge(ta, how='outer', left_index=True, right_index=True, copy=False)
        
        
        left = df[['complex_id', 'census_tract_id', 'join_year']].drop_duplicates()
        spacetime = left.merge(spacetime_tract, how='left', left_on=['census_tract_id', 'join_year'], right_index=True, copy=False)
        spacetime = spacetime.merge(spacetime_address, how='left', left_on=['complex_id', 'join_year'], right_index=True, copy=False)

        # acs data
        left = df[['census_tract_id', 'join_year']].drop_duplicates()
        acs = self.tables['acs'].set_index(['census_tract_id', 'year'])
        prefix_columns(acs, 'acs_5yr_')
        # use outer join for backfilling
        acs = left.merge(acs, how='outer', left_on=['census_tract_id', 'join_year'], right_index=True, copy=False)
        acs_filled = acs.groupby('census_tract_id').transform(lambda d: d.sort('join_year').fillna(method='backfill'))
        # left join and groupby preserved the left index but groupby dropped the tract
        # so put the tract back
        acs_filled['census_tract_id'] = acs['census_tract_id']

        spacetime = spacetime.merge(acs_filled, on=['census_tract_id', 'join_year'], how='left', copy=False)
        spacetime.set_index(['complex_id', 'join_year'], inplace=True)
        spacetime.drop(['census_tract_id'], axis=1, inplace=True)
        spacetime.fillna(0, inplace=True)

        if spacetime_normalize_method is not None:
            spacetime = spacetime.groupby(level='join_year').apply(lambda x: util.normalize(x, method=spacetime_normalize_method))

        df = df.merge(spacetime, left_on=['complex_id', 'join_year'], right_index=True, how='left', copy=False )

        if not address_history:
            exclude.update(['address_inspections_.*', 'address_tests_.*'])
        if not tract_history:
            exclude.update(['tract_inspections_.*', 'tract_tests_.*', 'acs_5yr_.*'])
        
        # additional features
        if census_tract_binarize:
            exclude.remove('census_tract_id')
            CATEGORY_CLASSES['census_tract_id'] = self.tables['tracts'].census_tract_id.values

        if building_year_decade:
            df['complex_building_year_decade'] = (df['complex_building_year'] // 10)
            CATEGORY_CLASSES['complex_building_year_decade'] =  df['complex_building_year_decade'].dropna().unique()
            #df['address_building_year_decade'] = (df['address_building_year'] // 10)
            #CATEGORY_CLASSES['address_building_year_decade'] =  df['address_building_year_decade'].dropna().unique()

        df.set_index('test_id', inplace=True)
        X,y = Xy(df, y_column = 'kid_minmax_bll', exclude=exclude, impute=impute, normalize=normalize, train=train)
 
        self.X = X
        self.y = y > bll_threshold

        if drop_collinear:
            util.drop_collinear(X)
        
        if undersample is not None:
            # undersample is the desired *proportion* of the majority class
            # calculate p, the desired proportion by which to undersample
            y_train = self.y[self.cv[0]]
            T = y_train.sum()
            F = len(y_train) - T
            p = (undersample)*T/((1-undersample)*F)
            self.cv = (undersample_cv(df, self.cv[0], p), self.cv[1])

    def aggregate_inspections(self, years, levels):
        inspections = self.tables['inspections']
        #res_columns = {'res_count': {'numerator': 'residential'}}
        #for index in levels:
        #    count = aggregate(self.tables['addresses'], res_columns, index=index)
        #    count.columns = [index + '_res_count']
        #    inspections = inspections.merge(count.reset_index(), how='left', on=index)
        
        for column in ['hazard_int','hazard_ext']:
            inspections[column].fillna(True, inplace=True)
            inspections[column] = inspections[column].astype(int)
        
        inspections['year'] = inspections['init_date'].fillna(inspections['comply_date']).apply(lambda d:d.year)
        inspections = join_years(inspections, years)
        
        comply_not_null = inspections[inspections.comply_date.notnull()]
        inspections['comply'] = (comply_not_null['comply_date'].apply(lambda d: d.year) < comply_not_null.year)
        inspections['comply'] = inspections['comply'].fillna(False).astype('int')
        dt = (inspections['comply_date'] - inspections['init_date']).where(inspections['comply'])
        inspections['days_to_compliance'] = dt[dt.notnull()] / np.timedelta64(1, 'D')
        
        INSPECTION_COLUMNS = {
            'count': {'numerator':1},
            'inspected': {'numerator':1, 'func': np.max},
            'hazard_int_count': {'numerator':'hazard_int'},
            'hazard_ext_count': {'numerator':'hazard_ext'},
            'hazard_int_prop': {'numerator':'hazard_int', 'denominator':1},
            'hazard_ext_ratio': {'numerator':'hazard_ext', 'denominator':1},
            'compliance_count': {'numerator': 'comply'},
            'compliance_prop': {'numerator': 'comply', 'denominator': 1},
            'avg_init_to_comply_days': {'numerator': 'days_to_compliance', 'func':'mean'},
        }
        
        r = []
        for level in levels:
            #INSPECTION_COLUMNS['pct_inspected'] = {'numerator': 1, 'denominator': level + '_res_count', 'denominator_func': np.max}
            r.append(aggregate(inspections, INSPECTION_COLUMNS, index=[level,'year']))
            
        return r
    
    def aggregate_tests(self, levels, period=1, years=None, df=None, multiaddress=False):
        ebll_test_count = lambda t: (t.test_bll > 5).astype(int)
        ebll_kid_ids = lambda t: t.kid_id.where(t.test_bll > 5)
        TEST_COLUMNS = {
            'count': {'numerator': 1},
            'tested': {'numerator': 1, 'func': np.max},
            'poisoned': {'numerator': lambda t: (t.test_bll > 5).astype(int), 'func':np.max},
            
            'ebll_count': {'numerator': ebll_test_count},
            'ebll_prop': {'numerator': ebll_test_count, 'denominator': 1},
            'avg_bll': {'numerator': 'test_bll', 'func':np.mean}, 
            'median_bll': {'numerator': 'test_bll', 'func':np.median}, 
            'max_bll': {'numerator': 'test_bll', 'func':np.max}, 
            'min_bll': {'numerator': 'test_bll', 'func':np.min}, 
            'std_bll': {'numerator': 'test_bll', 'func':np.std}, 
            
            'kid_count': {'numerator': 'kid_id', 'func':count_unique},

             # count number of kids with
            'kid_ebll_here_count': {'numerator': ebll_kid_ids, 'func': count_unique }, # ebll here
            'kid_ebll_first_count': {'numerator': lambda t: (t.test_minmax & (t.test_bll > 5))}, # first ebll here
        }
        if multiaddress:
            TEST_COLUMNS['kid_ebll_ever_count'] = {'numerator': lambda t: t.kid_id.where( t.kid_minmax_bll > 5 ), 'func': count_unique} # ever ebll
            TEST_COLUMNS['kid_ebll_future_count'] = {'numerator': lambda t: t.kid_id.where( (t.kid_minmax_bll > 5) & (t.kid_minmax_date >= t.test_date) ), 'func': count_unique} # future ebll
        if df is None: df = self.tests
        
        if period != 1:
            df = join_years(df, years, period, column='test_year')
        
        ag = [aggregate(df, TEST_COLUMNS, index=[level, 'test_year']) for level in levels]
        for a in ag:
            a['kid_ebll_here_prop'] = a['kid_ebll_here_count']/a['kid_count']
            a['kid_ebll_first_prop'] = a['kid_ebll_first_count']/a['kid_count']
            if multiaddress:
                a['kid_ebll_ever_prop'] = a['kid_ebll_ever_count']/a['kid_count']
                a['kid_ebll_future_prop'] = a['kid_ebll_future_count']/a['kid_count']
            a.index.rename('year', level='test_year', inplace=True)
        return ag

def join_years(left, years, period=None, column='year'):
    years = pd.DataFrame({column:years})
    if period is None:
        cond = lambda df: (df[column + '_left'] <= df[column + '_right'])
    else:
        cond = lambda df: (df[column + '_left'] <= df[column + '_right']) & (df[column +'_left'] > df[column + '_right'] - period)
        
    df = util.conditional_join(left, years, left_on=[column], right_on=[column], condition=cond)
    df.rename(columns={column + '_y': column}, inplace=True)
    return df

# generate year, month, day features from specified date features
def expand_dates(df, columns=[]):
    columns=df.columns.intersection(columns)
    df2 = df.reindex(columns=set(df.columns).difference(columns))
    for column in columns:
        df2[column + '_year'] = df[column].apply(lambda x: x.year)
        df2[column + '_month'] = df[column].apply(lambda x: x.month)
        df2[column + '_day'] = df[column].apply(lambda x: x.day)
    return df2

# binarize specified categoricals using specified category class dictionary
def binarize(df, category_classes):
    #for category,classes in category_classes.iteritems():
    columns = set(category_classes.keys()).intersection(df.columns)
    
    for category in columns:
        classes = category_classes[category]
        for i in range(len(classes)-1):
            df[category + '_' + str(classes[i]).replace( ' ', '_')] = (df[category] == classes[i])
        #binarized = pd.get_dummies(df[category], prefix=df[category])#.drop(classes[len(classes)-1], axis=1, inplace=True)
        #df = df.merge(binarized, left_index=True, right_index=True, copy=False)
        
    df.drop(columns, axis=1, inplace=True)                                      
    return df

# returns endogenous and exogenous variables
# normalization requires imputation (can't normalize null values)
# training mask is used for normalization
def Xy(df, y_column, include=None, exclude=None, impute=True, normalize=True, train=None):
    y = df[y_column]
    exclude.add(y_column)

    X = select_features(df, include, exclude)
    
    X = binarize(X, CATEGORY_CLASSES)
    
    nulcols = null_columns(X)
    if len(nulcols) > 0:
        print 'Warning: null columns ' + str(nulcols)
    
    nonnum = non_numeric_columns(X)
    if len(nonnum) > 0:
        print 'Warning: non-numeric columns ' + str(nonnum)
    
    if impute:
        imputer = preprocessing.Imputer()
        Xfit = X[train] if train is not None else X
        imputer.fit(Xfit)
        d = imputer.transform(X)
        if normalize:
            d = preprocessing.StandardScaler().fit_transform(d)
        X = pd.DataFrame(d, index=X.index, columns = X.columns)

    
    return (X,y)

def select_features(df, include=None, exclude=None, regex=True):

    if isinstance(include, collections.Iterable):
        columns = set.union(*[set(filter(re.compile('^' + feature + '$').match, df.columns)) for feature in include])
    else: 
        columns = set(df.columns)
        
    if isinstance(exclude, collections.Iterable):
        d = set.union(*[set(filter(re.compile('^'  + feature + '$').search, df.columns)) for feature in exclude])
        columns = columns.difference(d)
    
    df = df.reindex(columns = columns)
    return df

def prefix_columns(df, prefix, ignore=[]):
    df.columns =  [prefix + c if c not in ignore else c for c in df.columns]

def null_columns(df):
    nulcols = df.isnull().sum() == len(df)
    return nulcols[nulcols==True].index

def non_numeric_columns(df):
    columns = []
    for c in df.columns:
        try: 
            df[c].astype(float)
        except:
            columns.append(c)
            
    return columns

def get_correlates(df, c=.99):
    corr = df.corr().values
    for i in range(len(df.columns)):
        for j in range(i):
            if corr[i][j] >= c:
                print df.columns[i] + ', ' + df.columns[j] + ': ' + str(corr[i][j])
                
def undersample_cv(d, train, p):
    a = pd.Series([random.random() < p for i in range(len(d))], index=d.index)
    return train & ((d.test_bll > 5).values | a) 

def count_unique(series):
    return series.nunique()

