import pandas as pd
from sklearn import preprocessing
from functools import partial
from scipy import stats
import re
import os
import numpy as np
import collections
import re

import random
import datetime
from lead.output import tests_aggregated,buildings_aggregated,inspections_aggregated,permits_aggregated,violations_aggregated

from drain import util
from drain.util import prefix_columns, join_years
from drain.data import get_aggregation, ModelData, undersample_to
from drain import data

import warnings



SPATIAL_LEVELS = ['address_id', 'building_id', 'complex_id', 'census_block_id', 'census_tract_id', 'ward_id', 'community_area_id']

class LeadData(ModelData):
    basedir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DEPENDENCIES = [os.path.join(basedir, d) for d in ['psql/output/tests_aggregated', 'psql/output/buildings_aggregated', 'psql/output/permits_aggregated', 'psql/output/violations_aggregated', 'data/output/tests.pkl', 'data/output/acs.pkl']]

    # default exclusions set
    # TODO: organize and explain these
    EXCLUDE = { 
        'kid_id', 'test_id', 'address_id', 'wic_address_id', 'building_id', 'complex_id', 'census_block_id', 'census_tract_id', # ids
        'first_name', 'last_name', 'address_apt',
        'kid_first_name', 'kid_last_name', 'address_method', 'address', # strings
        'test_bll', 'test_minmax', 'kid_minmax_date', 'kid_max_bll', 'kid_max_date', # leakage
        'kid_min_sample_date', 'kid_max_sample_date', 'kid_max_sample_age_days',
        'test_date', 'kid_date_of_birth', #date 
        'join_year', 'aggregation_end',# used for join
        'first_address', # used for train=first_address
        'kid_birth_days_to_test',  'test_kid_age_days', 'address_inspection_init_days_to_test', # variables that confuse the model?
    }

    CATEGORY_CLASSES = {
        'kid_sex' : ['M', 'F'],
        'test_type' : ['V', 'C'],
        'surname_ethnicity': ['black', 'white', 'api', 'aian', 'p2race'],
        'kid_ethnicity': ['black', 'white', 'hispanic', 'asian'],
        'tract_ethnicity': ['asian', 'black', 'white', 'latino'],
        'ward_ethnicity': ['black', 'white', 'latino'], # there are no majority asian wards
        'ward_id': range(1,51),
        'community_area_id': range(1,78),
        'wic_clinic' : [ 'GreaterLawn', 'FriendFamily', 'LowerWest', 'NearWest', 'WestTown', 'ChicagoFamily', 'ErieSuperior', 'HenryBooth', 'WestsideHP', 'Lakeview', 'Englewood', 'Uptown', 'Austin'],
    }

    DATE_COLUMNS = {
        'inspections' : ['init_date', 'comply_date']
    }
    
    KIDS_DATE_COLUMNS = ['kid_date_of_birth', 'test_date', 
                    'address_inspection_init_date', 'address_inspection_comply_date']
    
    def __init__(self, directory,
                year, train_years,
                aggregation_end_lag=False):
        self.directory = directory
        self.tables = {'addresses':None, 'acs':None}
        self.year = year
        self.train_years = train_years
        self.aggregation_end_lag = aggregation_end_lag

        self.today = datetime.date(self.year, 1, 1)
    
    def write(self, dirname):
        self.df.to_hdf(os.path.join(dirname, 'df.h5'), 'df', mode='w')
        
    # read tables from disk
    def read_pkl(self):
        self.tests = pd.read_pickle(os.path.join(self.directory, 'tests.pkl'))
        
        for table in self.tables:
            self.tables[table] = pd.read_pickle(os.path.join(self.directory, table + '.pkl'))
    
    def read(self, dirname=None):
        if dirname is not None:
            self.df = pd.read_hdf(os.path.join(dirname, 'df.h5'), 'df')
            return

        engine = util.create_engine()
        self.read_pkl()

        # get tests relevant to date
        df = pd.concat((tests_aggregated.censor_tests(self.tests, self.today),
                        self.tests[self.tests.test_date >= self.today]), copy=False)

        date_from = datetime.date(self.year - self.train_years, 1, 1)
        date_mask = (df.test_date >= date_from)
        df = df[date_mask]
        df = df[df.kid_date_of_birth.notnull()]

        #if self.wic_address:
        #    df['address_id'] = df.address_id.where(~df.wic, df.wic_address_id)
        df = self._expand_wic_addresses(df)

        df = df.merge(self.tables['addresses'], on='address_id', how='left', copy=False)
        df = df[df.address_id.notnull() & df.census_tract_id.notnull()]

        train_or_first_test_subset(self.today, df)

        # used for train=first_address, indicates whether a past test is the first for that kid at that address
        # group by currbllshort_address so that if an address is wic but also partially currbllshort...
        cur_first_address = df[df.currbllshort_address][df.test_date < self.today].groupby(['kid_id', 'address_id'])['test_kid_age_days'].idxmin()
        wic_first_address = df[df.wic_address][df.test_date < self.today].groupby(['kid_id'])['test_kid_age_days'].idxmin()
        df['first_address'] = pd.Series(df.index.isin(pd.concat((cur_first_address, wic_first_address))), index=df.index)

        df['join_year'] = df.test_date.apply(lambda d: min(d.year-1, self.year-1)) 
        df['aggregation_end'] = df.test_date.apply(lambda d: util.datetime64(min(d.year, self.year), self.today.month, self.today.day))

        if self.aggregation_end_lag:
            train = df.test_date < self.today
            df.loc[train, 'aggregation_end'] = df.loc[train, 'test_date'].apply(lambda d: util.datetime64(min(d.year-1, self.year), self.today.month, self.today.day))
        
        # spatio-temporal
        end_dates = df['aggregation_end'].unique()
        
        spacetime_columns = SPATIAL_LEVELS + ['join_year', 'aggregation_end']
        left = df[ spacetime_columns ].drop_duplicates()

        spacetime = get_aggregation('output.tests_aggregated', tests_aggregated.level_deltas, engine, left=left, prefix='tests')
        spacetime = get_aggregation('output.inspections_aggregated', tests_aggregated.level_deltas, engine, left=spacetime, prefix='inspections')
        spacetime = get_aggregation('output.permits_aggregated', permits_aggregated.level_deltas, engine, left=spacetime, prefix='permits')
        spacetime = get_aggregation('output.violations_aggregated', violations_aggregated.level_deltas, engine, left=spacetime, prefix='violations')
        
        # acs data
        acs_left = df[['census_tract_id', 'join_year']].drop_duplicates()
        acs = self.tables['acs'].set_index(['census_tract_id', 'year'])
        prefix_columns(acs, 'acs_5yr_')
        # use outer join for backfilling
        acs = acs_left.merge(acs, how='outer', left_on=['census_tract_id', 'join_year'], right_index=True, copy=False)
        acs_filled = acs.groupby('census_tract_id').transform(lambda d: d.sort('join_year').fillna(method='backfill'))
        # left join and groupby preserved the left index but grupby dropped the tract
        # so put the tract back
        acs_filled['census_tract_id'] = acs['census_tract_id']

        spacetime = spacetime.merge(acs_filled, on=['census_tract_id', 'join_year'], how='left', copy=False)
        spacetime.set_index(['address_id', 'aggregation_end'], inplace=True)
        spacetime.drop(['census_tract_id', 'building_id', 'complex_id', 'census_block_id', 'census_tract_id', 'ward_id', 'community_area_id', 'join_year'], axis=1, inplace=True)
        spacetime.fillna(0, inplace=True)

        df = df.merge(spacetime, left_on=['address_id', 'aggregation_end'], right_index=True, how='left', copy=False )

        space_columns = SPATIAL_LEVELS + ['address_lat', 'address_lng']
        left = df[ space_columns ].drop_duplicates()
        space = self._get_building_aggregation(buildings_aggregated.levels, engine, left=left)
        space.set_index('address_id', inplace=True)
        space.drop(['census_tract_id', 'building_id', 'complex_id', 'census_block_id', 'census_tract_id', 'ward_id', 'community_area_id' ], axis=1, inplace=True)
        df = df.merge(space, left_on='address_id', right_index=True, how='left', copy=False)

        df.set_index(['test_id', 'currbllshort_address', 'wic_address'], inplace=True)

        self.df = df

    def _get_building_aggregation(self, building_aggregations, engine, left=None):
        df = get_aggregation('output.buildings_aggregated', building_aggregations, engine, left=left)
    
        # fillna on not_null columns
        not_null_columns = [c for c in df.columns if c.endswith('_not_null')]
        not_null_df = df.loc[:,not_null_columns]
        not_null_df = not_null_df.fillna(False)

        return df

    # when wic_address is not null and differs from (currbllshort) address
    # create another entry for it with address_id set to it
    # also drop wic_address_id and create booleans wic_address and currbllshort_address indicating that
    def _expand_wic_addresses(self, tests):
        tests['currbllshort_address'] = tests.address_id.notnull()
        tests['wic_address'] = tests.wic_address_id.notnull() & (tests['address_id'] == tests['wic_address_id'])
    
        wic_tests = tests[ (tests.wic_address_id != tests.address_id) & tests.wic_address_id.notnull() ].copy()
        wic_tests['address_id'] = wic_tests['wic_address_id']
        wic_tests['wic_address'] = True
        wic_tests['currbllshort_address'] = False
    
        tests = pd.concat((tests, wic_tests), ignore_index=True, copy=False)
        tests.drop('wic_address_id', axis=1, inplace=True)
        return tests

    def transform(self,
                bll_threshold=5,
                training='all', # all, preminmax
                testing='all',  # all, never_tested
                training_max_test_age=None,
                testing_max_test_age=None,
                testing_max_today_age=None,
                testing_min_today_age=0,
                community_area = False, # don't include community area binaries
                exclude={},
                include={},
                wic_sample_weight=1,
                ebll_sample_weight=1,
                impute=True, normalize=True, drop_collinear=False,
                spacetime_normalize_method = None, # whether or not to normalize each year of tract data
                buildings_impute_params=None,
                impute_strategy='mean', 
                ward_id = None, # filter to this particular ward
                cluster_columns={}, # dict of ('column_name': n_clusters) pairs
                test_date_season=True,
                kid=True, tract=True, address=True, ward=True, address_history=True, tract_history=True,
                spatial_resolution=None,
                normalize_date_of_birth=False,
                spacetime_differences={},
                testing_test_number = None,
                testing_masks = None,
                training_min_max_sample_age=None, # only use samples of sufficient age (or poisoned)
                training_wic_address=None
        ):

        df = self.df

        if ward_id is not None:
            df = df[df.ward_id==ward_id]

        train = (df.test_date < self.today)
        if training == 'preminmax':
            train = train & (df.test_date <= df.kid_minmax_date)
        elif training == 'max':
            train = train & ( (df.test_bll > 5) == (df.kid_max_bll > 5) )
        elif training == 'first':
            train = train & (df.kid_test_number == 1)
        elif training == 'first_address':
            train = train & df['first_address']

        if training_max_test_age is not None:
            train = train & (df.test_kid_age_days <= training_max_test_age)
        if training_min_max_sample_age is not None:
            df['kid_max_sample_age_days'] = (df.kid_min_sample_date - df.kid_date_of_birth) / np.timedelta64(1, 'D')
            train = train & ( (df.kid_max_sample_age_days >= training_min_max_sample_age) | (df.kid_max_bll > 5) ) 
        if training_wic_address is None:
            train = train & util.index_as_series(df, 'currbllshort_address')

        test = (df.test_date >= self.today) & (df.kid_date_of_birth.apply(lambda d: (self.today - d).days > testing_min_today_age))
        if testing == 'all':
            pass
        elif testing == 'untested':
            test = (df.test_date >= self.today) & (df.kid_date_of_birth < self.today) & (df.min_sample_date >= self.today)
        else:
            print 'Warning: testing option \'{}\' not supported'.format(testing)

        if testing_max_test_age is not None:
            test = test & (df.test_kid_age_days <= testing_max_test_age)
        if testing_max_today_age is not None:
            test = test & (df.kid_date_of_birth.apply(lambda d: (self.today - d).days <= testing_max_today_age))
        if testing_test_number is not None:
            test = test & (df.kid_test_number == testing_test_number)

        self.masks = pd.DataFrame({
            'infant': df.kid_date_of_birth < self.today,
            'in_utero': df.kid_date_of_birth >= self.today,
            'wic': df.wic,
        })
        self.masks.index = df.index

        if testing_masks is not None:
            test = test & reduce(lambda a,b: a & b, (self.masks[mask] for mask in testing_masks))

        df, train, test = data.train_test_subset(df, train, test)
        self.cv = (train,test)
        self.masks = self.masks.loc[df.index]

        # set test details for (future) test set to nan to eliminate leakage!
        # can't know about minmax bll,date for future poisoningsa!
        # since test_id is currently the index it does not get cleared!
        test_columns = [c for c in df.columns if c.startswith('test_')]
        df.loc[test, test_columns] = np.nan

        # generate a fake test date for future tests
        epoch = datetime.date.fromtimestamp(0)
        mean_age = datetime.timedelta(df['test_kid_age_days'].mean())
        future_test_date = df['kid_date_of_birth'].apply(lambda d: max(d+mean_age, self.today))
        df['test_date'] = df['test_date'].where(train, future_test_date)

        df['kid_date_of_birth_month'] = df['kid_date_of_birth'].apply(lambda d: d.month)
        df['kid_birth_date'] = df['kid_date_of_birth']


        for c in ['kid_birth']: #['address_inspection_init', 'address_inspection_comply', 'kid_birth']:
            df[c + '_days'] = df[c + '_date'].apply(lambda d: None if pd.isnull(d) else (d - epoch).days)
            df[c + '_days_to_test'] = pd.to_timedelta((df['test_date'] - df[c + '_date']), 'D').astype(int)
            df.drop(c + '_date', axis=1, inplace=True)

        if normalize_date_of_birth:
            df.loc[train, ['kid_birth_days']] = preprocessing.scale(df.loc[train, ['kid_birth_days']])
            df.loc[test, ['kid_birth_days']] = preprocessing.scale(df.loc[test, ['kid_birth_days']])

        for event in spacetime_differences:
            for space in spacetime_differences[event]:
                for t1, t2 in spacetime_differences[event][space]:
                    d = spacetime_difference(df, event, space, t1, t2)
                    df = pd.concat((df,d), axis=1, copy=False)
 
        #if spacetime_normalize_method is not None:
        #    spacetime = spacetime.groupby(level='aggregation_end').apply(lambda x: util.normalize(x, method=spacetime_normalize_method))

        exclude = self.EXCLUDE.union(exclude)

        if not address_history:
            exclude.update(['address_inspections_.*', 'address_tests_.*'])
        if not tract_history:
            exclude.update(['tract_inspections_.*', 'tract_tests_.*', 'acs_5yr_.*'])

        if not kid:
            exclude.add('kid_.*')
        if not address:
            #TODO update this for complexes
            exclude.update(['address_building_.*', 'address_assessor_.*', 'address_lat', 'address_lng'])
        if not ward:
            self.CATEGORY_CLASSES.pop('ward_id')
        if not community_area:
            exclude.add('community_area_id')

        if test_date_season:
            df['test_date_month'] = df['test_date'].apply(lambda d: d.month).where(train)
            self.CATEGORY_CLASSES['test_date_month'] = range(1,13)

        if spatial_resolution is not None:
            i = SPATIAL_LEVELS.index(spatial_resolution + '_id')
            if i > 0:
                exclude.update(map(lambda d: d[:-3] + '_.*', SPATIAL_LEVELS[:i-1]))

        if buildings_impute_params is not None:
            regex = re.compile('.*_(assessor|footprint)_.*')
            data_columns = filter(regex.match, df.columns)
            data.nearest_neighbors_impute(df, ['address_lat', 'address_lng'], data_columns, buildings_impute_params)

        X,y = data.Xy(df, y_column = 'kid_minmax_bll', exclude=exclude, category_classes=self.CATEGORY_CLASSES)

        if impute:
            X = data.impute(X, train=train)
            if normalize:
                X = data.normalize(X, train=train)

        if drop_collinear:
            util.drop_collinear(X)

        self.X = X
        self.y = y > bll_threshold

# this helper finds the first test for each kid_id in the test set if they haven't been poisoned as of today
# call that the test set, and drop (in place!) anything not in train (default = ~test) or test
def train_or_first_test_subset(today, df):
    train = (df.test_date < today)
    test = ~train
    # want to get a single test for each future kid
    # if they get poisoned, take their first poisoned test
    # if they don't, take their first test
    df2 = df[test]
    cur_testix = df2[df2.currbllshort_address].groupby('kid_id')['test_kid_age_days'].idxmin()
    wic_testix = df2[df2.wic_address].groupby('kid_id')['test_kid_age_days'].idxmin()
    first_test = pd.Series(df.index.isin(pd.concat((cur_testix, wic_testix))), index=df.index)
    test = first_test & ( ((df.kid_minmax_date >= today) & (df.kid_minmax_bll > 5)) | (df.kid_minmax_bll <= 5))

    df.drop(df.index[~(train | test)], inplace=True)

# return the column-wise subset corresponding to the given event space and time
# e.g. get_spacetime(df, 'tests', 'census_tract', '1y')
def get_spacetime(df, event, space, time):
    columns = filter(lambda c: c.startswith(space + '_' + event + '_' + time + '_'), df.columns)
    return df[columns]

# difference between two events at the same space in different times
def spacetime_difference(df, event, space, time1, time2):
    # get the two event dfs
    df1 = get_spacetime(df, event, space, time1)
    df2 = get_spacetime(df, event, space, time2)
    # remove the prefixes
    df1.columns = [c[len(space + '_' + event + '_' + time1 + '_'):] for c in df1.columns]
    df2.columns = [c[len(space + '_' + event + '_' + time2 + '_'):] for c in df2.columns]
    # difference
    df = df1 - df2
    # prefix
    prefix_columns(df, space + '_' + event + '_' + time1 + '-' + time2 + '_')
    return df
