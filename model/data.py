from drain.step import Step
from drain import util, data
from lead.output import aggregations

import pandas as pd
import numpy as np
import logging

class LeadData(Step):
    EXCLUDE = {'first_name', 'last_name', 'address_residential', 'address'}
    AUX = { 'wic_min_date', 'test_min_date', 'address_count', 
            'test_count', 'first_bll6_sample_date', 'first_bll10_sample_date', 
            'max_bll', 'first_sample_date', 'last_sample_date'}
    PARSE_DATES = ['date_of_birth', 'test_min_date', 'wic_min_date', 'first_bll6_sample_date',
            'first_bll10_sample_date', 'first_sample_date', 'last_sample_date', 'wic_min_date', 'test_min_date']

    def __init__(self, month, day, year_min=2008, **kwargs):
        Step.__init__(self, month=month, day=day, year_min=year_min, **kwargs)

#        self.kids = FromSQL('select * from output.kids', 
#                parse_dates=self.PARSE_DATES['kids'])
#        self.kid_addresses = FromSQL('select * from output.kid_addresses', 
#                parse_dates=self.PARSE_DATES['kid_addresses'])
#        self.addresses = FromSQL('select * From output.addresses')

#        self.inputs = [self.kids, self.kid_addresses, self.addresses] + aggregations.buildings()# + aggregations.assessor()
        self.inputs = aggregations.buildings()

    def run(self, *args, **kwargs):
        engine = util.create_engine()

        # Read data
        X = pd.read_sql("""
select * from output.kids join output.kid_addresses using (kid_id)
join output.addresses using (address_id)
where date_of_birth >= '{date_min}'
        """.format(date_min='%s-%s-%s' % (self.year_min, self.month, self.day)), 
            engine, parse_dates=self.PARSE_DATES)
        
        # join before setting index
        X = self.inputs[0].join(X)

        # Set index
        X.set_index(['kid_id', 'address_id'], inplace=True)

        # Separate aux
        aux = X[list(self.AUX)]
        X = data.select_features(X, exclude=(self.AUX | self.EXCLUDE))

        # Sample dates used for training_min_max_sample_age in LeadTransform
        # TODO: could
        sample_dates = pd.read_sql("""
select kid_id, sample_date, date_of_birth
from output.tests join output.kids using (kid_id)""", engine, parse_dates=['date_of_birth', 'sample_date'])
        
        return {'X':X, 'aux':aux, 'sample_dates':sample_dates}

class LeadTransform(Step):
    EXCLUDE = {'address_id', 'building_id', 'complex_id', 'census_block_id', 
            'census_tract_id', 'ward_id', 'community_area_id', 'date_of_birth'}

    def __init__(self, month, day, year, train_years, 
            train_min_max_sample_age = 3*365,
            **kwargs):
        Step.__init__(self, month=month, day=day, year=year, train_years=train_years, 
                train_min_max_sample_age=train_min_max_sample_age, **kwargs)

    def run(self, X, aux, sample_dates):
        # TODO: move this into an HDFReader for efficiency
        drop = X.date_of_birth < util.timestamp(self.year-self.train_years-1, self.month, self.day)
        X.drop(X.index[drop], inplace=True)
        aux.drop(aux.index[drop], inplace=True)

        logging.info('Splitting train and test sets')
        today = util.timestamp(self.month, self.day, self.year)
        X['date'] = X.date_of_birth.apply(lambda t: util.date_ceil(t, self.month, self.day))

        X.set_index('date', append=True, inplace=True) # add date column to index
        aux.index = X.index

        train = data.index_as_series(X, 'date') < today
        # subset to potential training kids
        train_sample_dates = sample_dates.kid_id.isin(X[train].index.get_level_values('kid_id'))
        sample_dates.drop(sample_dates.index[~train_sample_dates], inplace=True)
        # calculate age
        sample_dates['age'] = (sample_dates.sample_date - sample_dates.date_of_birth)/util.day
        # find max sample age for each kid
        max_sample_age = sample_dates.groupby('kid_id')['age'].max()
        kids_min_max_sample_age = max_sample_age[(max_sample_age > self.train_min_max_sample_age)].index
        train &= data.index_as_series(X, 'kid_id').isin(kids_min_max_sample_age)
        

        train &= ((aux.last_sample_date - X.date_of_birth)/util.day) > self.train_min_max_sample_age

        test = data.index_as_series(X, 'date') == today
        aux.drop(aux.index[~(train | test)], inplace=True)
        X,train,test = data.train_test_subset(X, train, test)

        logging.info('Binarizing')
        # binarize census tract
        data.binarize(X, {'community_area_id'})
    
        # set outcome variable
        y = (aux.first_bll6_sample_date < today).fillna(False)

        X = data.select_features(X, exclude=self.EXCLUDE)
        X = data.impute(X, train=train)

        return {'X': X.astype(np.float32), 'y': y, 'train': train, 'test': test, 'aux': aux}
