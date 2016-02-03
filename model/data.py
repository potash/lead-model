from drain.step import Step
from drain import util, data
from drain.util import day

import pandas as pd
import numpy as np
import logging

class LeadData(Step):
    EXCLUDE = {'first_name', 'last_name', 'address_residential', 'address'}
    AUX = { 'wic_min_date', 'test_min_date', 'address_count', 
            'test_count', 'first_bll6_sample_date', 'first_bll10_sample_date', 
            'max_bll', 'first_sample_date', 'last_sample_date'}
    PARSE_DATES = ['date_of_birth', 'test_min_date', 'wic_min_date', 'first_bll6_sample_date',
            'first_bll10_sample_date', 'first_sample_date', 'last_sample_date']

    def __init__(self, month, day, year_min=2005, **kwargs):
        Step.__init__(self, month=month, day=day, year_min=year_min, **kwargs)

    def run(self):
        engine = util.create_engine()

        # Read data
        X = pd.read_sql("""
select * from output.kids join output.kid_addresses using (kid_id)
join output.addresses using (address_id)
where date_of_birth >= '{date_min}'
        """.format(date_min='%s-%s-%s' % (self.year_min, self.month, self.day)), 
            engine, parse_dates=self.PARSE_DATES)

        # Set index
        X.set_index(['kid_id', 'address_id'], inplace=True)

        tests = pd.read_sql('select * from output.tests', engine)

        # Separate aux
        aux = X[list(self.AUX)]
        X = data.select_features(X, exclude=(self.AUX | self.EXCLUDE))
        
        return {'X':X, 'aux':aux}

class LeadTransform(Step):
    EXCLUDE = {'address_id', 'building_id', 'complex_id', 'census_block_id', 
            'census_tract_id', 'ward_id', 'community_area_id', 'date_of_birth'}

    def __init__(self, month, day, year, train_years, 
            training_min_max_sample_age = 3*365,
            **kwargs):
        Step.__init__(self, month=month, day=day, year=year, train_years=train_years, 
                training_min_max_sample_age=training_min_max_sample_age, **kwargs)

    def run(self, X, aux):
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
        train &= ((aux.last_sample_date - X.date_of_birth)/day) > self.training_min_max_sample_age

        test = data.index_as_series(X, 'date') == today
        aux.drop(aux.index[~(train | test)], inplace=True)
        X,train,test = data.train_test_subset(X, train, test)

        logging.info('Binarizing')
        # binarize census tract
        #data.binarize(X, {'community_area_id'})
    
        # set outcome variable
        y = (aux.first_bll6_sample_date < today).fillna(False)

        X = data.select_features(X, exclude=self.EXCLUDE)

        return {'X': X.astype(np.float32), 'y': y, 'train': train, 'test': test, 'aux': aux}
