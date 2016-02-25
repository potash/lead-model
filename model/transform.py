from drain.step import Step
from drain import util, data
from drain.aggregation import SpacetimeAggregation

from lead.output.kids import revise_kid_addresses
from lead.model.data import LeadData

from sklearn import preprocessing
from datetime import date
import pandas as pd
import numpy as np
import logging

class LeadTransform(Step):
    EXCLUDE = {'address_id', 'building_id', 'complex_id', 
            'census_block_id', 'census_tract_id', 'ward_id', 
            'community_area_id'}

    def __init__(self, month, day, year, outcome_expr, train_years, 
            train_query=None,
            spacetime_normalize=False,
            wic_sample_weight=1, **kwargs):
        Step.__init__(self, month=month, day=day, year=year, 
                outcome_expr=outcome_expr,
                train_years=train_years,
                train_query=train_query,
                spacetime_normalize=spacetime_normalize,
                wic_sample_weight=wic_sample_weight, **kwargs)

        lead_data = LeadData(month=month, day=day, 
                target=True)
        today = date(year, month, day)
        kid_addresses_revised = revise_kid_addresses(date=today, test=True)

        self.inputs = [lead_data, kid_addresses_revised]

    def run(self, kid_addresses_revised, X, aux):
        today = util.timestamp(self.month, self.day, self.year)

        # TODO: move this into an HDFReader for efficiency
        drop = aux.date_of_birth < util.timestamp(
                self.year-self.train_years-1, self.month, self.day)
        X.drop(X.index[drop], inplace=True)
        aux.drop(aux.index[drop], inplace=True)

        # align kid_addresses_revised with the index of X and aux
        revised = X[['date']].merge(kid_addresses_revised, how='left', 
                left_index=True, right_on=['kid_id', 'address_id'])
        revised.set_index(['kid_id', 'address_id', 'date'],
                inplace=True)
        revised['last_sample_age'] = (revised.last_sample_date - 
                 revised.date_of_birth)/util.day
        revised['wic'] = revised.wic_date.notnull()

        date = data.index_as_series(revised, 'date')
        revised['today_age'] = (today - revised.date_of_birth)/util.day
        revised['age'] = (date - revised.date_of_birth)/util.day
        revised['last_sample_here_age'] = (revised.address_test_max_date - 
                 revised.date_of_birth)/util.day

        logging.info('Splitting train and test sets')

        # add date column to index
        X.set_index('date', append=True, inplace=True) 
        aux.index = X.index

        date = data.index_as_series(aux, 'date')

        train = date < today
        # don't include future addresses in training
        train &= (aux.address_min_date < today)
        train &= train.index.isin(
                revised.query(self.train_query).index)
        test = date == today

        aux.drop(aux.index[~(train | test)], inplace=True)
        X,train,test = data.train_test_subset(X, train, test)

        #logging.info('Binarizing')
        # TODO: include gender, ethnicity, etc.
        # binarize census tract
        # data.binarize(X, {'community_area_id'})
    
        y = revised.loc[X.index].eval(self.outcome_expr)
        X = data.select_features(X, exclude=self.EXCLUDE)

        if self.spacetime_normalize:
            prefixes = ['%s_.*' % a.prefix for a in 
                    self.inputs[0].aggregations 
                        if isinstance(a, SpacetimeAggregation)]
            spacetime = data.select_regexes(X.columns, prefixes)
            logging.info('Normalize %s columns' % len(spacetime))
            X.loc[:, spacetime] = X.loc[:,spacetime].groupby(
                    level='date').apply(
                        lambda x: pd.DataFrame(preprocessing.scale(x), 
                        index=x.index, columns=x.columns))

        X = data.impute(X, train=train)

        sample_weight = 1 + (
                aux.address_wic_min_date.notnull() * self.wic_sample_weight)

        c = data.non_numeric_columns(X)
        if len(c) > 0:
            logging.warning('Non-numeric columns: %s' % c)

        return {'X': X.astype(np.float32), 'y': y, 
                'train': train, 'test': test, 
                'aux': aux, 'sample_weight': sample_weight}
