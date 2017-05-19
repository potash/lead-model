from drain.step import Step
from drain.data import ToHDF
from drain import util, data
from drain.aggregation import SpacetimeAggregation

from lead.features.kids import revise_kid_addresses
from lead.model.data import LeadData

from sklearn import preprocessing
from datetime import date
import pandas as pd
import numpy as np
import logging
from drain.util import lru_cache

YEAR_MIN=2003
YEAR_MAX=2017

@lru_cache(maxsize=10)
def lead_data(month, day, wic_lag):
    ld = LeadData(month=month, day=day, year_min=YEAR_MIN, year_max=YEAR_MAX, 
                    wic_lag=wic_lag)
    ld.target = True
    return ld

class LeadTransform(Step):
    EXCLUDE = ['address_id', 'building_id', 'complex_id', 
            'census_block_id', 'census_tract_id', 'ward_id', 
            'community_area_id']

    def __init__(self, month, day, year, outcome_expr, train_years, 
            aggregations,
            wic_lag=None,
            train_query=None,
            spacetime_normalize=False,
            wic_sample_weight=1, exclude=[], include=[]):
        Step.__init__(self, month=month, day=day, year=year, 
                outcome_expr=outcome_expr,
                train_years=train_years,
                aggregations=aggregations,
                wic_lag=wic_lag,
                train_query=train_query,
                spacetime_normalize=spacetime_normalize,
                wic_sample_weight=wic_sample_weight, 
                exclude=exclude, include=include)

        if not YEAR_MIN <= year <= YEAR_MAX:
            raise ValueError('Invalid year: %s' % year)

        today = date(year, month, day)
        # use kid_addresses_revised for a revised aux matrix for temporally valid training queries
        kid_addresses_revised = revise_kid_addresses(date=today)
        self.inputs = [lead_data(month, day, wic_lag), kid_addresses_revised]

    def run(self, revised, X, aux):
        today = util.timestamp(self.year, self.month, self.day)
        min_date = util.timestamp(self.year - self.train_years, self.month, self.day)

        date = data.index_as_series(X, 'date')
        X = X[date.between(min_date, today)]
        aux = aux[date.between(min_date,today)]

        logging.info('Splitting train and test sets')
        # add date column to index

        # don't include future addresses in training
        date = data.index_as_series(aux, 'date')
        train = (date < today) & (aux.address_min_date < today)
        test = date == today

        revised = revise_helper(revised=revised, aux=aux, 
                train=train, test=test, today=today)

        if self.train_query is not None:
            old_train = train
            train &= train.index.isin(revised.query(self.train_query).index)
            revised = revised[revised.index.isin(train[train | test].index)]

        aux = aux[train | test]
        X,train,test = data.train_test_subset(X, train, test, drop=True)

        logging.info('Selecting aggregations')
        aggregations = self.inputs[0].aggregations # dictionary of Aggregations
        for a, args in self.aggregations.iteritems():
            X = aggregations[a].select(X, args, inplace=True)

        logging.info('Binarizing')
        X = data.binarize(X, ['community_area_id', 'ward_id'], astype=np.float32)
        
        # TODO: include gender, ethnicity, etc.
        y = revised.loc[X.index].eval(self.outcome_expr)
        X = data.select_features(X, exclude=self.EXCLUDE + self.exclude, 
                include=self.include)

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

        logging.info('Imputing')
        X = data.impute(X, train=train)

        sample_weight = 1 + (revised.wic * self.wic_sample_weight)

        c = data.non_numeric_columns(X)
        if len(c) > 0:
            logging.warning('Non-numeric columns: %s' % c)

        return {'X': X, 'y': y, 
                'train': train, 'test': test, 
                'aux': aux, 'sample_weight': sample_weight}

def revise_helper(revised, aux, train, test, today):
    """
    given revised and unrevised kid_addresses (aux), merge the unrevised for the test set
    with the revised for training
    """
    revised = aux[[]][train].reset_index().merge(revised, how='left', 
            on=['kid_id', 'address_id'])
    revised.set_index(['kid_id', 'address_id', 'date'], inplace=True)
    revised = pd.concat((revised, aux[test]))

    augment(revised)
    revised['today_age'] = (today - revised.date_of_birth)/util.day

    return revised

def augment(y):
    """
    augment the aux matrix with variables that are useful for train and test queries
    """
    y['age'] = (data.index_as_series(y, 'date') - y.date_of_birth) / util.day
    y['last_sample_age'] = (y.last_sample_date - y.date_of_birth) / util.day
    y['first_sample_age'] = (y.first_sample_date - y.date_of_birth) / util.day
    y['address_test_max_age'] = (y.address_test_max_date - y.date_of_birth) / util.day
    y['first_bll6_sample_age'] = (y.first_bll6_sample_date - y.date_of_birth) / util.day
    y['wic'] = y.first_wic_date.notnull()

    y['true6'] = y.max_bll >= 6
    y['true5'] = y.max_bll >= 5
    y['true4'] = y.max_bll >= 4

    # bll6 or tested after age about age 2
    y['true6_2y'] = y.true6.where((y.max_bll >= 6) | (y.last_sample_age > 365*1.9))
    # bll6 or tested at this address after about age 2
    y['true6_2y_here'] = y.true6.where((y.max_bll >= 6) | (y.address_test_max_age > 365*1.9))
    # bll6 at this address or tested at this address after about age 2
    y['true6_here_2y_here'] = y.true6.where((y.address_max_bll >= 6) | (y.address_test_max_age > 365*1.9))
