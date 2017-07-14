from drain.step import Step
from drain import util, data

from lead.features.kids import revise_kid_addresses
from lead.model.data import LeadData

from datetime import date
import pandas as pd
import numpy as np
import logging
from drain.util import lru_cache


YEAR_MIN = 2003
YEAR_MAX = 2017

@lru_cache(maxsize=10)
def lead_data(month, day, wic_lag):
    ld = LeadData(month=month, day=day, year_min=YEAR_MIN, year_max=YEAR_MAX, 
                    wic_lag=wic_lag)
    ld.target = True
    return ld


class LeadCrossValidate(Step):
    """
    This step takes the output of LeadData and prepares it
    for modeling by selecting a training and test set and
    auxillary features used for evaluation.
    """
    def __init__(self, month, day, year, train_years, 
            wic_lag=None, train_query=None):
        """
        Args:
            month: the month of the train-test split
            day: the day of the train-test split
            year: the year of the train-test split
            train_years: the number of training years
            wic_lag: an optional lag for the wic data, in days
            train_query: an optional additional query for training
        """
        Step.__init__(self,
                month=month, day=day, year=year, 
                train_years=train_years,
                wic_lag=wic_lag,
                train_query=train_query)

        if not YEAR_MIN <= year <= YEAR_MAX:
            raise ValueError('Invalid year: %s' % year)

        today = date(year, month, day)
        # use kid_addresses_revised for a revised aux matrix for temporally valid training queries
        kid_addresses_revised = revise_kid_addresses(date=today)
        self.inputs = [lead_data(month, day, wic_lag), kid_addresses_revised]

    def run(self, revised, X, aux):
        """
        Args:
            revised: auxillary informtaion revised for the train-test date
            X: the full feature matrix from LeadData
            aux: the unrevised auxillary data from LeadData
        """
        logging.info('Splitting train and test sets')
        today = util.timestamp(self.year, self.month, self.day)
        min_date = util.timestamp(self.year - self.train_years, self.month, self.day)

        date = data.index_as_series(X, 'date')
        X = X[date.between(min_date, today)]
        aux = aux[date.between(min_date,today)]

        date = data.index_as_series(aux, 'date')
        train = (date < today) & (aux.address_min_date < today)
        test = date == today

        aux = revise_helper(revised=revised, aux=aux, 
                train=train, test=test, today=today)

        if self.train_query is not None:
            train &= train.index.isin(aux.query(self.train_query).index)

        aux = aux[train | test]
        X,train,test = data.train_test_subset(X, train, test, drop=False)

        c = data.non_numeric_columns(X)
        if len(c) > 0:
            logging.warning('Non-numeric columns: %s' % c)

        return {'X': X, 'aux':aux,
                'train': train, 'test': test}

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
