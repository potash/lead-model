from drain.step import Step
from drain import util, data
from drain.aggregation import SpacetimeAggregation

from sklearn import preprocessing
import pandas as pd
import numpy as np
import logging

class LeadTransform(Step):
    EXCLUDE = {'address_id', 'building_id', 'complex_id', 
            'census_block_id', 'census_tract_id', 'ward_id', 
            'community_area_id'}

    def __init__(self, month, day, year, train_years, 
            train_min_last_sample_age = 3*365, 
            train_non_wic = True,
            spacetime_normalize=False,
            wic_sample_weight=1, **kwargs):
        Step.__init__(self, month=month, day=day, year=year, 
                train_years=train_years, train_non_wic=train_non_wic,
                train_min_last_sample_age=train_min_last_sample_age,
                spacetime_normalize=spacetime_normalize,
                wic_sample_weight=wic_sample_weight, **kwargs)

    def run(self, X, aux, sample_dates):
        # TODO: move this into an HDFReader for efficiency
        drop = aux.date_of_birth < util.timestamp(
                self.year-self.train_years-1, self.month, self.day)
        X.drop(X.index[drop], inplace=True)
        aux.drop(aux.index[drop], inplace=True)

        logging.info('Splitting train and test sets')
        today = util.timestamp(self.month, self.day, self.year)

        # add date column to index
        X.set_index('date', append=True, inplace=True) 
        aux.index = X.index

        date = data.index_as_series(aux, 'date')

        train = date < today
        if not self.train_non_wic:
            train &= aux.wic

        # don't include future addresses in training
        train &= (aux.wic_min_date < today) | (aux.test_min_date < today)
        # subset to potential training kids
        max_sample_ages = censor_max_sample_ages(
                X[train].index.get_level_values('kid_id'), 
                sample_dates, today)

        kids_min_max_sample_age = max_sample_ages[
                (max_sample_ages > self.train_min_last_sample_age)].index
        train &= (
                data.index_as_series(X, 'kid_id').isin(
                    kids_min_max_sample_age) |
                (aux.first_bll6_sample_date < today).fillna(False))
         
        test = data.index_as_series(X, 'date') == today
        aux.drop(aux.index[~(train | test)], inplace=True)
        X,train,test = data.train_test_subset(X, train, test)

        #logging.info('Binarizing')
        # TODO: include gender, ethnicity, etc.
        # binarize census tract
        # data.binarize(X, {'community_area_id'})
    
        # set outcome variable, censored in training
        y = aux.first_bll6_sample_date.notnull().where(
            test | (aux.first_bll6_sample_date < today), False)

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
                aux.wic_min_date.notnull() * self.wic_sample_weight)

        c = data.non_numeric_columns(X)
        if len(c) > 0:
            logging.warning('Non-numeric columns: %s' % c)

        return {'X': X.astype(np.float32), 'y': y, 
                'train': train, 'test': test, 
                'aux': aux, 'sample_weight': sample_weight}

def censor_max_sample_ages(kids, sample_dates, today):
    # get max sample age for specified kids, censoring by today
    train_sample_dates = sample_dates.kid_id.isin(kids)
    sample_dates.drop(sample_dates.index[~train_sample_dates], inplace=True)
    # calculate age
    sample_dates['age'] = (sample_dates.sample_date - sample_dates.date_of_birth)/util.day
    # find max sample age for each kid
    max_sample_ages = sample_dates.groupby('kid_id')['age'].max()
    return max_sample_ages
