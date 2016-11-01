from drain.step import Step
from drain import util, data
from drain.data import FromSQL, Merge
from drain.aggregation import SpacetimeAggregationJoin

from lead.output import aggregations
from lead.output.kids import KIDS_PARSE_DATES, KID_ADDRESSES_PARSE_DATES
from lead.model.left import LeadLeft

from datetime import date
import pandas as pd
import numpy as np
import logging

class LeadData(Step):
    def __init__(self, month, day, year_min, year_max, wic_lag=None, **kwargs):
        Step.__init__(self, month=month, day=day, 
                year_min=year_min, year_max=year_max,
                wic_lag=wic_lag,
                **kwargs)

        acs = FromSQL(table='output.acs', target=True)
        left = LeadLeft(month=month, day=day, year_min=year_min, target=True)

        dates = tuple((date(y, month, day) 
                for y in range(year_min, year_max+1)))
        self.aggregations = aggregations.all_dict(dates, wic_lag)
        self.aggregation_joins = [
                SpacetimeAggregationJoin(target=True, inputs=[left, a], 
                lag = wic_lag if name.startswith('wic') else None,
                inputs_mapping=[{'aux':None}, 'aggregation']) 
            for name, a in self.aggregations.iteritems()]

        self.inputs = [acs, left] + self.aggregation_joins
        self.inputs_mapping=['acs', {}] + [None]*len(self.aggregations)

    def run(self, acs, left, aux):
        # join all aggregations
        logging.info('Joining aggregations')
        aggregation_results = [a.get_result().drop(left.columns, axis=1) 
                for a in self.aggregation_joins]
        X = left.join(aggregation_results)

        logging.info('Joining ACS')
        # backfill missing acs data
        census_tract_id = acs.census_tract_id # store tracts
        acs = acs.groupby('census_tract_id').apply(
                lambda d: d.sort_values('year', ascending=True)\
                    .fillna(method='backfill'))
        acs['census_tract_id'] = census_tract_id
        data.prefix_columns(acs, 'acs_', ignore=['census_tract_id'])

        # >= 2014, use acs2014, <= 2010 use acs2010
        # TODO use use 2009 after adding 2000 census tract ids!
        X['acs_year'] = X.date.apply(lambda d: 
                min(2014, max(2010, d.year)))
        X = X.merge(acs, how='left', 
                on=['acs_year', 'census_tract_id'])
        X.drop(['acs_year'], axis=1, inplace=True)

        logging.info('Dates')
        X['age'] = (aux.date - aux.date_of_birth)/util.day
        X['date_of_birth_days'] = util.date_to_days(aux.date_of_birth)
        X['date_of_birth_month'] = aux.date_of_birth.dt.month
        X['wic'] = (aux.first_wic_date < aux.date).fillna(False)

        logging.info('Binarizing sets')
        # TODO: faster to just binarize in the wic aggregation
        binarize = {'enroll': ['employment_status', 'occupation', 'assistance', 'language', 'clinic'],
                    'prenatal': ['clinic', 'service'],
                    'birth': ['clinic', 'complication', 'disposition', 'place_type']}
        for table, columns in binarize.iteritems():
            for column in columns:
                c = 'wic%s_kid_all_%s' % (table, column)
                X[c].replace(0, np.nan, inplace=True) # TODO: handle this better in Aggregation fillna
                data.binarize_set(X, c)

        X.set_index(['kid_id', 'address_id', 'date'], inplace=True)
        aux.set_index(['kid_id', 'address_id', 'date'], inplace=True)

        c = data.non_numeric_columns(X)
        if len(c) > 0:
            logging.warning('Non-numeric columns: %s' % c)

        return {'X':X.astype(np.float32), 'aux':aux}
