from drain.step import Step
from drain import util, data
from drain.data import FromSQL, Merge
from drain.aggregation import AggregationJoin

from lead.output import aggregations
from lead.output.kids import KIDS_PARSE_DATES, KID_ADDRESSES_PARSE_DATES
from lead.model.left import LeadLeft

from datetime import date
import pandas as pd
import numpy as np
import logging

class LeadData(Step):
    def __init__(self, month, day, year_min, year_max, **kwargs):
        Step.__init__(self, month=month, day=day, year_min=year_min, year_max=year_max,
                **kwargs)

        acs = FromSQL(table='output.acs', target=True)
        left = LeadLeft(month=month, day=day, year_min=year_min, target=True)

        dates = tuple((date(y, month, day) for y in range(year_min, year_max+1)))
        self.aggregations = aggregations.all_dict(dates)
        self.aggregation_joins = [AggregationJoin(target=True, inputs=[left, a], 
                inputs_mapping=[{'aux':None}, None]) for a in self.aggregations.values()]

        self.inputs = [acs, left] + self.aggregation_joins
        self.inputs_mapping=['acs', {}] + [None]*len(self.aggregations)

    def run(self, acs, left, aux):
        X = left

        # join all aggregations
        for aj in self.aggregation_joins:
            logging.info('Joining %s' % aj.inputs[1].__class__.__name__)
            a = aj.get_result()
            a = a[a.columns.difference(left.columns)]
            X = X.join(a)

        logging.info('Joining ACS')
        # backfill missing acs data
        census_tract_id = acs.census_tract_id # store tracts
        acs = acs.groupby('census_tract_id').transform(
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
        X['date_of_birth_days'] = aux.date_of_birth.apply(util.date_to_days)
        X['date_of_birth_month'] = aux.date_of_birth.apply(lambda d: d.month)
        X['wic'] = (aux.wic_date < aux.date).fillna(False)

        logging.info('Binarizing sets')
        binarize = {'enroll': ['employment_status', 'occupation', 'assistance', 'language', 'clinic'],
                    'prenatal': ['clinic', 'service'],
                    'birth': ['clinic', 'complication', 'disposition', 'place_type']}
        for table, columns in binarize.iteritems():
            for column in columns:
                c = 'wic_%s_kid_all_%s' % (table, column)
                X[c].replace(0, np.nan, inplace=True) # TODO: handle this better in Aggregation fillna
                data.binarize_set(X, c)

        X.set_index(['kid_id', 'address_id', 'date'], inplace=True)
        aux.set_index(['kid_id', 'address_id', 'date'], inplace=True)

        c = data.non_numeric_columns(X)
        if len(c) > 0:
            logging.warning('Non-numeric columns: %s' % c)

        return {'X':X.astype(np.float32), 'aux':aux}
