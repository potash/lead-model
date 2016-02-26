from drain.step import Step
from drain import util, data
from drain.data import FromSQL, Merge
from lead.output import aggregations
from lead.output.kids import KIDS_PARSE_DATES, KID_ADDRESSES_PARSE_DATES

import pandas as pd
import numpy as np
import logging

class LeadData(Step):
    def __init__(self, month, day, year_min=2008, **kwargs):
        Step.__init__(self, month=month, day=day, year_min=year_min, 
                **kwargs)

        kid_addresses = Merge(on='kid_id', inputs=[
                FromSQL(table='output.kid_addresses', 
                    parse_dates=KID_ADDRESSES_PARSE_DATES, target=True), 
                FromSQL(table='output.kids', 
                    parse_dates=KIDS_PARSE_DATES, 
                    to_str=['first_name','last_name'], target=True)])

        addresses = FromSQL(table='output.addresses', target=True)

        acs = FromSQL(table='output.acs', target=True)

        # TODO request aggregations of the right dates
        # remember about date_floor for kids poisoned before 12 mo
        self.aggregations = aggregations.all()
        self.inputs = [kid_addresses, acs, addresses] + self.aggregations
        self.input_mapping=['aux', 'acs', 'addresses']

    def run(self, aux, acs, addresses, *args, **kwargs):
        min_date = util.timestamp(self.year_min, self.month, self.day)
        aux.drop(aux.index[aux.date_of_birth < min_date], inplace=True)
        # Date stuff
        logging.info('dates')
        aux['date'] = aux.date_of_birth.apply(
                util.date_ceil(self.month, self.day))
        
        # if bll6 happens before dob.date_ceil() use date_floor instead
        bll6_before_date = aux.first_bll6_sample_date < aux.date
        aux.loc[bll6_before_date, 'date'] =  aux.loc[bll6_before_date, 
                'first_bll6_sample_date'].apply(
                    util.date_floor(self.month, self.day))

        X = aux[['kid_id', 'date', 'address_id']]
        addresses.drop(['address'], axis=1, inplace=True)
        X = X.merge(addresses, on='address_id')
        aux.drop(aux.index[aux.address_id.isnull()], inplace=True)

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

        X['age'] = (aux.date - aux.date_of_birth)/util.day
        X['date_of_birth_days'] = aux.date_of_birth.apply(util.date_to_days)
        X['date_of_birth_month'] = aux.date_of_birth.apply(lambda d: d.month)
        X['wic'] = (aux.wic_date < aux.date).fillna(False)

        # join before setting index
        for aggregation in self.aggregations:
            logging.info('Joining %s' % aggregation)
            X = aggregation.join(X)

        # Sample dates used for training_min_max_sample_age in LeadTransform
        # TODO: could make this more efficient
        engine = util.create_engine()
        return {'X':X, 'aux':aux}
