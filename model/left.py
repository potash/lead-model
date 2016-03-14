from drain.step import Step
from drain import util, data
from drain.data import FromSQL, Merge
from lead.output.kids import KIDS_PARSE_DATES, KID_ADDRESSES_PARSE_DATES

import pandas as pd
import numpy as np
import logging

class LeadLeft(Step):
    def __init__(self, month, day, year_min, **kwargs):
        Step.__init__(self, month=month, day=day, year_min=year_min, 
                **kwargs)

        kid_addresses = Merge(on='kid_id', inputs=[
                FromSQL(table='output.kid_addresses', 
                    parse_dates=KID_ADDRESSES_PARSE_DATES, target=True), 
                FromSQL(table='output.kids', 
                    parse_dates=KIDS_PARSE_DATES, 
                    to_str=['first_name','last_name'], target=True)])

        addresses = FromSQL(table='output.addresses', target=True)

        self.inputs = [kid_addresses, addresses]

    def run(self, aux, addresses):
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

        columns = aux.columns
        aux = aux.merge(addresses, on='address_id')

        left_columns = ['kid_id', 'date'] + list(addresses.columns)
        left_columns.remove('address')
        left = aux[left_columns]

        return {'left':left, 'aux':aux}
