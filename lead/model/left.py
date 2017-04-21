from drain.step import Step
from drain import util, data
from drain.data import FromSQL, Merge

import pandas as pd
import numpy as np
import logging

kid_addresses = FromSQL(table='output.kid_addresses')
kid_addresses.target = True

kids = FromSQL(table='output.kids', to_str=['first_name','last_name'])
kids.target = True

addresses = FromSQL(table='output.addresses')
addresses.target = True

class LeadLeft(Step):
    def __init__(self, month, day, year_min):
        Step.__init__(self, month=month, day=day, year_min=year_min)

        aux = Merge(on='kid_id', inputs=[kid_addresses, kids])
        self.inputs = [aux, addresses]

    def run(self, aux, addresses):
        min_date = util.timestamp(self.year_min, self.month, self.day)
        aux.dropna(subset=['date_of_birth'], inplace=True)
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
