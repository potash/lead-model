from drain.step import Step
from drain import util, data
from drain.data import FromSQL
from lead.output import aggregations

import pandas as pd
import numpy as np
import logging

class LeadData(Step):
    EXCLUDE = {'first_name', 'last_name', 'address_residential', 
               'address'}

    PARSE_DATES = ['date_of_birth', 'first_bll6_sample_date', 
        'first_bll10_sample_date', 'first_sample_date', 
        'last_sample_date', 'address_min_date', 'address_max_date', 
        'address_wic_min_date', 'address_test_min_date', 
        'address_wic_max_date', 'address_test_max_date', 'wic_date']

    AUX = {'address_count', 'test_count', 
        'first_bll6_address_id', 'first_sample_address_id', 
        'max_bll', 'mean_bll'}

    AUX.update(PARSE_DATES)

    def __init__(self, month, day, year_min=2008, **kwargs):
        Step.__init__(self, month=month, day=day, year_min=year_min, **kwargs)

        kid_addresses = FromSQL(query="""
select * from output.kids join output.kid_addresses using (kid_id)
join output.addresses using (address_id)
where date_of_birth >= '{date_min}'
""".format(date_min='%s-%s-%s' % (self.year_min, self.month, self.day)), 
                parse_dates=self.PARSE_DATES, tables=['output.kids', 'output.addresses'], target=True)

        # TODO request aggregations of the right dates
        # remember about date_floor for kids poisoned before 12 mo
        self.aggregations = aggregations.all()
        self.inputs = [kid_addresses] + self.aggregations
        self.input_mapping=['X']

    def run(self, X, *args, **kwargs):
        # Date stuff
        # TODO: include people who are born and poisoned before a date
        # TODO: exclude them from test
        logging.info('dates')
        X['date'] = X.date_of_birth.apply(
                util.date_ceil(self.month, self.day))
        import pdb; pdb.set_trace()
        
        # if bll6 happens before dob.date_ceil() use date_floor instead
        bll6_before_date = X.first_bll6_sample_date < X.date
        X.loc[bll6_before_date, 'date'] =  X.loc[bll6_before_date, 
                'first_bll6_sample_date'].apply(
                    util.date_floor(self.month, self.day))
        X['age'] = (X.date - X.date_of_birth)/util.day
        X['date_of_birth_days'] = X.date_of_birth.apply(util.date_to_days)
        X['date_of_birth_month'] = X.date_of_birth.apply(lambda d: d.month)
        X['wic'] = (X.wic_date < X.date).fillna(False)

        # join before setting index
        for aggregation in self.aggregations:
            logging.info('Joining %s' % aggregation)
            X = aggregation.join(X)

        # Set index
        X.set_index(['kid_id', 'address_id'], inplace=True)

        # Separate aux
        aux = X[list(self.AUX)]
        aux['age'] = X.age
        X = data.select_features(X, exclude=(self.AUX | self.EXCLUDE))

        # Sample dates used for training_min_max_sample_age in LeadTransform
        # TODO: could make this more efficient
        engine = util.create_engine()
        sample_dates = pd.read_sql("""
select kid_id, sample_date, date_of_birth
from output.tests join output.kids using (kid_id)""", engine, parse_dates=['date_of_birth', 'sample_date'])
        
        return {'X':X, 'aux':aux, 'sample_dates':sample_dates}
