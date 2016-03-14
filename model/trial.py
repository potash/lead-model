from drain import explore, model, step, data, util

from lead.model import steps
import lead.model.data
import lead.output.aggregations

import pandas as pd

step.BASEDIR='/home/epotash/lead/data/drain/'
step.configure_yaml()

predictions = steps.bll6_forest()
query = 'address_wic_min_date < date'

s = [p for p in predictions 
        if p.named_arguments[('transform', 'year')] == 2016][0]
s.load()
result = s.get_result()
y = result['y']
y['age'] = (data.index_as_series(y, 'date') - y.date_of_birth) / util.day

d = s.get_input('transform').inputs[0]
d.load()
X = d.get_result()['X']

engine = util.create_engine()
y.query(query)[['score', 'age', 'address', 'first_name' , 'last_name', 'date_of_birth', 'max_bll', 'test_count', 'address_count', 'address_wic_infant']].join(
        X[['inspections_address_1y_inspected','inspections_address_1y_complied']])\
        .to_sql(name='predictions', con=engine, if_exists='replace')
