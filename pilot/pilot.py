#!/bin/bash
from drain import explore, model, step, data, util, yaml

from lead.model import steps

import pandas as pd
import logging

step.OUTPUTDIR='/home/epotash/lead/data/drain/'
yaml.configure()

predictions = steps.bll6_forest_today()

logging.info('Loading')
predictions = step.load(predictions)

logging.info('Concatenating')
y = pd.concat((p.get_result()['y'] for p in predictions))

logging.info('Importing')
db = util.create_db()
db.to_sql(y, 'predictions', if_exists='replace')
