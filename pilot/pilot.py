import pandas as pd
from drain import util

engine = util.create_engine()
pilot = pd.read_sql('select * from pilot.pilot03', engine)
contact = pd.read_sql('select * from pilot.pilot03_contact', engine)

# unstack rank
contact = contact.set_index(['kid_id', 'address_id', 'rank']).unstack('rank')
# swap levels and sort by rank
contact = contact.swaplevel(axis=1).sort_index(axis=1, level=0)
# rename (1, addr_ln_1) to 1_addr_ln_1, etc.
contact.columns = ['%s_%s' % c for c in contact.columns.values]

pilot = pilot.merge(contact.reset_index(), on=['kid_id', 'address_id'])

# make apartments array str for cleaner printing
pilot['investigation_apts'] = pilot.investigation_apts.dropna().apply(lambda a: map(str, a))
# format floats (phone numbers) as integers
print pilot.to_csv(index=False, float_format='%d')
