from drain.step import Step
from drain.data import FromSQL, prefix_columns


class ACS(Step):
    def __init__(self, inputs):
        """
        Args:
            inputs: array containing a LeadLeft instance
        """
        acs = FromSQL(table='output.acs')
        acs.target = True
        inputs = inputs + [acs]

        Step.__init__(self, inputs=inputs)
        self.inputs_mapping = [{'aux':None}, 'acs']

    def run(self, left, acs):
        acs = acs.groupby('census_tract_id').apply(
                lambda d: d.sort_values('year', ascending=True)
                .fillna(method='backfill'))

        prefix_columns(acs, 'acs_', ignore=['census_tract_id'])
        # Assume ACS y is released on 1/1/y+2
        # >= 2017, use acs2015, <= 2012 use acs2010
        # TODO: use use 2009 after adding 2000 census tract ids!
        left['acs_year'] = left.date.dt.year.apply(
                lambda y: min(2015, max(2010, y-2)))
        merged = left.merge(acs, how='left',
                            on=['acs_year', 'census_tract_id'])
        merged = merged.drop(list(left.columns), axis=1)

        return merged
