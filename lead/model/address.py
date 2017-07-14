from drain.step import Step
from drain.util import timestamp, cross_join
from drain.data import FromSQL, Merge

import pandas as pd
import numpy as np
import logging

addresses = FromSQL(table='output.addresses')
addresses.target = True

class LeadAddressLeft(Step):
    """
    This Step simply adds dates to all addresses in the database. It is used
    by LeadData for building an address dataset.
    """
    def __init__(self, month, day, year_min, year_max):
        """
        Args:
            month: the month to use
            day: the day of the month to use
            year_min: the year to start
            year_max: the year to end
        """
        Step.__init__(self, month=month, day=day, year_min=year_min, year_max=year_max, inputs=[addresses])

    def run(self, addresses):
        """
        Returns:
            - left: the cross product of the output.addresses table with the
                specified dates.
        """
        dates = [timestamp(year, self.month, self.day)
                 for year in range(self.year_min, self.year_max+1)]
        if len(dates) == 1:
            # when there's exactly one date modify in place for efficiency
            addresses['date'] = dates[0]
            left = addresses
        else:
            left = cross_join(addresses, pd.DataFrame(dates))
            
        return {'left':left}
