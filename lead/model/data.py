from drain.step import Step, Call, MapResults
from drain import util, data
from drain.data import FromSQL
from drain.aggregation import SpacetimeAggregationJoin

from lead.features import aggregations
from lead.features.acs import ACS
from lead.model.left import LeadLeft
from lead.model.address import LeadAddressLeft

from datetime import date
import pandas as pd
import numpy as np
import logging

class LeadData(Step):
    """
    This Step builds the dataset for modeling. There are two main kinds of datasets, controlled by the address argument below.
    
    Kid datasets are used for training and for scoring kids. They contain one
    row per child-address.
    
    Address datasets contain one row per address. They are built primarily to
    be able to later quickly access the features for scoring.
    """
    def __init__(self, month, day, year_min, year_max, wic_lag=None, dtype=None, address=False, left=None, index_columns=None):
        """
        Args:
            month: the month for feature generation
            day: the day of the month for feature generation
            year_min: the year to start generating features
            year_max: the year to stop generating features
            wic_lag: a lag for the WIC aggregations, parsed by
                drain.data.parse_delta, e.g. '6m' is a six month lag.
                Default is to None, which is no lag.
            dtype: the dtype to use for features. Defaults to np.float16 for memory efficiency.
            address: whether to build an address dataset. Defaults to False,
                which builds a kid dataset.
            left: optional Left step. if None, defaults to LeadLeft (when address=False) or LeadAddressLeft (when address=True)
            index_columns: columns of left to use as index
        """
        if dtype is None:
            dtype = np.float16

        Step.__init__(self, month=month, day=day, 
                year_min=year_min, year_max=year_max,
                wic_lag=wic_lag, dtype=dtype, address=address,
                index_columns=index_columns)

        if left is None:
            if address:
                left = LeadAddressLeft(month=month, day=day, year_min=year_min, year_max=year_max)
                # left_only is left without aux
                # in the address case it's the same as left
                left_only = left
                self.index_columns = ['address', 'census_block_id', 'ward_id', 'community_area_id', 'date']
            else:
                left = LeadLeft(month=month, day=day, year_min=year_min)
                left.target = True
                left_only = MapResults([left], {'aux':None})
                self.index_columns = ['kid_id', 'address_id', 'date']
        else:
            left_only = MapResults([left], {'aux':None})

        acs = Call(ACS(inputs=[left_only]),  "astype", dtype=dtype)
        acs.target = True

        dates = tuple((date(y, month, day) for y in range(year_min, year_max+1)))
        self.aggregations = aggregations.all_dict(dates, wic_lag)

        self.aggregation_joins = []
        for name, a in self.aggregations.items():
            aj = SpacetimeAggregationJoin(
                    inputs=[a, left_only],
                    lag=wic_lag if name.startswith('wic') else None)
            aj = Call(aj, "astype", dtype=dtype)
            aj.target = True
            self.aggregation_joins.append(aj)

        self.inputs = [MapResults([acs, left] + self.aggregation_joins,
                                 ['acs', {}] + [None]*len(self.aggregations))]

    def run(self, acs, left, aux=None):
        """
        Returns:
            - X: the feature matrix, containing all aggregation features, as
                well as ACS features, and a handful of simple features like
                age and sex.
            - aux: auxillary features used for selecting a training set, setting
                sample weights, and evaluation.
        """
        if self.address:
            left_columns = ['address_lat', 'address_lng']
        else:
            left_columns = ['ward_id', 'community_area_id', 'address_lat', 'address_lng']

        left = left[self.index_columns + left_columns]

        logging.info('Binarizing community area and ward')
        left = data.binarize(left, 
                             {'community_area_id':range(1,78), 'ward_id':range(1,51)},
                             astype=self.dtype, drop=(not self.address))

        logging.info('Joining aggregations')
        X = left.join([a.result for a in self.aggregation_joins] + [acs])
        # delete all aggregation inputs so that memory can be freed
        for a in self.aggregation_joins: del a.result

        X.set_index(self.index_columns, inplace=True)

        c = data.non_numeric_columns(X)
        if len(c) > 0:
            logging.warning('Non-numeric columns: %s' % c)

        if self.address:
            return {'X':X}
        else:
            aux.set_index(self.index_columns, inplace=True)
            return {'X':X, 'aux':aux}
