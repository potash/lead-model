from drain import data, step, model, data
from drain.util import dict_product, make_list
from drain.step import Call, MapResults, GetItem
from drain.data import ToHDF

from itertools import product
import pandas as pd
import os

import lead.model.data
import lead.model.transform
import lead.model.cv
import lead.model.score
from lead.features import aggregations

from .split import split

def args_list(*args):
    return args

def kid_predictions_past():
    """
    Temporal cross validation workflow
    """
    ps = address_predictions_past()
    w = []
    for p in ps:
        t = p.get_input('transform')
        s = lead.model.score.LeadScore(inputs=[MapResults(
                [p, t], 
                ['scores', {'train':None, 'X':None, 'sample_weight':None}])])
        s.target = True
        w.append(s)

    return w


def kid_predictions_today():
    """
    Predictions for kids today
    """
    p = address_predictions_today()
    t = p.get_input('transform')
    s = lead.model.score.LeadScore(inputs=[MapResults(
            [p, t], 
            ['scores', {'train':None, 'X':None, 'sample_weight':None}])])
    s.target = True
    return s


def address_predictions_today():
    """
    Predictions for all addresses today
    """
    today = pd.Timestamp(os.environ['TODAY'])
    p = bll6_models(
            forest(),
            dict(year=today.year,
                 month=today.month,
                 day=today.day),
            dump_estimator=True)[0]
    return p


def address_predictions_past():
    """
    Predictions for addresses in the past (for cross validation)
    """
    return bll6_models(forest(), dump_estimator=True)
    
def forest(**update_kwargs):
    """
    Returns a step constructing a scikit-learn RandomForestClassifier
    """
    kwargs = dict(
        n_estimators=2000,
        n_jobs=int(os.environ.get('N_JOBS', -1)),
        criterion='entropy',
        class_weight='balanced_bootstrap',
        max_features='sqrt',
        random_state=0)

    kwargs.update(**update_kwargs)

    return step.Call('sklearn.ensemble.RandomForestClassifier', **kwargs)

def bll6_models(estimators, cv_search={}, transform_search={}, dump_estimator=False):
    """
    Provides good defaults for transform_search to models()
    Args:
        estimators: list of estimators as accepted by models()
        transform_search: optional LeadTransform arguments to override the defaults

    """
    cvd = dict(
        year=range(2011, 2014+1),
        month=1,
        day=1,
        train_years=[6],
        train_query=[None],
    )
    cvd.update(cv_search)

    transformd = dict(
        wic_sample_weight=[0],
        aggregations=aggregations.args,
        outcome_expr='max_bll0 >= 6',
        outcome_where_expr='max_bll0 == max_bll0' # this means max_bll0.notnull()
    )
    transformd.update(transform_search)
    return models(make_list(estimators), cvd, transformd, dump_estimator=dump_estimator)

def models(estimators, cv_search, transform_search, dump_estimator):
    """
    Grid search prediction workflows. Used by bll6_models, test_models, and product_models.
    Args:
        estimators: collection of steps, each of which constructs an estimator
        cv_search: dictionary of arguments to LeadCrossValidate to search over
        transform_search: dictionary of arguments to LeadTransform to search over
        dump_estimator: whether to dump the estimator.

    Returns: a list drain.model.Predict steps constructed by taking the product of
        the estimators with the the result of drain.util.dict_product on each of
        cv_search and transform_search.

        Each Predict step contains the following in its inputs graph:
            - lead.model.cv.LeadCrossValidate
            - lead.model.transform.LeadTransform
            - drain.model.Fit
    """
    steps = []
    for cv_args, transform_args, estimator in product(
            dict_product(cv_search), dict_product(transform_search), estimators):

        cv = lead.model.cv.LeadCrossValidate(**cv_args)
        cv.name = 'cv'

        X_train = GetItem(GetItem(cv, 'X'), GetItem(cv, 'train'))
        mean = Call(X_train, 'mean')
        mean.name = 'mean'
        mean.target = True

        X_impute = Call(data.impute,
                        inputs=[MapResults([GetItem(cv, 'X'), mean], 
                                           ['X', 'value'])])

        cv_imputed = MapResults([X_impute, cv], ['X', {'X':None}])
        cv_imputed.target = True

        transform = lead.model.transform.LeadTransform(inputs=[cv_imputed], **transform_args)
        transform.name = 'transform'

        fit = model.Fit(inputs=[estimator, transform], return_estimator=True)
        fit.name = 'fit'
        
        y = model.Predict(inputs=[fit, transform],
                return_feature_importances=True)
        y.name = 'predict'
        
        if dump_estimator:
            fit.target = True
        
        X_test = lead.model.data.LeadData(
            year_min=cv_args['year'],
            year_max=cv_args['year'],
            month=cv_args['month'],
            day=cv_args['day'],
            address=True)

        # there are over 800k addresses
        # to avoid running out of memory, we split into pieces for prediction
        k = 4
        pieces = list(map(str, range(k)))
        X_split = Call(split, inputs=[MapResults([X_test], {'X':'df'})], k=k)
        tohdf = ToHDF(inputs = [MapResults([X_split], [pieces])])
        tohdf.target = True

        ys = []
        for j in pieces:
            X_impute = Call(data.impute,
                            inputs=[MapResults(Call(tohdf, 'get', key=j), 'X'),
                                    MapResults([mean], 'value')]) 

            y = model.Predict(inputs=[fit, MapResults([X_impute], 'X')])
            y.target = True
            ys.append(GetItem(y, 'y'))

        # concatenate the pieces
        y = Call(pd.concat, inputs=[MapResults([Call(args_list, inputs=ys)], 'objs')])
        y.target = True
        steps.append(y)
        
    return steps
