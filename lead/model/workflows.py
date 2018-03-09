from drain import data, step, model, data
from drain.util import dict_product, make_list
from drain.step import Call, Construct, MapResults

from itertools import product
import pandas as pd
import os

import lead.model.data
import lead.model.transform
import lead.model.cv
from lead.features import aggregations


def bll6_forest():
    """
    The basic temporal cross-validation workflow
    """
    return bll6_models(forest(), dump_estimator=True)


def bll6_forest_today():
    """
    The workflow used to construct a current model
    Parses the environment variable TODAY using pd.Timestamp to set the date
    """
    today = pd.Timestamp(os.environ['TODAY'])
    p = bll6_models(
            forest(),
            dict(year=today.year,
                 month=today.month,
                 day=today.day),
            dump_estimator=True)[0]
    
    # put the predictions into the database
    tosql = data.ToSQL(table_name='predictions', if_exists='replace',
            inputs=[MapResults([p], mapping=[{'y':'df', 'feature_importances':None}])])
    tosql.target = True
    return tosql

def address_data_past():
    """
    Builds address-level features for the past
    Plus saves fitted models and means for the past
    """
    ys = [] # lead address data
    for y in range(2011,2011+1):
        X = lead.model.data.LeadData(
                year_min=y,
                year_max=y,
                month=1,
                day=1,
                address=True)

        p = bll6_forest()[0]
        mean = p.get_input('mean')
        fit = p.get_input('fit')

        X_impute = Construct(data.impute,
                             inputs=[X, MapResults([mean], 'value')]) 

        y = model.Predict(inputs=[fit, MapResults([X_impute], 'X')])
        y.target = True
        ys.append(y)

    return ys    

def address_data_today():
    """
    Builds address-level features today
    """
    today = pd.Timestamp(os.environ['TODAY'])
    X = lead.model.data.LeadData(
            year_min=today.year,
            year_max=today.year,
            month=today.month,
            day=today.day,
            address=True)

    p = bll6_forest_today()
    mean = p.get_input('mean')
    fit = p.get_input('fit')

    X_impute = Construct(data.impute,
                         inputs=[X, MapResults([mean], 'value')]) 

    y = model.Predict(inputs=[fit, MapResults([X_impute], 'X')])
    y.target = True

    return y
    
def forest(**update_kwargs):
    """
    Returns a step constructing a scikit-learn RandomForestClassifier
    """
    kwargs = dict(
        _class='sklearn.ensemble.RandomForestClassifier',
        n_estimators=2000,
        n_jobs=int(os.environ.get('N_JOBS', -1)),
        criterion='entropy',
        class_weight='balanced_bootstrap',
        max_features='sqrt',
        random_state=0)

    kwargs.update(**update_kwargs)

    return step.Construct(**kwargs)

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
        dump_estimator: whether to dump the estimator (and the mean).
            Necessary for re-using the model for more scoring later.

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

        X_train = Call('__getitem__', inputs=[MapResults([cv], {'X':'obj', 'train':'key',
                                                       'test':None, 'aux':None})])
        mean = Call('mean', inputs=[X_train])
        mean.name = 'mean'

        X_impute = Construct(data.impute,
                             inputs=[MapResults([cv], {'aux':None, 'test':None, 'train':None}),
                              MapResults([mean], 'value')])

        cv_imputed = MapResults([X_impute, cv], ['X', {'X':None}])
        cv_imputed.target = True

        transform = lead.model.transform.LeadTransform(inputs=[cv_imputed], **transform_args)
        transform.name = 'transform'

        fit = model.Fit(inputs=[estimator, transform], return_estimator=True)
        fit.name = 'fit'
        
        y = model.Predict(inputs=[fit, transform],
                return_feature_importances=True)
        y.name = 'predict'
        y.target = True

        if dump_estimator:
            mean.target = True
            fit.target = True

        steps.append(y)

    return steps
