from drain import data, step, model
from drain.util import dict_product
import lead.model.data
import lead.model.transform
from itertools import product

wic_query = 'address_wic_min_date < date'
metrics = [
    {'metric':'baseline', 'query':wic_query},
    {'metric':'count', 'query':wic_query},
    {'metric':'precision', 'k':100, 'query':wic_query},
    {'metric':'precision', 'k':200, 'query':wic_query},
    {'metric':'precision', 'k':500, 'query':wic_query},
    {'metric':'auc', 'query':wic_query},
]

def forest():
    return [step.Construct('sklearn.ensemble.RandomForestClassifier', n_estimators=500, n_jobs=-1, criterion='entropy', balanced=True, max_features='sqrt')]

def model_svm():
    return mdoels(model.svms())

def model_forests():
    return bll6_models(model.forests(n_estimators=[500], balanced=[True]))

def model_logits():
    return bll6_models(model.logits())

def model_data():
    d = lead.model.data.LeadData(month=1, day=1, year_min=2007, target=True)
    return [d]

def bll6_forest():
    return bll6_models(forest())

def test_forest():
    return test_models(forest())

def train_min_last_sample_age():
    return bll6_models(forest(), dict(train_min_last_sample_age=[None, 0, 365, 365*1.5, 365*2, 365*2.5, 365*3]))

def bll6_models(estimators, transform_search = {}):
    transformd = dict(
        train_years = [1,2,3,4,5],
        year = range(2011, 2013+1),
        spacetime_normalize = [False],
        wic_sample_weight = [0],
        train_query = [None, 'last_sample_age > 365*2'],
        outcome_expr = ['max_bll >= 6']
    )
    transformd.update(transform_search)
    return models(estimators, transformd)

def test_models(estimators, transform_search = {}):
    transformd = dict(
        train_years = [3,4,5],
        year = range(2011, 2013+1),
        spacetime_normalize = [False],
        wic_sample_weight = [0,1],
        train_query = ['wic and today_age > 365*2', 'today_age > 365*2'],
        outcome_expr = ['address_test_max_age > 30*22 or address_max_bll >= 6']
    )
    transformd.update(transform_search)
    return models(estimators, transformd)

def models(estimators, transform_search):
    steps = []
    md = model_data()
    for transform_args, estimator in product(
            dict_product(transform_search), estimators):
    
        transform = lead.model.transform.LeadTransform(
                month=1, day=1, inputs=md, 
                name='transform',
                **transform_args)

        y = model.FitPredict(inputs=[estimator, transform], 
                name='y', target=True)
        m = model.PrintMetrics(metrics, inputs=[y], target=True)
        steps.append(m)

    return steps
