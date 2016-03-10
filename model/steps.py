from drain import data, step, model
from drain.util import dict_product
import lead.model.data
import lead.model.transform
from lead.output import aggregations
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

def bll6_forest():
    return bll6_models(forest(), {
            'year': range(2011, 2016+1), 
            'train_years': [6], 
            'train_query': [None], 
            'aggregations': aggregations.args,
            'outcome_expr':['address_max_bll >= 6']})

def bll6_forests():
    return bll6_models(forest())

def bll6_aggregations():
    args = aggregations.args
    args_search = [args]
    for name, a in args.iteritems():
        if isinstance(a, dict):
            for space, deltas in a.iteritems():
                for i in range(len(deltas)):
                    copy = dict(args)
                    copy[name] = dict(args[name])
                    copy[name][space] = deltas[:i]
                    args_search.append(copy)
        else:
            for i in range(len(a)):
                copy = dict(args)
                copy[name] = a[:i]
                args_search.append(copy)
    print len(args_search)

    return  bll6_models(forest(), {
            'year': range(2012, 2014),
            'train_years': [6],
            'train_query': [None],
            'aggregations': args_search,
            'outcome_expr':['address_max_bll >= 6']})
                    
def test_forests():
    return test_models(forest())

    return test_models(forest())
def product_forests():
    return product_models(forest())

def train_min_last_sample_age():
    return bll6_models(forest(), dict(train_min_last_sample_age=[None, 0, 365, 365*1.5, 365*2, 365*2.5, 365*3]))

def bll6_models(estimators, transform_search = {}):
    transformd = dict(
        train_years = [4,5,6,7],
        year = range(2010, 2014+1)+[2016],
        spacetime_normalize = [False],
        wic_sample_weight = [0],
        aggregations = aggregations.args,
        train_query = [None],
        outcome_expr = ['max_bll >= 6', 'address_max_bll >= 6']
    )
    transformd.update(transform_search)
    return models(estimators, transformd)

def test_models(estimators, transform_search = {}):
    transformd = dict(
        train_years = [3,4,5,6,7],
        #train_years = [3],
        year = range(2011, 2014+1),
        spacetime_normalize = [False],
        #wic_sample_weight = [0,1],
        wic_sample_weight = [0],
        train_query = ['wic and today_age > 365*2'],
        outcome_expr = ['address_test_max_age > 30*22 or address_max_bll >= 6']
    )
    transformd.update(transform_search)
    return models(estimators, transformd)

def product_models(estimators, transform_search = {}):
    steps = []
    for year in range(2011, 2016+1):
        transform_search['year'] = [year]
        ts = test_models(estimators, transform_search)
        bs = bll6_models(estimators, transform_search)
        for t in ts:
            t.__name__ = 'estimator_t'
            t.get_input('transform').__name__ = 'transform_t'

        for t,b in product(ts, bs):
            
            p = model.PredictProduct(inputs=[t,b], 
                    inputs_mapping=['test', 'bll6'], target=True)
            steps.append(p)
            
    return steps

def models(estimators, transform_search):
    steps = []
    for transform_args, estimator in product(
            dict_product(transform_search), estimators):
    
        transform = lead.model.transform.LeadTransform(
                month=1, day=25,
                name='transform',
                **transform_args)

        y = model.FitPredict(inputs=[estimator, transform], 
                name='y', target=True)
        steps.append(y)

    return steps
