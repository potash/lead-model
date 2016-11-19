from drain import data, step, model, util
from drain.util import dict_product
import lead.model.data
import lead.model.transform
from lead.output import aggregations
from itertools import product

def forest():
    return [step.Construct('sklearn.ensemble.RandomForestClassifier', n_estimators=1000, n_jobs=-1, criterion='entropy', balanced=True, max_features='sqrt', random_state=0)]

def model_svm():
    return mdoels(model.svms())

def model_forests():
    return bll6_models(model.forests(n_estimators=[800], balanced=[True], random_state=0))

def model_logits():
    return bll6_models(model.logits())

# annual, quarterly, and monthly random forest models
def bll6_forest_lag6m():
    return bll6_models(forest(), {'month':[1,4,7,10], 'wic_lag':'6m'})

def bll6_forest():
    return bll6_models(forest())

def bll6_forest_less_tract():
    args = dict(aggregations.args)
    for k in args:
        if k in ('tests', 'inspections', 'events', 'permits', 'kids'):
            args[k] = dict(**args[k])
            args[k]['block'] = ['3y']
            args[k]['tract'] = ['1y']
    return bll6_models(forest(), {'aggregations':args})

def bll6_forest_today():
    p = bll6_models(forest(), {'year':2016})[0]
    # save the model
    p.named_steps['fit']._target = True

    # put the predictions into the database
    return data.ToSQL(table_name='predictions', if_exists='replace', 
            inputs=[p], 
            inputs_mapping=[{'y':'df', 'feature_importances':None}, 'db'], 
            target=True)

def bll6_forest_quarterly():
    return bll6_models(forest(), 
        {'month':[1,4,7,10], 'year':range(2010,2014+1)})

def bll6_forest_monthly():
    return bll6_models(forest(), 
        {'month':range(1,13), 'year':range(2010,2014+1)})

def bll6_forest_no_complex():
    args = dict(aggregations.args)
    for name, indexes in args.iteritems():
        if isinstance(indexes, dict):
            args[name] = {k:v for k,v in indexes.iteritems() if k != 'complex'}

    return bll6_models(forest(), {
            'aggregations': args,
    })

def bll6_forest_no_tract():
    args = dict(aggregations.args)
    args['kids'] = util.dict_subset(args['kids'], ['address', 'complex', 'block'])
    args['tests'] = util.dict_subset(args['tests'], ['address', 'complex', 'block'])

    return bll6_models(forest(), {
            'aggregations': args,
    })

def bll6_complex():
    args = dict(aggregations.args)
    args['kids'] = {'kid':['all'], 'complex':['1y']}
    args['tests'] = {'kid':['all'], 'complex':['1y']}

    return bll6_models(forest(), {
            'aggregations': args,
    })

def bll6_kids_complex_1y():
    args = dict(aggregations.args)
    for k in args:
        if isinstance(args[k], dict):
            args[k] = {}
    args['kids'] = {'kid':['all'], 'complex':['1y']}

    return bll6_models(forest(), {'aggregations': args})

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
    print(len(args_search))

    return  bll6_models(forest(), {
            'year': range(2012, 2014),
            'aggregations': args_search,
    })
                    
def test_forests():
    return test_models(forest())

    return test_models(forest())
def product_forests():
    return product_models(forest())

def train_min_last_sample_age():
    return bll6_models(forest(), dict(train_min_last_sample_age=[None, 0, 365, 365*1.5, 365*2, 365*2.5, 365*3]))

def bll6_models(estimators, transform_search = {}):
    transformd = dict(
        month = 9,
        day = 9,
        train_years = [6],
        year = range(2010, 2014+1),
        spacetime_normalize = [False],
        wic_sample_weight = [0],
        aggregations = aggregations.args,
        train_query = [None],
        outcome_expr = ['address_max_bll >= 6']
    )
    transformd.update(transform_search)
    return models(estimators, transformd)

def test_models(estimators, transform_search = {}):
    transformd = dict(
        month = 1,
        day = 25,
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
    for year in range(2011, 2014+1):
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
                name='transform', **transform_args)

        fit = model.Fit(inputs=[estimator, transform], 
                name='fit', target=False, return_estimator=True)

        y = model.Predict(inputs=[fit, transform], 
                name='predict', 
                return_feature_importances=True, 
                target=True)

        steps.append(y)

    return steps
