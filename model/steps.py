from drain import data, step, model
from drain.util import dict_product
import lead.model.data
import lead.model.transform
from itertools import product

metrics = [
    {'metric':'baseline'},
    {'metric':'count'},
    {'metric':'precision', 'k':100},
    {'metric':'precision', 'k':200},
    {'metric':'precision', 'k':500},
    {'metric':'auc'},
]

def forest():
    return [step.Construct('sklearn.ensemble.RandomForestClassifier', n_estimators=500, n_jobs=-1, criterion='entropy', balanced=True, max_features='sqrt')]

def model_svm():
    return mdoels(model.svms())

def model_forests():
    return models(model.forests(n_estimators=[500], balanced=[True]))

def model_logits():
    return models(model.logits())

def model_data():
    d = lead.model.data.LeadData(month=1, day=1, year_min=2007, target=True)
    return [d]

def model_forest():
    return models(forest())

def train_min_last_sample_age():
    return models(forest(), dict(train_min_last_sample_age=[None, 0, 365, 365*1.5, 365*2, 365*2.5, 365*3]))

def models(estimators, transform_search = {}):
    steps = []
    transformd = dict(
        train_years = [3],
        year = range(2011, 2013+1),
        spacetime_normalize = [False],
        wic_sample_weight = [1],
        train_min_last_sample_age=[365*2],
        train_non_wic = [True],
    )
    transformd.update(transform_search)

    for transform_args, estimator in product(
            dict_product(transformd), estimators):
    
        transform = lead.model.transform.LeadTransform(
                month=1, day=1, inputs=model_data(), 
                name='transform',
                **transform_args)

        y = model.FitPredict(inputs=[estimator, transform], 
                name='y', target=True)
        m = model.PrintMetrics(metrics, inputs=[y], target=True)
        steps.append(m)

    return steps
