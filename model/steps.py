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

def model_data():
    d = lead.model.data.LeadData(month=1, day=1, year_min=2007, target=True)
    return [d]

def models():
    steps = []
    transform_search = dict(
        train_years = [3],
        year = range(2011, 2013+1),
        spacetime_normalize = [False],
        wic_sample_weight = [1],
        train_non_wic = [True],
    )
    estimator_search = dict(
        n_estimators=[1000],
    )

    for transform_args, estimator_args in product(
            dict_product(transform_search), 
            dict_product(estimator_search)):
        transform = lead.model.transform.LeadTransform(
                month=1, day=1, inputs=model_data(), 
                name='transform',
                **transform_args)

        estimator = step.Construct(
                'sklearn.ensemble.RandomForestClassifier',
                criterion='entropy', n_jobs=-1, 
                name='estimator', balanced=True,
                **estimator_args)

        y = model.FitPredict(inputs=[estimator, transform], 
                name='y', target=True)
        m = model.PrintMetrics(metrics, inputs=[y], target=True)
        steps.append(m)

    return steps
