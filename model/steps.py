from drain import data, step, model
import lead.model.data
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
    d = lead.model.data.LeadData(month=1, day=1, year_min=2005, target=True)
    return [d]

def models():
    steps = []
    for train_years, year, wic_sample_weight in product(
            [2,3], range(2011, 2013+1), [1]):
        transform = lead.model.data.LeadTransform(
                month=1, day=1, year=year, 
                train_years=train_years, 
                wic_sample_weight=wic_sample_weight,
                inputs=model_data(), name='transform')

        estimator = step.Construct(
                'sklearn.ensemble.RandomForestClassifier',
                n_estimators=1000, criterion='entropy', 
                n_jobs=-1, name='estimator', balanced=True)

        y = model.FitPredict(inputs=[estimator, transform], 
                name='y', target=True)
        m = model.PrintMetrics(metrics, inputs=[y], target=True)
        steps.append(m)

    return steps
