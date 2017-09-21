Preventing Childhood Lead Poisoning
====

## Introduction

Lead poisoning is a major public health problem that affects hundreds of thousands of children in the United States every
year. A common approach to identifying lead hazards is to test all children for elevated blood lead levels and then investigate
and remediate the homes of children with elevated tests. This can prevent exposure to lead of future residents,
but only after a child has been irreversibly poisoned. In parternship with the Chicago Department of Public
Health (CDPH),  we have built a model that predicts the risk of a child being poisoned. Our model's risk scores facillitates
an intervention before lead posioning occurs. Using two decades of blood lead level tests, home lead inspections, property value assessments,
and census data, our model allows inspectors to prioritize houses on an intractably long list of potential hazards
and identify children who are at the highest risk. This work has been described by CDPH as pioneering in the use
of machine learning and predictive analytics in public health and has the potential to have a significant impact on both
health and economic outcomes for communities across the US. For a longer overview of the project, see our preliminary results which were written up and
published in the [21st ACM SIGKDD Proceedings](https://github.com/dssg/lead-public/raw/master/kdd.pdf). This project is closely based on previous
[work](https://dssg.uchicago.edu/project/predictive-analytics-to-prevent-lead-poisoning-in-children/) of Joe Brew, Alex Loewi, Subho Majumdar, and Andrew Reece
as part of the 2014 [Data Science for Social Good Summer Fellowship](http://dssg.uchicago.edu).

## Implementation

The code for each phase is located in the corresponding subdirectory and is executed using a drakefile.
The output of each phase is contained in a database schema of the same name. Each folder also has a
corresponding README documenting the steps.

**features**: Generate model features by aggregating the datasets at various spatial and temporal resolutions.

**model**: Use our [drain pipeline](https://github.com/dssg/drain/) to run models in parallel and serialize the results.

**pilot**: SQL script for generating a contact list for the model pilot.

**explore**: Miscellanous scripts for one-off exploration of the data.


## Deployment

### 1. Load and transform the data
Follow the instructions in the `lead-etl` repository to load and transform the data.

### 2. Configure variables:
Copy `./lead/example_profile` to `./lead/default_profile` and set the indicated variables.

Include this repository in your Python path, e.g. by adding this line to your `.bashrc`:
```
export PYTHONPATH=$PYTHONPATH:~/project/lead-model
```

### 3. Install requirements
Install python requirements:
```
pip install -r requirements.txt
```

### 3. Run models using `drain`.
To fit a current model and make predictions change to `./lead` and run:
```
drain execute lead.model.workflows::bll6_forest_today ...
```
Here `lead.model.workflows.bll6_forest_today` is a drain workflow, i.e. a function taking no arguments that returns collection of drain steps.

For temporal cross validation use the `bll6_forest` workflow.

# License

See [LICENSE](https://raw.githubusercontent.com/dssg/public-lead/master/LICENSE)

# Contributors
    - Eric Potash (epotash@uchicago.edu)

# References
 1. Potash, Eric, Joe Brew, Alexander Loewi, Subhabrata Majumdar, Andrew Reece, Joe Walsh, Eric Rozier, Emile Jorgenson, Raed Mansour, and Rayid Ghani. "Predictive modeling for public health: Preventing childhood lead poisoning." In Proceedings of the 21th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, pp. 2039-2047. ACM, 2015.
