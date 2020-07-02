# WNNEM
Weighted Nearest Neighbours Control Group Selection with Error Minimization.

The WNNEM method is a multivariate weighted nearest neighbours-based control group selection method, which aims to eliminate of disadvantages of propensity score matching method in cases of few descriptive covariates. The WNNEM method performs better when individuals are characterized by fewer covariates, and the search space is such small that there exist many individuals for selecting as control which are the most similar pairs of more than one case (treated) subject.

WNNEM method pairs the elements of the case and control groups in the original vector space of the covariates and the dissimilarities of the individuals are calculated as the weighted distances of the subjects. The weight factors are derived from a logistic regression model fitted on the status of treatment assignment.

The WNNEM method is published in ...
In case of using the datasets or applying the WNNEM method, please cite the article above.


# Structure of the folder
- Datasets: 
  This folder contains the datasets for three different scenarios for control group selection:
  -- Scenario I: Each dataset conains 1.000 indivuduals. Individuals are characterized by 10 binary variables (x_1, x_2, ..., x_{10}), each from a Bernoulli distribution (x_j∼B(0.5),  j= 1,...,10).
- Results: 
  Contains the result files for Scenario I, II and III.
  A pair column is defined for each run of each method (pair_method_run).
- Scripts: 
  Contains the implemented python modules.
- example.py: 
  An example code on the usage of the uploaded scripts.
