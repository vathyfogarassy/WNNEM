# WNNEM
Weighted Nearest Neighbours Control Group Selection with Error Minimization.

The WNNEM method is a multivariate weighted nearest neighbours-based control group selection method, which aims to eliminate of disadvantages of propensity score matching method in cases of few descriptive covariates. The WNNEM method performs better when individuals are characterized by fewer covariates, and the search space is such small that there exist many individuals for selecting as control which are the most similar pairs of more than one case (treated) subject.

WNNEM method pairs the elements of the case and control groups in the original vector space of the covariates and the dissimilarities of the individuals are calculated as the weighted distances of the subjects. The weight factors are derived from a logistic regression model fitted on the status of treatment assignment.

The WNNEM method is published in ...

In case of using the datasets or applying the WNNEM method, please cite the article above.


# Structure of the folder
- **Datasets**: 
  This folder contains the datasets for three different scenarios for control group selection. In each scenario 100 datasets are available.
  - *Scenario I*: Each dataset conains 1.000 indivuduals. Individuals are characterized by 10 binary variables (<img src="https://render.githubusercontent.com/render/math?math=x_1, x_2, ..., x_{10}">), each from a Bernoulli distribution (<img src="https://render.githubusercontent.com/render/math?math=x_j\sim\textrm{B}(0.5), \quad j=1,\dots,10">).
  - *Scenario II*: The second scenario models such studies in which fewer descriptive variables are available. In this scenario, each individual is characterized by 1 ordinal and 5 binary variables (<img src="https://render.githubusercontent.com/render/math?math=x_1, x_2, ..., x_6">). The ordinal variable (<img src="https://render.githubusercontent.com/render/math?math=x_1">) represents, for example, 5 age groups, while the binary variables (<img src="https://render.githubusercontent.com/render/math?math=x_2, x_3, ..., x_6">) may represent, for example, the gender of the subject or various diagnoses.  In this scenario, 700 individuals are simulated in each dataset, and the ratio of the candidate subjects to the treated individuals (case group) in the 100 datasets is between 2.0 and 3.1.
  - *Scenario III*: The third scenario is similar to the second one regarding the attributes of individuals and the total number of subjects in each dataset. However, it simulates a more difficult control group selection problem. Although each dataset still contained 700 individuals, the number of treated individuals (case group) in the case of the third scenario is higher than in the second one. While in scenario II, the size of the treated group varies between 24.5 and 33.0 percent of the dataset, in the case of the third scenario, it is between 31.7 and 40.7 percent (the ratio of the candidate subjects to the treated individuals is between 1.5 and 2.2).
- **Results**:
  Contains the result files for Scenario I, II and III.
  Each output contains the following attributes:
  - *_id*: index of the individual
  - *x1, ..., xn*: attributes (covariates) of the individulas
  - *treated*: this attributevalue yields if individual is the member of the case group (treated=1) or it is an individual which can be selected into the control group ( treated=0)
  - *ps*: propensity score value of the individual
  - *ss_pair*: index (\_id) of the matched pair resulted by the stratified matching method  
  - *wnnem_pair*: index (\_id) of the matched pair resulted by the WNNEM method
  - *nn_euk_pair_numberoftrial*: index (\_id) of the matched pair resulted by the nearest neighbour method based on the Euclidean distances of the individuals
  - *nn_mah_pair_numberoftrial*: index (\_id) of the matched pair resulted by the Mahalanobis matching (nearest neighbour matching based on Mahalanobis distances) 
  - *nn_psm_02_pair_numberoftrial*: index (\_id) of the matched pair resulted by the greedy 1:1 PSM (propensity score matching) method applying caliper size set as 0.2 of the standard deviation of the logit of the propensity scores.
  - *nn_psm_dyn_numberoftrial*: index (\_id) of the matched pair resulted by the greedy 1:1 PSM (propensity score matching) method applying caliper size determined dynamically and set at the minimal value for which 1:1 matching can be performed.
- **Scripts**: 
  Contains the implemented python modules.
- **example.py**:
  An example code on the usage of the uploaded scripts.
