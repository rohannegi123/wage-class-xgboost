# Activity Recognition system based on Multisensor data fusion (AReM) 
# The model will be used in predicting wether a person makes over50K per year or not from classic adult dataset using XGBoost.

* Dataset Information
Extraction was done by Barry Becker from the 1994 Census
database. A set of reasonably clean records was extracted using
the
following conditions: ((AAGE&gt;16) &amp;&amp; (AGI&gt;100) &amp;&amp;
(AFNLWGT&gt;1)&amp;&amp; (HRSWK&gt;0))
Attribute Information:
Listing of attributes: &gt;50K, &lt;=50K.
age: continuous.
workclass: Private, Self-emp-not-inc, Self-emp-inc, Federal-gov,
Local-gov, State-gov, Without-pay, Never-worked.
fnlwgt: continuous.
education: Bachelors, Some-college, 11th, HS-grad, Prof-school,
Assoc-acdm, Assoc-voc, 9th, 7th-8th, 12th, Masters, 1st-4th, 10th,
Doctorate, 5th-6th, Preschool.
education-num: continuous.
marital-status: Married-civ-spouse, Divorced, Never-married,
Separated, Widowed, Married-spouse-absent, Married-AF-spouse.
occupation: Tech-support, Craft-repair, Other-service, Sales, Exec-
managerial, Prof-specialty, Handlers-cleaners, Machine-op-inspct,
Adm-clerical, Farming-fishing, Transport-moving, Priv-house-serv,
Protective-serv, Armed-Forces.
relationship: Wife, Own-child, Husband, Not-in-family, Other-
relative,
Unmarried.
race: White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other, Black.
sex: Female, Male.
capital-gain: continuous.
capital-loss: continuous.
hours-per-week: continuous.
native-country: United-States, Cambodia, England, Puerto-Rico,
Canada, Germany, Outlying-US(Guam-USVI-etc), India, Japan,
Greece, South, China, Cuba, Iran, Honduras, Philippines, Italy,
Poland, Jamaica, Vietnam, Mexico, Portugal, Ireland, France,
Dominican-Republic, Laos, Ecuador, Taiwan, Haiti, Columbia,
Hungary, Guatemala, Nicaragua, Scotland, Thailand, Yugoslavia,
El-
Salvador, Trinadad&amp;Tobago, Peru, Hong, Holand-Netherlands.

# Packages used
Numpy
Pandas
Pandas_profiling
Matlab
sklearn
xgboost



# model deployed on aws server 
https://wage-class-pred.herokuapp.com/