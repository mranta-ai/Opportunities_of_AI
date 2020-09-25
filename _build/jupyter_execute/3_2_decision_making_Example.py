## Example of decisiong making - gradient boosting and a collection of interpretation metrics

import logging, sys
logging.disable(sys.maxsize)
import warnings
warnings.filterwarnings('ignore')

In this example, we will train an Xgboost-model to company data that has a collection of financial figures. Then, we will use a set of interpretation metrics to analyse our results.

Let's start by loading our data. For that, we need the pandas library that has a convenient function to read csv-files.

import pandas as pd

**index_col=0** defines the location of an index column. This csv-file is not available anywhere. If you want to repeat the analysis, create a csv-file that has companies as rows and different financial figures as columns.

master_df = pd.read_csv('FINAL_FIGURES_PANEL.csv',index_col=0)

Our example data is not the best one. It has many missing values and even some inf-values. To make further analysis easier, I set pandas to consider inf-values as nan-values.

pd.options.mode.use_inf_as_na = True

We build a model where we try to predict Tobin's Q using other financial figures from companies. Therefore we should remove those instances that do not have a Tobin's Q value. With **loc** we can locate instances, in this case, those instances that have a missing value (**isna()** tells that) in the Tobin's Q variable.
Because we removed some instances, the index is now all messed up. With **reset_index()** we can set it back to a sequential row of numbers.

master_df = master_df.loc[~master_df['TobinQ'].isna()]
master_df.reset_index(inplace=True,drop=True)

Below, we apply winsorisation to the data. An explanation from Wikipedia: "Winsorisation is the transformation of statistics to set all outliers to a specified percentile of the data; for example, a 90% winsorisation would see all data below the 5th percentile set to the 5th percentile, and data above the 95th percentile set to the 95th percentile.

master_df['TobinQ'].clip(lower=master_df['TobinQ'].quantile(0.05), upper=master_df['TobinQ'].quantile(0.95),inplace = True)

With pandas, we can quickly draw a histogram of a variable. Here is Tobin's Q. The higher frequency of values around twelve is caused by winsorisation.

master_df['TobinQ'].hist()

Our dataframe has many variables. However, there are very similar variables, and we use only part of them to get meaningful results.

master_df.columns

As predictors, we pick the following variables. There are still highly correlated variables. One good aspect of tree-based boosting methods is that multicollinearity is much less of an issue.

features = ['Year', 'DIVIDEND YIELD - CLOSE',
       'NET SALES/REVENUES -1YR ANN GR', 'NET INCOME - 1 YR ANNUAL GROWT',
       'RETURN ON EQUITY - TOTAL (%)', 'RETURN ON ASSETS',
       'RETURN ON INVESTED CAPITAL', 'SELLING, GENERAL & ADM / SALES',
       'RESEARCH & DEVELOPMENT/SALES', 'OPERATING PROFIT MARGIN',
       'TOTAL DEBT % TOTAL CAPITAL/STD', 'QUICK RATIO', 'CURRENT RATIO',
       'TOTAL INVESTMENT RETURN',
       'PRICE VOLATILITY', 'FOREIGN ASSETS % TOTAL ASSETS',
       'FOREIGN SALES % TOTAL SALES', 'FOREIGN INCOME % TOTAL INCOME',
       'FOREIGN RETURN ON ASSETS', 'FOREIGN INCOME MARGIN',
       'ACCOUNTS PAYABLE/SALES', 'CASH FLOW/SALES', 'COST OF GOODS SOLD/SALES']

We temporarily move the predictor variables to another dataframe for winsorisation.

features_df = master_df[features]

features_df.clip(lower=features_df.quantile(0.05), upper=features_df.quantile(0.95), axis = 1,inplace = True)

We move back the winsorised predictor variables to master_df.

master_df[features] = features_df

With the pandas function **describe()**, we can easily calculate basic statistics for the features.

master_df[features].describe().transpose()

Tobin's Q to the **y_df** dataframe.

y_df = master_df['TobinQ']

The features to the **x_df** dataframe.

x_df = master_df[features]

### Gradient boosting
**Xgboost** is implemented as a Python library, which we import here and name it **xgb**.

import xgboost as xgb

Xgboost uses its' own data structure, called DMatrix. It speeds up calculations significantly and saves memory. We feed the data as pandas dataframes. The data can also be numpy arrays. **nthread = -1** tells Xgboost to use all the cores available for calculations.

dtrain = xgb.DMatrix(x_df, label=y_df, nthread = -1)

Next, we need to define the parameters of the xgboost model. This is a very difficult task and more like black magic than science. You can easily play with different hyperparameter settings for days, and still finding combinations that improve performance. And here is only part of the parameters! More info about the parameters is here: https://xgboost.readthedocs.io/en/latest/parameter.html

m_depth = 5
eta = 0.1
ssample = 0.8
col_tree = 0.8
m_child_w = 3
gam = 1.
objective = 'reg:squarederror'
param = {'max_depth': m_depth, 'eta': eta, 'subsample': ssample,
         'colsample_bytree': col_tree, 'min_child_weight' : m_child_w, 'gamma' : gam,'objective' : objective}

Xgboost has a function for cross-validation. We use here 5 folds. The metric is **mean absolute error**.
![validation](./images/Train-Test-Validation.png)

temp = xgb.cv(param,dtrain,num_boost_round=1500,nfold=5,seed=10,metrics='mae')

To plot how our *mae* is decreasing, we load Matplotlib.

import matplotlib.pyplot as plt

There are indications for overfitting, but let's proceed. Around 800 rounds (decision trees), the validation error is minimum, so let's use that.

fig, axs = plt.subplots(1,2,figsize=(12,6),squeeze=True)
axs[0].plot(temp['test-mae-mean'][400:1500],'r--')
axs[1].plot(temp['train-mae-mean'][400:1500],'b--')

b_rounds = 800

**train()** is used for training. We feed the parameters, the data in a DMAtrix format and the number of boosting rounds to the function.

bst = xgb.train(param,dtrain,num_boost_round=b_rounds)

### SHAP

Now we have our model trained, and we can start analysing it. Let's start with SHAP (https://github.com/slundberg/shap)

import shap

j=0
shap.initjs()

We define a SHAP tree-explainer and use the data to calculate the SHAP values.

explainerXGB = shap.TreeExplainer(bst)
shap_values_XGB_test = explainerXGB.shap_values(x_df,y_df)

SHAP has many convenient functions for model analysis.

Summary_plot with **plot_type = 'bar'** for a quick feature importance analysis. However, for global importance analysis, you should use SAGE instead, because SHAP is prone to errors with the least important features.

shap.summary_plot(shap_values_XGB_test,x_df,plot_type='bar',max_display=30)

With **plot_type = 'dot'** we get a much more detailed plot.

shap.summary_plot(shap_values_XGB_test,x_df,plot_type='dot',max_display=30)

Next, we use the SHAP values to build up 2D scatter graphs for every feature. It shows the effect of a feature for the prediction for every instance.

fig, axs = plt.subplots(7,3,figsize=(16,22),squeeze=True)
ind = 0
for ax in axs.flat:
    feat = bst.feature_names[ind]
    ax.scatter(x_df[feat],shap_values_XGB_test[:,ind],s=1,color='gray')
#    ax.set_ylim([-0.2,0.2])
    ax.set_title(feat)
    ind+=1
plt.subplots_adjust(hspace=0.8)
plt.savefig('shap_sc.png')

**Decision_plot()** is interesting as it shows how the prediction is formed from the contributions of different features.

shap.decision_plot(explainerXGB.expected_value,shap_values_XGB_test[0:100],features)

**Force_plot** is similar to decision_plot. We plot only the first 100 instances because it would be very slow to draw a force_plot with all the instances.

shap.force_plot(explainerXGB.expected_value,shap_values_XGB_test[0:100],features,figsize=(20,10))

**Waterfall_plot** is great when you want to analyse one instance.

shap.waterfall_plot(explainerXGB.expected_value,shap_values_XGB_test[2000],x_df.iloc[2000],features)

### Other interpretation methods

For the following methods, we need to use Xgboost's Scikit-learn wrapper **XGBRegressor()** to turn our Xgboost to be compatible with the Scikit-learn ecosystem.

m_depth = 5
eta = 0.1
ssample = 0.8
col_tree = 0.8
m_child_w = 3
gam = 1.
objective = 'reg:squarederror'
param = {'max_depth': m_depth, 'eta': eta, 'subsample': ssample,
         'colsample_bytree': col_tree, 'min_child_weight' : m_child_w, 'gamma' : gam,'objective' : objective}

Our xgboost model as a Scikit-learn model.

best_xgb_model = xgb.XGBRegressor(colsample_bytree=col_tree, gamma=gam,
                                  learning_rate=eta, max_depth=m_depth,
                                  min_child_weight=m_child_w, n_estimators=800, subsample=ssample)

**fit()** is used to train a model in Scikit.

best_xgb_model.fit(x_df,y_df)

**pdpbox** library has a function for partial dependence plot and individual conditional expectations: https://github.com/SauceCat/PDPbox

from pdpbox import pdp

Here is a code to draw a partial dependence plot and individual conditional expectations. **features[5]** is the feature Return on Assets. These methods do not like missing values in features, so we fill missing values with zeroes. Not a theoretically valid approach, but...

plt.rcParams["figure.figsize"] = (20,20)
pdp_prep = pdp.pdp_isolate(best_xgb_model,x_df.fillna(0),features,features[5])
fig, axes = pdp.pdp_plot(pdp_prep, features[5],center=False, plot_lines=True,frac_to_plot=0.5)
plt.savefig('ICE.png')

ALEPython has functions for ALE plots: https://github.com/blent-ai/ALEPython

import ale

plt.rcdefaults()
ale.ale_plot(best_xgb_model,x_df.fillna(0),features[5],monte_carlo=True)
plt.savefig('ale.png')

