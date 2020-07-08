# -*- coding: utf-8 -*-
"""
Created on Wed Jul  8 12:17:38 2020

@author: santi
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('eda data.csv')

#Choose relevent columns
df.columns
df_model = df[['Average_Salary','Rating','Size','Sector','Revenue','Competitors Count','Type of ownership',
              'Location','Age','Same_City','Python','AWS','Spark','Excel','Job Simp','Seniority','Desc Length']]

# Get Dummy Data
df_dum =pd.get_dummies(df_model)

# Train Test Splits
from sklearn.model_selection import train_test_split

X = df_dum.drop('Average_Salary', axis =1)
Y = df_dum['Average_Salary'].values

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.2, random_state = 42)

# Multiple linear regression
import statsmodels.api as sm
X_sm = X = sm.add_constant(X)
model = sm.OLS(Y,X_sm)
model.fit().summary()

from sklearn.linear_model import LinearRegression, Lasso
from sklearn.model_selection import cross_val_score
lm = LinearRegression()
lm.fit(X_train, Y_train)

np.mean(cross_val_score(lm,X_train, Y_train, scoring = 'neg_mean_absolute_error'))

# Lasso regression
lm_l = Lasso(alpha = 0.47)
lm_l.fit(X_train, Y_train)
np.mean(cross_val_score(lm_l,X_train, Y_train, scoring = 'neg_mean_absolute_error'))

alpha = []
error = []

for i in range (1,100):
    alpha.append(i/10)
    lm_l = Lasso(alpha = (i/10))
    error.append(np.mean(cross_val_score(lm_l,X_train, Y_train, scoring = 'neg_mean_absolute_error')))

plt.plot(alpha, error)
err = tuple(zip(alpha,error))
df_err = pd.DataFrame(err, columns = ['Alpha','Error'])
df_err[df_err['Error'] == max(df_err['Error'])]

# Random Forest
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor()

np.mean(cross_val_score(rf, X_train,Y_train,scoring = 'neg_mean_absolute_error'))

# Tune models using gridsearchCV
from sklearn.model_selection import GridSearchCV
parameters = {'n_estimators': range(10,300,10), 'criterion':('mse','mae'), 'max_features':('auto','sqrt','log2')}

gs = GridSearchCV(rf, parameters,scoring = 'neg_mean_absolute_error')
gs.fit(X_train, Y_train)
gs.best_score_
gs.best_estimator_

# Test ensembles
tpred_lm = lm.predict(X_test)
tpred_lm_l = lm_l.predict(X_test)
tpred_rf = gs.best_estimator_.predict(X_test)

from sklearn.metrics import mean_absolute_error
mean_absolute_error(Y_test,tpred_lm)
mean_absolute_error(Y_test,tpred_lm_l)
mean_absolute_error(Y_test,tpred_rf)

