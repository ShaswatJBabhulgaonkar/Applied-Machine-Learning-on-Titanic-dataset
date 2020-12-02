#!/usr/bin/env python
# coding: utf-8

# In[44]:


# import relevant commands
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[45]:


# Reading the titanic dateset Exploring the one who survived
titanic = pd.read_csv('titanic.csv')
titanic.head()


# Exploratory Data Analysis (EDA)

# In[47]:


# Drop all Categorical Features
cat_feat = ['PassengerId', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked']


# In[48]:


# All dropped categorical features should not be displayed
titanic.describe()


# In[49]:


# Explore continuous features
titanic.groupby('Survived').mean()


# In[92]:


titanic.isnull().sum()


# In[50]:


# Check if 'Age' has missing values and it returns true and false comparing to other features
titanic.groupby(titanic['Age'].isnull()).mean()


# In[31]:


# Plot continuous features
for i in ['Age', 'Fare']:
    died = list(titanic[titanic['Survived'] == 0] [i].dropna())
    survived = list(titanic[titanic['Survived'] == 1] [i].dropna())
    xmin = min(min(died), min(survived))
    xmax = max(max(died), max(survived))
    width = (xmax - xmin)/ 40
    sns.distplot(died, color='r', kde=False, bins=np.arange(xmin, xmax, width))
    sns.distplot(survived, color='g', kde=False, bins=np.arange(xmin, xmax, width))
    plt.legend(['Did not survive', 'Survived'])
    plt.title('Overlaid histogram for {}'.format(i))
    plt.show()


# In[51]:


# Plot remaining continuous features
for i, col in enumerate(['Pclass', 'SibSp', 'Parch']):
    plt.figure(i)
    sns.catplot(x=col, y='Survived', data=titanic, kind='point', aspect=2,)


# In[53]:


# fill the missing values of 'Age' by replacing the missing values with average mean age to make model unbiased
# The model will treat it different if the feature value are missing at random as Age was missing at random
titanic['Age'].fillna(titanic['Age'].mean(), inplace=True)
titanic.isnull().sum()


# In[ ]:


# comine 2 features: multicollinearity
titanic['Family_cnt'] = titanic['SibSp'] + titanic['Parch']
titanic.drop(['SibSp', 'Parch'], axis=1, inplace=True)


# In[54]:


# Drop all continuous features Axis=1 indicate drop columns and not rows
cont_feat = ['PassengerId', 'Pclass', 'Name', 'Age', 'SibSp', 'Parch', 'Fare']
titanic.drop(cont_feat, axis=1, inplace=True)


# In[56]:


# All dropped continuous features should not be displayed
titanic.head()


# In[57]:


# Explore categorical features: 'Sex', 'Cabin', 'Embarked'
titanic.info()


# In[59]:


# Check if 'Cabin' has missing values it only returned for 'Survived' bcoz only it is continuous feature now
# Cabin is not missing at random as we can see below due to splitting power on survival rate
titanic.groupby(titanic['Cabin'].isnull()).mean()


# In[61]:


# if statement using numpy passing a condition and check if cabin is null or not by 0 or 1
titanic['Cabin_ind'] = np.where(titanic['Cabin'].isnull(), 0, 1)
titanic.head(10)


# In[62]:


# Plotting categorical features
for i, col in enumerate(['Cabin_ind', 'Sex', 'Embarked']):
    plt.figure(i)
    sns.catplot(x=col, y='Survived', data=titanic, kind='point', aspect=2,)


# In[68]:


# using pivot table to get M and F boarded at S, C, Q (EDA)
# EDA shows that more people survived at C bcoz they boarded at Cabin and few male survived bcoz they boarded at S as more deaths were observed for S
titanic.pivot_table('Survived', index='Sex', columns='Embarked', aggfunc='count')


# In[69]:


titanic.pivot_table('Survived', index='Cabin_ind', columns='Embarked', aggfunc='count')


# Data Cleaning

# In[71]:


# Cleaning up categorical data Data Cleaning
import numpy as np
import pandas as pd

titanic = pd.read_csv('titanic.csv')
titanic.head()


# In[73]:


titanic.drop(['Name', 'Ticket'], axis=1, inplace=True)


# In[74]:


titanic.head()


# In[75]:


# create an indicator for cabin bcoz cabin had 687 missing values so cabin_ind shows 0 for missing values and 1 for non missing
titanic['Cabin_ind'] = np.where(titanic['Cabin'].isnull(), 0, 1)
titanic.head()


# In[77]:


# convert Sex to numeric
gender_num = {'male': 0, 'female': 1}

titanic['Sex'] = titanic['Sex'].map(gender_num)
titanic.head()


# In[78]:


# drop Cabin and Embarked
# So here we get a nice clean dataset for modeling with all numeric values using EDA and Data Cleaning process
titanic.drop(['Cabin', 'Embarked'], axis=1, inplace=True)
titanic.head()


# Split the dataset:
# 1. Training Data: 60%
# 2. Validation Data: 20%
# 3. Test Data: 20%

# In[89]:


# Splitting of data into train, test and validation set
import pandas as pd
from sklearn.model_selection import train_test_split

titanic = pd.read_csv('titanic.csv')
titanic.head()


# In[90]:


# Splitting of data into train, test and validation set Below is the format
features = titanic.drop('Survived', axis=1)
labels = titanic['Survived']

X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.4, random_state=42)
X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.5, random_state=42)


# In[91]:


print(len(labels), len(y_train), len(y_val), len(y_test))


# In[93]:


# Clean continuous features
import numpy as np
import pandas as pd

titanic = pd.read_csv('titanic.csv')
titanic.head()


# In[94]:


titanic.isnull().sum()


# In[95]:


titanic['Age'].fillna(titanic['Age'].mean(), inplace=True)


# In[96]:


titanic.isnull().sum()


# In[97]:


# comine 2 features: multicollinearity As we can see from the graph it shows the same relation
for i, col in enumerate(['SibSp', 'Parch']):
    plt.figure(i)
    sns.catplot(x=col, y='Survived', data=titanic, kind='point', aspect=2,)


# In[98]:


# comine 2 features: multicollinearity
titanic['Family_cnt'] = titanic['SibSp'] + titanic['Parch']


# In[99]:


# Drop unnecessary variables
titanic.drop(['PassengerId', 'SibSp', 'Parch'], axis=1, inplace=True)


# In[100]:


titanic.head(10)


# In[101]:


# Write out cleaned data
titanic.to_csv('titanic_cleaned.csv', index=False)


# In[114]:


# Clean continuous features
import numpy as np
import pandas as pd

titanic = pd.read_csv('titanic_cleaned.csv')
titanic.head()


# In[115]:


# if statement using numpy passing a condition and check if cabin is null or not by 0 or 1 Create indicator for cabin
titanic['Cabin_ind'] = np.where(titanic['Cabin'].isnull(), 0, 1)


# In[116]:


# convert Sex to numeric
gender_num = {'male': 0, 'female': 1}

titanic['Sex'] = titanic['Sex'].map(gender_num)


# In[117]:


titanic.drop(['Cabin', 'Embarked', 'Name', 'Ticket'], axis=1, inplace=True)
titanic.head()


# In[118]:


# Write out cleaned data
titanic.to_csv('titanic_cleaned.csv', index=False)


# In[157]:


# Splitting of data into train, test and validation set
import pandas as pd
from sklearn.model_selection import train_test_split

titanic = pd.read_csv('titanic_cleaned.csv')
titanic.head()


# In[158]:


# Splitting of data into train, test and validation set Below is the format
features = titanic.drop('Survived', axis=1)
labels = titanic['Survived']

X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.4, random_state=42)
X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.5, random_state=42)


# In[159]:


for dataset in(y_train, y_val, y_test):
    print(round(len(dataset) / len(labels), 2))


# In[160]:


# Write out train, test, validation data
X_train.to_csv('train_features.csv' , index=False)
X_val.to_csv('val_features.csv' , index=False)
X_test.to_csv('test_features.csv' , index=False)

y_train.to_csv('train_labels.csv' , index=False)
y_val.to_csv('val_labels.csv' , index=False)
y_test.to_csv('test_labels.csv' , index=False)


# In[198]:


# Fit Random Forest Model using cross-validation
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

tr_features = pd.read_csv('train_features.csv')
tr_labels = pd.read_csv('train_labels.csv', header=None)


# In[199]:


# using 5 fold cross validation will tune the hyperparameters later
rf = RandomForestClassifier()

scores = cross_val_score(rf, tr_features, tr_labels.values.ravel(), cv=5)


# In[200]:


# 5 scores using cross val average is around 81% 
scores


# In[201]:


# Tuning hyperparameters  replace cross_val_score by GridSearchCV
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

tr_features = pd.read_csv('train_features.csv')
tr_labels = pd.read_csv('train_labels.csv', header=None)


# In[202]:


# prints average and standard deviation accuracy score of 5 fold cross validation
def print_results(results):
    print('BEST PARAMS: {}\n'.format(results.best_params_))
    
    means = results.cv_results_['mean_test_score']
    stds = results.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, results.cv_results_['params']):
        print('{} (+/-{}) for {}'.format(round(mean,3), round(std * 2, 3), params))


# In[206]:


# 2 hyperparameters n_estimators and max_depth
# 12 hyperparameters and 5 cross val so 12*5 = 60 individual models
# Best is 81% as below
rf = RandomForestClassifier()
parameters = {
    'n_estimators': [5, 50, 100],
    'max_depth': [2, 10, 20, None]
}

cv = GridSearchCV(rf, parameters, cv=5)
cv.fit(tr_features, tr_labels.values.ravel())

print_results(cv)


# In[191]:


# Evaluate results on the validation set
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score

tr_features = pd.read_csv('train_features.csv')
tr_labels = pd.read_csv('train_labels.csv', header=None)

val_features = pd.read_csv('val_features.csv')
val_labels = pd.read_csv('val_labels.csv', header=None)

te_features = pd.read_csv('test_features.csv')
te_labels = pd.read_csv('test_labels.csv', header=None)


# In[207]:


# choosing 3 best models from previous tuning hyperparameters
rf1 = RandomForestClassifier(n_estimators=100, max_depth=10)
rf1.fit(tr_features, tr_labels.values.ravel())

rf2 = RandomForestClassifier(n_estimators=100, max_depth=None)
rf2.fit(tr_features, tr_labels.values.ravel())

rf3 = RandomForestClassifier(n_estimators=100, max_depth=20)
rf3.fit(tr_features, tr_labels.values.ravel())


# In[208]:


# Evaluate models on the validation set
for mdl in [rf1, rf2, rf3]:
    y_pred = mdl.predict(val_features)
    accuracy = round(accuracy_score(val_labels, y_pred), 3)
    precision = round(precision_score(val_labels, y_pred), 3)
    recall = round(recall_score(val_labels, y_pred), 3)
    print('MAX DEPTH: {} / # OF EST: {} -- A: {} / P: {} / R: {}'.format(mdl.max_depth, mdl.n_estimators, accuracy, precision, recall))


# In[209]:


# Evaluate the best model on the test set
# On evaluating rf1, rf2, rf3 models on validation set we got rf1 as best model so we used rf1 model for evaluation on test set
y_pred = rf1.predict(te_features)
accuracy = round(accuracy_score(te_labels, y_pred), 3)
precision = round(precision_score(te_labels, y_pred), 3)
recall = round(recall_score(te_labels, y_pred), 3)
print('MAX DEPTH: {} / # OF EST: {} -- A: {} / P: {} / R: {}'.format(mdl.max_depth, mdl.n_estimators, accuracy, precision, recall))


# In[210]:


# Logistic Regression Model
# Gives potential hyperparameters we could tune. We would tune the one which has the largest impact
# The C hyperparameter is regularisation parameter is logistic reg that controls how closely model fits to training data
# More value of C, less regularization and classification good, if less C then more regularization and underfitting
from sklearn.linear_model import LogisticRegression

LogisticRegression()


# In[211]:


dir(LogisticRegression)


# In[214]:


import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

tr_features = pd.read_csv('train_features.csv')
tr_labels = pd.read_csv('train_labels.csv', header=None)


# In[215]:


def print_results(results):
    print('BEST PARAMS: {}\n'.format(results.best_params_))
    
    means = results.cv_results_['mean_test_score']
    stds = results.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, results.cv_results_['params']):
        print('{} (+/-{}) for {}'.format(round(mean,3), round(std * 2, 3), params))


# In[216]:


# Logistic Regression with 5-fold cross validation and tuning the hyperparameter C
# Shows that model underfits when low C & high Regularization & less accuracy; C=1 best accuracy; model overfits when high C & low Reg
# 7 hyperparameters and 5 cross val so 7*5 = 35 individual models
# Best is 80% as below
lr = LogisticRegression()
parameters = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
}

cv = GridSearchCV(lr, parameters, cv=5)
cv.fit(tr_features, tr_labels.values.ravel())

print_results(cv)


# In[217]:


cv.best_estimator_


# In[218]:


# Write out pickled model
joblib.dump(cv.best_estimator_,'LR_model.pkl')


# In[230]:


# SVM Classification Model
# Gives potential hyperparameters we could tune. We would tune the one which has the largest impact
from sklearn.svm import SVC

SVC()


# In[231]:


dir(SVC)


# In[232]:


import joblib
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

tr_features = pd.read_csv('train_features.csv')
tr_labels = pd.read_csv('train_labels.csv', header=None)


# In[233]:


def print_results(results):
    print('BEST PARAMS: {}\n'.format(results.best_params_))
    
    means = results.cv_results_['mean_test_score']
    stds = results.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, results.cv_results_['params']):
        print('{} (+/-{}) for {}'.format(round(mean,3), round(std * 2, 3), params))


# In[234]:


# SVM with 5-fold cross validation and tuning the hyperparameter C and kernel
# Shows that model underfits when low C & high Regularization & low penalty for misclassification in training, when model overfits when high C & low Reg, high penalty for misclassification in training 
# Best is 79.6% as below
svc = SVC()
parameters = {
    'kernel': ['linear', 'rbf'],
    'C': [0.1, 1, 10],
}

cv = GridSearchCV(svc, parameters, cv=5)
cv.fit(tr_features, tr_labels.values.ravel())

print_results(cv)


# In[235]:


cv.best_estimator_


# In[236]:


# Write out pickled model
joblib.dump(cv.best_estimator_,'SVM_model.pkl')


# In[237]:


# Multi-Layer Perceptron MLP Model
# Gives a lot of potential hyperparameters we could tune. We would tune the one which has the largest impact
from sklearn.neural_network import MLPRegressor, MLPClassifier

print(MLPRegressor())
print(MLPClassifier())


# In[241]:


import joblib
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

tr_features = pd.read_csv('train_features.csv')
tr_labels = pd.read_csv('train_labels.csv', header=None)


# In[242]:


def print_results(results):
    print('BEST PARAMS: {}\n'.format(results.best_params_))
    
    means = results.cv_results_['mean_test_score']
    stds = results.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, results.cv_results_['params']):
        print('{} (+/-{}) for {}'.format(round(mean,3), round(std * 2, 3), params))


# In[243]:


# MLP with 5-fold cross validation and tuning the hyperparameter hidden layer sizes, activation, learning rate
# 7 hyperparameters and 5 cross val so 7*5 = 35 individual models
# Best is 80% as below
mlp = MLPClassifier()
parameters = {
    'hidden_layer_sizes': [(10,), (50,), (100,)],
    'activation': ['relu', 'tanh', 'logistic'],
    'learning_rate': ['constant', 'invscaling', 'adaptive']
}

cv = GridSearchCV(mlp, parameters, cv=5)
cv.fit(tr_features, tr_labels.values.ravel())

print_results(cv)


# In[244]:


cv.best_estimator_


# In[245]:


# Write out pickled model
joblib.dump(cv.best_estimator_,'MLP_model.pkl')


# In[246]:


# Random Forest RF Model
# Gives a lot of potential hyperparameters we could tune. We would tune the one which has the largest impact
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

print(RandomForestRegressor())
print(RandomForestClassifier())


# In[247]:


import joblib
import pandas as pd
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

tr_features = pd.read_csv('train_features.csv')
tr_labels = pd.read_csv('train_labels.csv', header=None)


# In[248]:


def print_results(results):
    print('BEST PARAMS: {}\n'.format(results.best_params_))
    
    means = results.cv_results_['mean_test_score']
    stds = results.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, results.cv_results_['params']):
        print('{} (+/-{}) for {}'.format(round(mean,3), round(std * 2, 3), params))


# In[249]:


# RF with 5-fold cross validation
# 2 hyperparameters n_estimators and max_depth
# Best is 82.8% as below
rf = RandomForestClassifier()
parameters = {
    'n_estimators': [5, 50, 250],
    'max_depth': [2, 4, 8, 16, 32, None]
}

cv = GridSearchCV(rf, parameters, cv=5)
cv.fit(tr_features, tr_labels.values.ravel())

print_results(cv)


# In[250]:


cv.best_estimator_


# In[251]:


# Write out pickled model
joblib.dump(cv.best_estimator_,'RF_model.pkl')


# In[252]:


# Boosting Model
# Gives a lot of potential hyperparameters we could tune. We would tune the one which has the largest impact
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier

print(GradientBoostingRegressor())
print(GradientBoostingClassifier())


# In[253]:


import joblib
import pandas as pd
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

tr_features = pd.read_csv('train_features.csv')
tr_labels = pd.read_csv('train_labels.csv', header=None)


# In[254]:


def print_results(results):
    print('BEST PARAMS: {}\n'.format(results.best_params_))
    
    means = results.cv_results_['mean_test_score']
    stds = results.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, results.cv_results_['params']):
        print('{} (+/-{}) for {}'.format(round(mean,3), round(std * 2, 3), params))


# In[255]:


# Gradient Boosting Classifier with 5-fold cross validation
# 3 hyperparameters n_estimators, learning rate and max_depth 
# High learning rate is generating less optimised results
# Best is 84.1% as below
gb = GradientBoostingClassifier()
parameters = {
    'n_estimators': [5, 50, 250, 500],
    'max_depth': [1, 3, 5, 7, 9],
    'learning_rate': [0.01, 0.1, 1, 10, 100]
}

cv = GridSearchCV(gb, parameters, cv=5)
cv.fit(tr_features, tr_labels.values.ravel())

print_results(cv)


# In[256]:


cv.best_estimator_


# In[257]:


# Write out pickled model
joblib.dump(cv.best_estimator_,'GB_model.pkl')


# In[258]:


# Evaluate results on the validation set
import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score
from time import time

val_features = pd.read_csv('val_features.csv')
val_labels = pd.read_csv('val_labels.csv', header=None)

te_features = pd.read_csv('test_features.csv')
te_labels = pd.read_csv('test_labels.csv', header=None)


# In[259]:


models = {}

for mdl in ['LR', 'SVM', 'MLP', 'RF', 'GB']:
    models[mdl] = joblib.load('{}_model.pkl'.format(mdl))


# In[260]:


models


# In[261]:


# Evaluated models on the validation set
def evaluate_model(name, model, features, labels):
    start = time()
    pred = model.predict(features)
    end = time()
    accuracy = round(accuracy_score(labels, pred), 3)
    precision = round(precision_score(labels, pred), 3)
    recall = round(recall_score(labels, pred), 3)
    print('{} -- Accuracy: {} / Precision: {} / Recall: {} / Latency: {}ms'.format(name, accuracy, precision, recall, round((end - start))))


# In[265]:


for name, mdl in models.items():
    evaluate_model(name, mdl, val_features, val_labels)


# In[269]:


# Evaluate the model which performed best for the validation set on the test set
evaluate_model('Gradient Boosting', models['GB'], te_features, te_labels)

