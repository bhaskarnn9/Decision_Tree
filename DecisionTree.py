# import statements

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns

# read csv
data = pd.read_csv('input/BackOrders.csv')

# head
pd.options.display.html.table_schema = True
pd.options.display.max_rows = None
data.head()

# data-types
data.dtypes

# drop na values
data.shape
data.dropna(axis=0, inplace=True)
data.shape

# drop columns
data.drop('sku', axis=1, inplace=True)

# summary statistics of numerical features
data.describe()  # data.describe(include='all') gives info about num + cat

# summary statistics of object type features
data.describe(include='object')

# get count of unique items in a column
data.nunique()

# get a list of columns that are of dtype 'object'
object_attr = list(data.select_dtypes('object').columns)
object_attr

# convert column type object to cat
for item in object_attr:
    data[item] = data[item].astype('category')

# validate above operation
data.describe(include='category')

# data-types after type casting
data.dtypes

# columns without target variable
object_attr.remove('went_on_backorder')
cat_attr = object_attr
cat_attr

# create dummy variables
data = pd.get_dummies(data, columns=cat_attr, prefix=cat_attr, prefix_sep='_', drop_first=True)
data.columns

# check head to see new columns added after creating dummy variables.
data.head()

# Define predictors and target variables
X, y = data.drop('went_on_backorder', axis=1), data.went_on_backorder

# train test train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=123)

# validate the splitting operation
print(X.shape)
print(X_train.shape)
print(X_test.shape)

# Validate whether or not our splitting operation is stratified
print(y_train.value_counts())
print(y_train.value_counts() / X_train.shape[0])
print(y_test.value_counts() / X_test.shape[0])

# Basic Modeling

# instantiating model_selection
model_dt = DecisionTreeClassifier()

# fitting model
model_dt.fit(X_train, y_train)

# predict on train data first
y_train_pred = model_dt.predict(X_train)
y_test_pred = model_dt.predict(X_test)

# confusion matrix on train to determine how it learnt
confusion_matrix(y_true=y_train, y_pred=y_train_pred)

# confusion matrix on test to determine how it performed
confusion_matrix(y_true=y_test, y_pred=y_test_pred)

# accuracy score on train prediction
accuracy_score(y_true=y_train, y_pred=y_train_pred)

# accuracy score on test prediction
accuracy_score(y_true=y_test, y_pred=y_test_pred)

# get tree length (if interested)
print(model_dt.get_depth())

# get tree length with leaves (if interested)
print(model_dt.get_n_leaves())

# recall score on train prediction
recall_score(y_true=y_train, y_pred=y_train_pred, pos_label='Yes')

# recall score on test prediction
recall_score(y_true=y_test, y_pred=y_test_pred, pos_label='Yes')

# feature importance
feature_importances = model_dt.feature_importances_
feature_importances

# plot feature_importances
plt.figure(figsize=(12, 8))
sns.set_style('whitegrid')
plt.barh(range(len(feature_importances)), feature_importances)
plt.show()

features = list(X_train.columns)
plt.figure(figsize=(12, 8))
d_f = pd.DataFrame(feature_importances)
d_f[0].sort_values().plot(kind='barh')

sorted_indices = np.argsort(feature_importances)
plt.figure(figsize=(10, 8))
sns.set_style('darkgrid')
plt.barh(range(len(feature_importances)), feature_importances[sorted_indices])
plt.yticks(range(len(feature_importances)), [features[i] for i in sorted_indices])
plt.show()

##  Model 2
model_dt2 = DecisionTreeClassifier(max_leaf_nodes=20)
model_dt2.fit(X_train, y_train)
# predict on train data first
y_train_pred2 = model_dt2.predict(X_train)
y_test_pred2 = model_dt2.predict(X_test)

# confusion matrix on train to determine how it learnt
confusion_matrix(y_true=y_train, y_pred=y_train_pred2)

# confusion matrix on test to determine how it performed
confusion_matrix(y_true=y_test, y_pred=y_test_pred2)

# accuracy score on train prediction
accuracy_score(y_true=y_train, y_pred=y_train_pred2)

# accuracy score on test prediction
accuracy_score(y_true=y_test, y_pred=y_test_pred2)

# get tree length (if interested)
print(model_dt2.get_depth())

# get tree length with leaves (if interested)
print(model_dt2.get_n_leaves())

# recall score on train prediction
recall_score(y_true=y_train, y_pred=y_train_pred2, pos_label='Yes')

# recall score on test prediction
recall_score(y_true=y_test, y_pred=y_test_pred2, pos_label='Yes')

# feature importance
feature_importances2 = model_dt2.feature_importances_
feature_importances2

sorted_indices2 = np.argsort(feature_importances2)
plt.figure(figsize=(10, 8))
sns.set_style('whitegrid')
plt.barh(range(len(feature_importances2)), feature_importances2[sorted_indices2])
plt.yticks(range(len(feature_importances2)), [features[i] for i in sorted_indices2])
plt.show()

y_train.head()

## Model - Grid CV
param_grid = {'criterion': ["gini", "entropy"], 'max_depth': [None, 15, 20, 25], 'min_samples_split': [2, 10, 20, 30]}
dt_cv_model = DecisionTreeClassifier()
gs_object = GridSearchCV(dt_cv_model, param_grid=param_grid, verbose=0, n_jobs=-1, cv=3)
gs_object.fit(X_train.values, y_train.values)
