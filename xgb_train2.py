import csv
import pandas as pd
import xgboost as xgb
from sklearn import metrics, preprocessing
from sklearn.cross_validation import cross_val_score, train_test_split

df = pd.read_csv('data/transformed_train_full_with_counts.csv', sep=',')

# 90/10 split for cross cross_validation
train = df.sample(frac=0.9, random_state=1)
test = df.loc[~df.index.isin(train.index)]

print(train.shape)
print(test.shape)

# Get training data and target and test data and target
train_data = train.drop('TripType', axis=1)
train_target = train.TripType.values
test_data = test.drop('TripType', axis=1)
test_target = test.TripType.values

# Map categorical variables
mapping = {}
i = 0

for label in sorted(list(set(train_target.astype(int)))):
  mapping[label] = i
  i += 1

inv_mapping = dict((v, k) for k, v in mapping.iteritems())

train_target_mapped = map(lambda label: mapping[label], train_target)
test_target_mapped = map(lambda label: mapping[label], test_target)

xg_train = xgb.DMatrix(train_data.as_matrix(), label=train_target_mapped)
xg_test = xgb.DMatrix(test_data.as_matrix(), label=test_target_mapped)

param = {}
param['objective'] = 'multi:softprob'
param['num_class'] = i
param['max_depth'] = 5
param['eval_metric'] = 'mlogloss'
param['eta'] = 0.05
param['colsample_bytree'] = 0.3
param['min_child_weight'] = 1

watchlist = [ (xg_train,'train'), (xg_test, 'test') ]
num_round = 2000
bst = xgb.train(param, xg_train, num_round, watchlist, early_stopping_rounds=10)

bst.save_model('m1.xgb')

pred = bst.predict(xg_test)

