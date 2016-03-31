import csv
import pandas as pd
import xgboost as xgb
from sklearn import metrics, preprocessing
from sklearn.cross_validation import cross_val_score, train_test_split

df = pd.read_csv('data/transformed_train_full_with_counts.csv', sep=',')
unlabeled = pd.read_csv('data/transformed_test_full_with_counts.csv', sep=',')

# 80/20 split for cross cross_validation
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
xg_unlabeled = xgb.DMatrix(unlabeled.drop('VisitNumber', axis=1).as_matrix())

###
df = pd.read_csv('data/transformed_train_full.csv', sep=',')
fineline_cols = [v for v in df.columns.values if "FinelineNumber" in v]
counts = {}
i = 0
for col in fineline_cols:
  counts[col] = sum(df[col])
  i += 1
  if i % 100 == 0:
    print i

# Normalize
i = 0
for col in fineline_cols:
  i += 1
  if i % 100 == 0:
    print i
  df[col] = df[col].astype(float) / counts[col]

i = 0
for col in counts.keys():
  i += 1
  if i % 100 == 0:
    print i
  if counts[col] < 500:
    df = df.drop(col, axis=1)

# 80/20 split for cross cross_validation
train = df.sample(frac=0.8, random_state=1)
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
# xg_unlabeled = xgb.DMatrix(unlabeled.drop('VisitNumber', axis=1).as_matrix())
###

# Use softmax for multi-class classification
param = {'objective':'multi:softprob','subsample':0.9,'max_depth':6,
                       'colsample_bytree':0.5,
                     'eval_metric': 'rmse',
                      'colsample_bylevel':0.03,
                      'num_class':38,'eta':0.03}

watchlist = [ (xg_train,'train'), (xg_test, 'test') ]
num_round = 5000
bst = xgb.train(param, xg_train, num_round, watchlist, early_stopping_rounds=10)

# Get test set prediction
# pred = bst.predict(xg_test).astype(int)
# pred = map(lambda x: inv_mapping[x], pred)

# print(metrics.classification_report(test_target, pred))

# Make submission
submission = pd.read_csv("data/sample_submission.csv", index_col="VisitNumber")
submission.iloc[:,:] = bst.predict(xg_unlabeled)
submission.to_csv("data/submission2.csv")

# Output predictions
# with open('data/submission.csv', 'wb') as f:
#   writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
#   header = ['VisitNumber']
#   for k in sorted(mapping.keys()):
#     header.append('TripType_' + str(k))
#   writer.writerow(header)
#   for index, row in unlabeled.iterrows():
#     row = [unlabeled.VisitNumber[index]]
#     for k in sorted(mapping.keys()):
#       if k == unlabeled.Prediction[index]:
#         row.append(1)
#       else:
#         row.append(0)
#     writer.writerow(row)
