import pandas as pd
from sklearn import metrics, preprocessing
from sklearn.cross_validation import cross_val_score, train_test_split
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

df = pd.read_csv('data/transformed_train_compact.csv', sep=',')

# Weekday is a categorical variable. Convert to dummy variables.
# weekday_dummies = pd.get_dummies(df.Weekday)
# df.drop('Weekday', axis=1, inplace=True)
# df = pd.concat([df, weekday_dummies], axis=1)

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

# Train a Logistic Regression classifier
log_reg_clf = LogisticRegression()
log_reg_clf.fit(train_data, train_target)

print(metrics.classification_report(test_target, log_reg_clf.predict(test_data)))

# Train a Random Forest Classifier
random_forest_clf = RandomForestClassifier(n_estimators=100, n_jobs=-1)
random_forest_clf.fit(train_data, train_target)

print(metrics.classification_report(test_target, random_forest_clf.predict(test_data)))

# Train a K Nearest Neighbors Classifier
k_neighbors_clf = KNeighborsClassifier(n_neighbors=10)
k_neighbors_clf.fit(train_data, train_target)

print(metrics.classification_report(test_target, k_neighbors_clf.predict(test_data)))

# Train an SVM classifier
svm_clf = SVC(probability=True)
svm_clf.fit(train_data, train_target)

print(metrics.classification_report(expected, svm_clf.predict(test_data)))

# Train a Voting Classifier
eclf = VotingClassifier([('lr', log_reg_clf), ('rf', random_forest_clf), ('knn', k_neighbors_clf), ('svm', svm_clf)], voting='soft')
eclf.fit(train_data, train_target)

print(metrics.classification_report(test_target, eclf.predict(test_data)))

# Ensemble Methods: Train with Adaptive Boosting
ada_boost_clf = AdaBoostClassifier(n_estimators=1000)
ada_boost_clf.fit(train.drop('TripType', axis=1), train.TripType.values)

expected = test.TripType.values
predicted = ada_boost_clf.predict(test.drop('TripType', axis=1))

print(metrics.classification_report(expected, predicted))

# Ensemble Methods: Gradient Boosting
gradient_boost_clf = GradientBoostingClassifier(
  n_estimators=100, learning_rate=1.0, max_depth=1, random_state=1)
gradient_boost_clf.fit(train.drop('TripType', axis=1), train.TripType.values)

expected = test.TripType.values
predicted = gradient_boost_clf.predict(test.drop('TripType', axis=1))

print(metrics.classification_report(expected, predicted))