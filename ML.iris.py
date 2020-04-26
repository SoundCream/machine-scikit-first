from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
from joblib import dump, load
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
iris = load_iris()

# View feature names and target names 
feature_names = iris["feature_names"]
target_names = iris["target_names"]

# Printing features and target names of our dataset.
print("Feature names:", feature_names) 
print("Target names:", target_names) 

## Dataset
X = iris["data"] 
y = iris["target"] 
# print("data  :", X) 
# print("target:", y) 

## Standardization (remove mean) **
# X_scaled = preprocessing.scale(X)

# splitting X and y into training and testing sets 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=1) 

## Normalize mode l2
X_train_normalize = preprocessing.normalize(X_train, norm='l2')
X_test_normalize = preprocessing.normalize(X_test, norm='l2')

## Mapping Uniform distribution
quantile_transformer = preprocessing.QuantileTransformer(n_quantiles=len(X_train_normalize), random_state=0)
X_train_trans = quantile_transformer.fit_transform(X_train_normalize)
X_test_trans = quantile_transformer.transform(X_test_normalize)

## Load model
# knn = load("model.iris_knn.pkl")

## Training the model on training set **
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

knn_with_preprocess = KNeighborsClassifier(n_neighbors=3)
knn_with_preprocess.fit(X_train_trans, y_train) 

score = knn.score(X_test, y_test)
score_with_preprocess = knn_with_preprocess.score(X_test_trans, y_test)
print("kNN Score is: %s, kNN (with normalized & unitform) is: %s" % (score, score_with_preprocess))

# making prediction for out of sample data 
sample = [[5.4, 3.4, 1.7, 0.2], [6.7, 3, 5.2, 2.3], [5, 2, 3.5, 1]]
preds = knn.predict(sample) 
pred_result = [target_names[p] for p in preds] 
print("Ex Predictions : ", pred_result) 

# saving the model 
dump(knn, "model.iris_knn.pkl")