from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from joblib import dump, load
import pandas as reader

data = reader.read_csv('dataset.weather.csv')

# Get feature & response
feature_names = data.columns.values[:-1]
response_values = data[data.columns[-1]].array
target_names = list(set(response_values))
print("Features names:", feature_names)
print("Target names  :", target_names)
X = data[data.columns[:-1]].values
y = data[data.columns[-1]].values

# Encoding categorical
outlook = ["overcast", "rainy", "sunny"]
temperature = ["hot", "mild", "cool"]
humidity = ["high", "normal"]
windy = [False, True]
onehoten= preprocessing.OneHotEncoder(categories=[outlook, temperature, humidity, windy])
X_encoded = onehoten.fit_transform(X).toarray()
lben = preprocessing.LabelEncoder()
lben.fit(y)
y_encoded = lben.transform(y)
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y_encoded, test_size=0.3, random_state=1) 

knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train) 

score = knn.score(X_test, y_test)
print("kNN Score:", score)

# sample = [[1. 0. 0. 0. 0. 1. 1. 0. 0. 1.]] = (overcast,cool,high,true) result : yes
sample = [[1., 0., 0., 0., 0., 1., 1., 0., 0., 1.]]
preds = knn.predict(sample) 
pred_species = [target_names[p] for p in preds] 
print("Ex Predictions:", pred_species)

# Save model
dump(knn, "model.weather_knn.pkl")