import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

fn = r'C:\Users\DELL I5558\Desktop\Python\ELEC5222\kNN\NSW-ER01.csv'
dataframe = pd.read_csv(fn)
dataset = dataframe.values
X = dataset[:, 0:23].astype(float)
Y = dataset[:, 23]
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=25)

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski', metric_params=None, n_jobs=1, n_neighbors=6, p=2, weights='uniform')
prediction = knn.predict(X_test)

print('Prediction: \n {}'.format(prediction))
accuracy = knn.score(X_test, y_test)
print(accuracy)
