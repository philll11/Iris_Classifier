from sklearn import datasets
from sklearn.model_selection import train_test_split
from classifier import SimpleKNN
from sklearn.metrics import accuracy_score

iris = datasets.load_iris()

X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .5)

my_classifier = SimpleKNN()

my_classifier.fit(X_train, y_train)

predictions = my_classifier.predict(X_test)
print(accuracy_score(y_test, predictions))
