from scipy.spatial import distance

class SimpleKNN:
    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test):
        predictions = []
        for row in X_test:
            labels = self.closest(row)
            predictions.append(labels)
        return predictions

    def closest(self, row):
        best_distance = distance.euclidean(row, self.X_train[0])
        best_idx = 0
        for i in range(1, len(self.X_train)):
            current = distance.euclidean(row, self.X_train[i])
            if current < best_distance:
                best_distance = current
                best_idx = i
        return self.y_train[best_idx]
