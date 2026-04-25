from sklearn.neural_network import MLPClassifier

def train(X_train, y_train):
    model = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=300)
    model.fit(X_train, y_train)
    return model

def predict(model, X):
    return model.predict(X)