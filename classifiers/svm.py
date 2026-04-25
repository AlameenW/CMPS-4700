from sklearn.svm import SVC
import joblib
import os

def train(X_train, y_train):
    model = SVC(probability=True)
    model.fit(X_train, y_train)
    return model

def predict(model, X):
    return model.predict(X)

def save(model, name='svm.pkl'):
    path = os.path.join('MODEL', name)
    joblib.dump(model, path)

def load(name='svm.pkl'):
    path = os.path.join('MODEL', name)
    return joblib.load(path)