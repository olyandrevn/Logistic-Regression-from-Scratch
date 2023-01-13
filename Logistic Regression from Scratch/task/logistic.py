# write your code here
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression


class CustomLogisticRegression:
    def __init__(self, fit_intercept=True, l_rate=0.01, n_epoch=100):
        self.fit_intercept = fit_intercept
        self.l_rate = l_rate
        self.n_epoch = n_epoch

    def sigmoid(self, t):
        return 1 / (1 + np.exp(-t))

    def predict_proba(self, row, coef_):
        t = row.T @ coef_[1:]
        if self.fit_intercept:
            t = t + coef_[0]
        return self.sigmoid(t)

    def fit_mse(self, X_train, y_train):
        N = X_train.shape[0]
        self.coef_ = np.zeros(X_train.shape[1] + 1)  # initialized weights

        errors = []

        for _ in range(self.n_epoch):
            errors_ = []
            for i, row in enumerate(X_train):
                y_hat = self.predict_proba(row, self.coef_)

                error = (y_hat - y_train[i])**2 / N
                errors_.append(error)

                # update all weights
                self.coef_[1:] = self.coef_[1:] - self.l_rate * (y_hat - y_train[i]) * y_hat * (1 - y_hat) * row
                if self.fit_intercept:
                    self.coef_[0] = self.coef_[0] - self.l_rate * (y_hat - y_train[i]) * y_hat * (1 - y_hat)
            errors.append(errors_)

        return errors

    def fit_log_loss(self, X_train, y_train):
        N = X_train.shape[0]
        self.coef_ = np.zeros(X_train.shape[1] + 1)  # initialized weights

        errors = []

        for _ in range(self.n_epoch):
            errors_ = []
            for i, row in enumerate(X_train):
                y_hat = self.predict_proba(row, self.coef_)

                error = -(y_train[i] * np.log(y_hat) + (1 - y_train[i]) * np.log(1 - y_hat)) / N
                errors_.append(error)

                # update all weights
                self.coef_[1:] = self.coef_[1:] - self.l_rate * (y_hat - y_train[i]) * row / N
                if self.fit_intercept:
                    self.coef_[0] = self.coef_[0] - self.l_rate * (y_hat - y_train[i]) / N
            errors.append(errors_)

        return errors

    def predict(self, X_test, cut_off=0.5):
        predictions = []
        for row in X_test:
            y_hat = self.predict_proba(row, self.coef_)
            y_hat = 1 if y_hat >= cut_off else 0
            predictions.append(y_hat)
        return np.array(predictions)  # predictions are binary values - 0 or 1


cancer = load_breast_cancer(as_frame=True)
df = cancer.frame
columns = ['worst concave points', 'worst perimeter', 'worst radius']

features = df[columns].to_numpy()
target = df['target'].to_numpy()

mean = np.mean(features, axis=0)
std = np.std(features, axis=0)

for i in range(len(features)):
    features[i] = (features[i] - mean) / std

x_train, x_test, y_train, y_test = train_test_split(features, target, train_size=0.8, random_state=43)

logreg_mse = CustomLogisticRegression(fit_intercept=True, l_rate=0.01, n_epoch=1000)
mse_error = logreg_mse.fit_mse(x_train, y_train)
mse_predict = logreg_mse.predict(x_test)
mse_acc = accuracy_score(y_test, mse_predict)

logreg_log_loss = CustomLogisticRegression(fit_intercept=True, l_rate=0.01, n_epoch=1000)
log_loss_error = logreg_log_loss.fit_log_loss(x_train, y_train)
log_loss_predict = logreg_log_loss.predict(x_test)
log_loss_acc = accuracy_score(y_test, log_loss_predict)

logreg_sklearn = LogisticRegression()
logreg_sklearn.fit(x_train, y_train)
sklearn_predict = logreg_sklearn.predict(x_test)
sklearn_acc = accuracy_score(y_test, sklearn_predict)

ans = {'mse_accuracy': mse_acc,
       'logloss_accuracy': log_loss_acc,
       'sklearn_accuracy': sklearn_acc,
       'mse_error_first': mse_error[0],
       'mse_error_last': mse_error[-1],
       'logloss_error_first': log_loss_error[0],
       'logloss_error_last': log_loss_error[-1]}

print(ans)

print(f'''Answers to the questions:
        1) {format(np.min(mse_error[0]), '.5f')}
        2) {round(np.min(mse_error[-1]), 5)}
        3) {round(np.max(log_loss_error[0]), 5)}
        4) {round(np.max(log_loss_error[-1]), 5)}
        5) expanded
        6) expanded''')