from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso

def linear_regression(x_train, y_train, x_test, y_test):
    model = LinearRegression()
    model.fit(x_train, y_train)
    y_preds = model.predict(x_test)
    accuracy = model.score(x_test, y_test)
    return accuracy

def ridge(x_train, y_train, x_test, y_test):
    model = Ridge()
    model.fit(x_train, y_train)
    accuracy = model.score(x_test, y_test)
    return accuracy * 100

def lasso(x_train, y_train, x_test, y_test):
    model = Lasso()
    model.fit(x_train, y_train)
    accuracy = model.score(x_test, y_test)
    return accuracy * 100

