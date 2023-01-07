from sklearn.ensemble import RandomForestRegressor
import mean_squared_error as mse

def random_forest_regression(x_train, y_train, x_test, y_test):
    model = RandomForestRegressor()
    model.fit(x_train, y_train)
    y_preds = model.predict(x_test)
    accuracy = model.score(x_test, y_test)
    error = mse.get_mean_squared_error(y_test, y_preds)
    return accuracy * 100, error

