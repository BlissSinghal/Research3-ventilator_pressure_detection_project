from sklearn.metrics import mean_squared_error

def get_mean_squared_error(y_test, y_preds):
    error = mean_squared_error(y_test, y_preds)
    return error