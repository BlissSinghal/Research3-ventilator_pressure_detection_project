from lightgbm.sklearn import LGBMRegressor

def light_gbm_regressor(x_train, y_train, x_test, y_test):
    model = LGBMRegressor()
    model.fit(x_train, y_train)
    return model.best_score_ 