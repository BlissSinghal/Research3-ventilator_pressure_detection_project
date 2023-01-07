
from os import error
import neural_network
import pca
import data_extraction as d_e
import make_plots
import logistic_regression as l_r
import random_forest
import light_gbm_regressor as lgbm
def execute(ending_index = 25000, split_index = 20000):
    x_train, x_test, y_train, y_test = d_e.extract_data(ending_index, split_index)
    #pca is not really needed
  
    accuracy, error = random_forest.random_forest_regression(x_train, y_train, x_test, y_test)
    print(accuracy)
    print(error)

    #neural_network.keras_neural_network(x_train, y_train, x_test, y_test)


execute()
