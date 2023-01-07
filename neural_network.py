from sklearn.neural_network import MLPRegressor
from sklearn import metrics
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, Activation, Flatten
from keras import Input




def keras_neural_network(x_train, y_train, x_test, y_test):
    model = Sequential()
    #adding the input layer
    model.add(Input(shape=(7)))
    #adding the other layers
    model.add(Dense(50, activation = 'relu'))
    model.add(Dense(20, activation= 'relu'))
    model.add(Dense(10, activation = 'relu'))
    #adding output layer
    model.add(Dense(1, activation='relu'))
    #fitting and testing the model
    model.compile(loss = "mean_squared_error", optimizer = "adam", metrics = ['mean_squared_error'])
    model.fit(x_train, y_train, epochs= 100, validation_split= 0.3, batch_size=20)
    