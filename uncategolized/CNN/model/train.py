import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import datetime as dt
import datetime
from keras.models import Sequential
from keras.layers import Dense, Dropout
# %matplotlib inline
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from sklearn.model_selection import train_test_split

import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'


class myclass():
    def __init__(self, model_stock):
        self.model_stock = model_stock

    def main(self):
        train_X = np.load('preprocess_data/{}_train_X.npy'.format(self.model_stock))
        train_Y = np.load('preprocess_data/{}_train_Y.npy'.format(self.model_stock))
        print(train_X.shape)
        print(train_Y.shape)



        model = Sequential()
        
        model.add(Convolution2D(2, 2, border_mode='same', activation='relu', 
                                input_shape=train_X.shape[1:]))
        model.add(Convolution2D(2, 2, activation='relu'))
        model.add(MaxPooling2D(pool_size=(2)))
        model.add(Dropout(0.25))
        
        model.add(Convolution2D(2, 2, activation='relu', border_mode='same'))
        model.add(Convolution2D(2, 2, activation='relu'))
        model.add(MaxPooling2D(pool_size=(2)))
        model.add(Dropout(0.25))
        
        model.add(Flatten())
        
        model.add(Dense(512, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(1, activation= 'sigmoid'))

        
        # model = Sequential()
        # model.add(Conv1D(10, kernel_size=2, padding='same', input_shape=(10, 32), activation='relu'))
        # model.add(MaxPooling1D(2, padding='same'))
        # model.add(Conv1D(10, 2, padding='same', activation='relu'))
        # model.add(MaxPooling1D(5, padding='same'))
        # model.add(Conv1D(5, 2, padding='same', activation='relu'))
        # model.add(Conv1D(1, 2, padding='same', activation='tanh'))

        # model.compile(loss='mse', optimizer='adam')

        # model.summary()


        # trainデータとtestデータに分割
        # X_train, X_test, y_train, y_test = train_test_split(
        #     train_X,
        #     train_Y,
        #     random_state = 0,
        #     test_size = 0.2
        # )
        
        # print(X_train.shape, y_train.shape, X_test.shape, y_test.shape) 



        epochs = 100
        history = model.fit(train_X, train_Y, validation_split=0.1, epochs=epochs)

        plt.plot(range(epochs), history.history['loss'], label='loss')
        plt.plot(range(epochs), history.history['val_loss'], label='val_loss')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.legend() 
        plt.show()




myclass = myclass('nasdaq100')
myclass.main()