
import tensorflow as tf
import struct
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import Dense, Dropout, SimpleRNN


from benchmark_class import Benchmark
from emg_class import HgrApplication

class Hgr_NN(HgrApplication):
    def __init__(self):
        super().__init__()  # Call the parent class initialization
        
        self.model_name = "nn" 


    def run_training(self, x_train, y_train):
            model = Sequential()
            model.add(
            Dense(
                units=64,
                kernel_initializer="normal",
                activation="sigmoid",
                input_dim=x_train.shape[1],
                )   
            )
            model.add(Dropout(0.2))
            model.add(Dense(units=6, kernel_initializer="normal", activation="softmax"))
            model.compile(
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
            )
            model.fit(x_train, y_train, batch_size=256, epochs=500)
            return model  
    
    def run_inference(self, model, x_test):
        y_pred = model.predict(x_test)
        y_pred = np.argmax(y_pred, axis=1)
        return y_pred
    


Benchmark.benchmark_classes["emg_nn"] = Hgr_NN