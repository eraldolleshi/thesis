
import tensorflow as tf
import struct
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.regularizers import l2
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.ensemble import RandomForestClassifier
from benchmark_class import Benchmark
from cifar_class import ImageClassificationApplication

class Ic_RF(ImageClassificationApplication):
    def __init__(self):
        super().__init__()  # Call the parent class initialization
        
        self.model_name = "rf" 


    def run_training(self, x_train, y_train):
        num_classes= 10
        x_test, y_test = self.reconstruct_test_dataset("bsim/bin/input_datasets/cifar10_test.bin")
        y_test = tf.keras.utils.to_categorical(y_test, num_classes)

        x_train = np.array(x_train)
        x_train = np.array(x_train).reshape(x_train.shape[0], -1)
        x_test = x_test.reshape(x_test.shape[0], -1)


# Create and train the Random Forest model
        model = RandomForestClassifier(n_estimators=100, random_state=42) 
    # Fiting the model.
        model.fit(
        x_train,
        y_train,
    )       
        return model
             
    
    def run_inference(self, model, x_test):
        x_test = np.array(x_test)
        x_test = x_test.reshape(x_test.shape[0], -1)
        y_pred = model.predict(x_test)
        return y_pred
    


Benchmark.benchmark_classes["cifar_rf"] = Ic_RF