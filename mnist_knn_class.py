
import tensorflow as tf
import struct
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier


from benchmark_class import Benchmark
from mnist_class import HandwrittenDigitClassificationApplication

class Hdr_KNN(HandwrittenDigitClassificationApplication):
    def __init__(self):
        super().__init__()  # Call the parent class initialization
        
        self.model_name = "knn" 


    def run_training(self, x_train, y_train):
        model = KNeighborsClassifier(n_neighbors=5)
        model.fit(x_train, y_train)
    
   
        return model
    
    def run_inference(self, model, x_test):
        y_pred = model.predict(x_test)
        return y_pred
    


Benchmark.benchmark_classes["mnist_knn"] = Hdr_KNN