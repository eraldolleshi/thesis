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
import joblib
from joblib import dump, load


from benchmark_class import Benchmark


class ImageClassificationApplication(Benchmark):
    def __init__(self):
        super().__init__(
            "cifar",
            "cnn",
            "input_datasets/cifar10_train.bin",  
            "input_datasets/cifar10_test.bin",
            3072,
            33000,
            1400000,
            0,
            127,
            255
        )

    def _reconstruct(self, path_to_binary_file):
        data_list = []

        with open(path_to_binary_file, "rb") as file:
            while True:
                row_list = []
                for _ in range(3072):
                    chunk = file.read(1)
                    if not chunk:
                        break
                    number = struct.unpack("<B", chunk)[0]
                   
                    row_list.append(number)
               
                if not row_list:
                    break
                row_list = np.array(row_list, dtype=np.uint8).reshape(32, 32, 3)
                row_list = row_list / 255
                

                data_list.append(row_list)
              

        return np.array(data_list)

    def reconstruct_test_dataset(self, path_to_binary_file):

        (_, _), (_, y_test) = tf.keras.datasets.cifar10.load_data()
        y_test = y_test[:1000]

        return self._reconstruct(path_to_binary_file), y_test

    def reconstruct_training_dataset(
        self, path_to_binary_file
    ): 
        (_, y_train), (_, _) = tf.keras.datasets.cifar10.load_data()
        #y_train = y_train
        y_train = np.concatenate([y_train] * 5, axis=0)
        
        return self._reconstruct(path_to_binary_file), y_train

    def run_training(self, x_train, y_train):
        pass

    def save_model(self, model, path_where_to_save_model):
       
        joblib.dump(model,path_where_to_save_model)
        return model

    def load_model(self, path_to_model):
        
        return joblib.load(path_to_model)

    def run_inference(self, model, x_test):
        pass
    def encode_labels(self, labels):
        le = LabelEncoder()
        labels = le.fit_transform(labels)
        return labels
    

Benchmark.benchmark_classes["Benchmark2"] = ImageClassificationApplication