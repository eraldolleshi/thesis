import tensorflow as tf
import struct

import numpy as np
from sklearn.preprocessing import LabelEncoder
import joblib
from joblib import dump, load




from benchmark_class import Benchmark


class HandwrittenDigitClassificationApplication(Benchmark):
    def __init__(self):
        super().__init__(
            application_name="mnist",
            model_name="nn",
            path_to_train_dataset="input_datasets/mnist_train.bin",
            path_to_test_dataset="input_datasets/mnist_test.bin",
            sample_size=784,
            test_simulation_length=11000,
            train_simulation_length=490000,
            filler_value_min=0,
            filler_value_average=127,
            filler_value_max=255
        )

    def _reconstruct(self, path_to_binary_file):
        # Read the sample bytes from the Bluetooth simulation output binary file
        data_list = []

        with open(path_to_binary_file, "rb") as file:
            while True:
                row_list = []
                for _ in range(784):
                    chunk = file.read(1)
                    if not chunk:
                        break
                    number = struct.unpack("<B", chunk)[0]
                   
                    row_list.append(number)
               
                if not row_list:
                    break
                row_list = np.array(row_list, dtype=np.uint8)
                row_list = row_list / 255.0

                data_list.append(row_list)
                
               
              

        return np.array(data_list)

    def reconstruct_test_dataset(self, path_to_binary_file):
        # Add the labels to the samples to reconstruct the test dataset
        (_, _), (_, y_test) = tf.keras.datasets.mnist.load_data()
        y_test = y_test[:1000]
    

        return self._reconstruct(path_to_binary_file), y_test

    def reconstruct_training_dataset(
        # Add the labels 
        self, path_to_binary_file
    ): 
        (_, y_train), (_, _) = tf.keras.datasets.mnist.load_data()
        # y_train = np.concatenate([y_train] * 4, axis=0)
        return self._reconstruct(path_to_binary_file), y_train
    
     # This method is defined at the subclasses for the different ML models.
    def run_training(self, x_train, y_train):
        pass

    def save_model(self, model, path_where_to_save_model):
        joblib.dump(model,path_where_to_save_model)
        return model

    def load_model(self, path_to_model):
        
        return joblib.load(path_to_model)
    
     # Defined at the subclasses of the ML models
    def run_inference(self, model, x_test):
        pass
    def encode_labels(self, labels):
        le = LabelEncoder()
        labels = le.fit_transform(labels)
        return labels
    

