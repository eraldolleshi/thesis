import tensorflow as tf
import struct
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import Dense, Dropout, SimpleRNN


from benchmark_class import Benchmark


class HarApplication(Benchmark):
    def __init__(self):
        super().__init__(
            "har",
            "input_datasets/har_train.bin",  
            "input_datasets/har_test.bin",
            2244,
            11000,
            150000 #TODO check simulation length since sending the training dataset takes much longer
        )

    def _reconstruct(self, path_to_binary_file):
        data_list = []

        with open(path_to_binary_file, "rb") as file:
            while True:
                row_list = []
                for _ in range(561):
                    chunk = file.read(4)
                    if not chunk:
                        break
                    number = struct.unpack("<f", chunk)[0]
                    if number < -1:
                        number = -1
                    elif number > 1:
                        number = 1
                    row_list.append(number)
                if not row_list:
                    break

                data_list.append(row_list)

        return pd.DataFrame(data_list)

    def reconstruct_test_dataset(self, path_to_binary_file):
        test_data = pd.read_csv("test_shuffled.csv")
        y_test = test_data.iloc[:, -1:]
        y_test = y_test.values

        return self._reconstruct(path_to_binary_file), y_test

    def encode_labels(self, labels):
        le = LabelEncoder()
        labels = le.fit_transform(labels)
        return labels

    def reconstruct_training_dataset(self, path_to_binary_file): 
        train_data = pd.read_csv("HAR_train.csv")
        y_train = train_data.iloc[:, -1:]
        return self._reconstruct(path_to_binary_file), y_train

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
        model.fit(x_train, y_train, batch_size=256, epochs=200)
        return model

    def save_model(self, model, path_where_to_save_model):
        model.save(path_where_to_save_model)
        return model
       

    def save_model(self, model, path_where_to_save_model):
        model.save(path_where_to_save_model)
        return model

    def load_model(self, path_to_model):
        return tf.keras.models.load_model(path_to_model)

    def run_inference(self, model, x_test):
        y_pred = model.predict(x_test)
        y_pred = np.argmax(y_pred, axis=1)
        return y_pred


Benchmark.benchmark_classes["Benchmark1"] = HarApplication
