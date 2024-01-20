import struct
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import joblib
from joblib import dump,load


from benchmark_class import Benchmark


class HgrApplication(Benchmark):
    def __init__(self):
        super().__init__(
            application_name="emg",
            model_name="nn",
            path_to_train_dataset="input_datasets/emg_train.bin",
            path_to_test_dataset="input_datasets/emg_test.bin",
            sample_size=64,
            test_simulation_length=2000,
            train_simulation_length=7000,
            filler_value_min=-128,
            filler_value_average=0,
            filler_value_max=127 #check simulation length since sending the training dataset takes much longer
        )

    def _reconstruct(self, path_to_binary_file):
        data_list = []

        with open(path_to_binary_file, "rb") as file:
            while True:
                row_list = []
                for _ in range(64):
                    chunk = file.read(1)
                    if not chunk:
                        break
                    number = struct.unpack("<b", chunk)[0]
                    
                    row_list.append(number)
                if not row_list:
                    break

                data_list.append(row_list)

        return pd.DataFrame(data_list)

    def reconstruct_test_dataset(self, path_to_binary_file):
        test_data = pd.read_csv("emg_test_labels.csv",header=None)
        
        y_test = test_data.values
        

        return self._reconstruct(path_to_binary_file), y_test

    def encode_labels(self, labels):
        le = LabelEncoder()
        labels = le.fit_transform(labels)
        return labels

    def reconstruct_training_dataset(self, path_to_binary_file): 
        train_data = pd.read_csv("emg_train_labels.csv",header=None)
       # train_data= pd.concat([train_data] * 5, ignore_index=True)
        y_train = train_data.iloc[:,:]
        
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



