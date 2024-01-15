from abc import ABC, abstractmethod


class Benchmark(ABC):
    def __init__(
        self,
        application_name,
        model_name,
        path_to_train_dataset,
        path_to_test_dataset,
        sample_size,
        test_simulation_length,
        train_simulation_length,
        filler_value_min,
        filler_value_average,
        filler_value_max
    ):
        self.application_name = application_name
        self.model_name = model_name
        self.path_to_train_dataset = path_to_train_dataset
        self.path_to_test_dataset = path_to_test_dataset
        self.sample_size = sample_size
        self.test_simulation_length = test_simulation_length
        self.train_simulation_length = train_simulation_length
        self.filler_value_min = filler_value_min
        self.filler_value_average = filler_value_average
        self.filler_value_max = filler_value_max


    benchmark_classes = {}

    @abstractmethod
    def _reconstruct(self, path_to_binary_file):
        pass

    @abstractmethod
    def reconstruct_test_dataset(self, path_to_binary_file):
        pass

    @abstractmethod
    def reconstruct_training_dataset(self, path_to_binary_file):
        pass

    @abstractmethod
    def encode_labels(self, labels):
        pass

    @abstractmethod
    def run_training(self, x_train, y_train):
        pass

    @abstractmethod
    def save_model(self, model, path_where_to_save_model):
        pass

    @abstractmethod
    def load_model(self, path_to_model):
        pass

    @abstractmethod
    def run_inference(self, model, x_test):
        pass
