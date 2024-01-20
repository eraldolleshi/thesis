

from sklearn.svm import LinearSVC


from benchmark_class import Benchmark
from mnist_class import HandwrittenDigitClassificationApplication

class Hdr_SVM(HandwrittenDigitClassificationApplication):
    def __init__(self):
        super().__init__()  # Call the parent class initialization
        
        self.model_name = "svm" 


    def run_training(self, x_train, y_train):
        model = LinearSVC(C=1.0, random_state=42)
        model.fit(x_train, y_train)
        return model
    
    def run_inference(self, model, x_test):
        y_pred = model.predict(x_test)
        return y_pred
    


Benchmark.benchmark_classes["mnist_svm"] = Hdr_SVM