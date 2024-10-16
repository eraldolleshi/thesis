from sklearn.ensemble import RandomForestClassifier
from benchmark_class import Benchmark
from emg_class import HgrApplication

class Hgr_RF(HgrApplication):
    def __init__(self):
        super().__init__()  # Call the parent class initialization
        
        self.model_name = "rf" 


    def run_training(self, x_train, y_train):
           model = RandomForestClassifier(n_estimators=100, random_state=42) 


           model.fit(x_train, y_train)
           return model
             
    
    def run_inference(self, model, x_test):
        y_pred = model.predict(x_test)
        return y_pred
    


Benchmark.benchmark_classes["emg_rf"] = Hgr_RF