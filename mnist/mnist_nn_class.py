
import tensorflow as tf
import numpy as np
from keras.layers import Dense
from tensorflow.keras.layers import Dense


from benchmark_class import Benchmark
from mnist_class import HandwrittenDigitClassificationApplication

class Hdr_NN(HandwrittenDigitClassificationApplication):
    def __init__(self):
        super().__init__()  # Call the parent class initialization
        
        self.model_name = "nn" 


    def run_training(self, x_train, y_train):
        # Instantiate the model.
        model = tf.keras.Sequential()
 
        # Build the model. 
        model.add(Dense(128, activation='relu', input_shape=(784,)))
        model.add(Dense(128, activation='relu'))
        model.add(Dense(10,  activation="softmax"))
        
        model.compile(optimizer="rmsprop",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])
        model.fit(x_train, 
                             y_train, 
                             epochs=21, 
                             batch_size=64, 
        )
    
   
        return model
    
    def run_inference(self, model, x_test):
        y_pred = model.predict(x_test)
        y_pred = np.argmax(y_pred, axis=1)
        return y_pred
    


Benchmark.benchmark_classes["mnist_nn"] = Hdr_NN