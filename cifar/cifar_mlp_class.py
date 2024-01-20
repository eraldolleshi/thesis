import tensorflow as tf
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.regularizers import l2
from keras.callbacks import ModelCheckpoint


from benchmark_class import Benchmark
from cifar_class import ImageClassificationApplication

class Ic_MLP(ImageClassificationApplication):
    def __init__(self):
        super().__init__()  # Call the parent class initialization
        
        self.model_name = "mlp" 


    def run_training(self, x_train, y_train):
        num_classes = 10
   
        y_train = tf.keras.utils.to_categorical(y_train, num_classes)

        model = Sequential()

# Flatten the input for the fully connected layers
        model.add(Flatten(input_shape=x_train.shape[1:]))

# Fully connected layer with 512 units and L2 regularization
        model.add(Dense(512, kernel_regularizer=l2(0.01)))
        model.add(BatchNormalization())
        model.add(Activation("relu"))
        model.add(Dropout(0.5))

# Fully connected layer with 256 units and L2 regularization
        model.add(Dense(256, kernel_regularizer=l2(0.01)))
        model.add(BatchNormalization())
        model.add(Activation("relu"))
        model.add(Dropout(0.5))

# Output layer: Fully connected layer with num_classes units for classification
        model.add(Dense(num_classes))
        model.add(Activation("softmax"))



        # compile (Hidden Output)
        model.compile(
    loss="categorical_crossentropy", optimizer="sgd", metrics=["categorical_accuracy"]
)
       
    
        # Automatically saves the best weights of the model, based on best val_accuracy
        model_mcp = ModelCheckpoint(
        filepath="cifar_10.h5",
        monitor="val_categorical_accuracy",
        save_best_only=True,
        verbose=1,
)


        x_test, y_test = self.reconstruct_test_dataset("bsim/bin/input_datasets/cifar10_test.bin")
        y_test = tf.keras.utils.to_categorical(y_test, num_classes)
    # Fiting the model.
        model.fit(
        x_train,
        y_train,
        batch_size=32,
        epochs=30,
        validation_data=(x_test, y_test),
        shuffle=True,
        callbacks=[model_mcp],
    )       
        return model
             
    
    def run_inference(self, model, x_test):
        y_pred = model.predict(x_test)
        y_pred = np.argmax(y_pred, axis=1)
        return y_pred
    


Benchmark.benchmark_classes["cifar_mlp"] = Ic_MLP