
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


from benchmark_class import Benchmark
from cifar_class import ImageClassificationApplication

class Ic_CNN(ImageClassificationApplication):
    def __init__(self):
        super().__init__()  # Call the parent class initialization
        
        self.model_name = "cnn" 


    def run_training(self, x_train, y_train):
        num_classes = 10
   
        y_train = tf.keras.utils.to_categorical(y_train, num_classes)

        model = Sequential()
        # Layer 1: Convolutional layer with 64 filters
        model.add(Conv2D(64, (3, 3), padding="same", input_shape=x_train.shape[1:]))
        model.add(Activation("relu"))
        model.add(BatchNormalization())

        # Layer 2: Convolutional layer with 64 filters
        model.add(Conv2D(64, (3, 3)))
        model.add(Activation("relu"))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        # Layer 3: Convolutional layer with 128 filters
        model.add(Conv2D(128, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization())

        # Layer 4: Convolutional layer with 128 filters
        model.add(Conv2D(128, (3, 3)))
        model.add(Activation("relu"))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Flatten())
        # Layer 5: Fully connected layer with 512 units and L2 regularizatio
        model.add(Dense(512, kernel_regularizer=l2(0.01)))
        model.add(Activation("relu"))
        model.add(Dropout(0.5))
        # Output layer: Fully connected layer with num_classes units for classification
        model.add(Dense(num_classes))
        model.add(Activation("softmax"))


        # compile (Hidden Output)
        model.compile(
    loss="categorical_crossentropy", optimizer="sgd", metrics=["categorical_accuracy"]
)
       
        # Creating callbacks for the model.
        # If the model dosen't continue to improve (loss), the trainning will stop.

        # Stop training if loss doesn't keep decreasing.
        model_es = EarlyStopping(monitor="loss", min_delta=1e-5, patience=6, verbose=1)
        model_rlr = ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=6, verbose=1)

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
        callbacks=[model_es, model_rlr, model_mcp],
    )       
        return model
             
    
    def run_inference(self, model, x_test):
        y_pred = model.predict(x_test)
        y_pred = np.argmax(y_pred, axis=1)
        return y_pred
    


Benchmark.benchmark_classes["cifar_cnn"] = Ic_CNN