import io
import os
import random
import re
import sys
import xml.etree.ElementTree
import numpy as np
import scipy.io
import scipy.signal
import pandas as p
import yaml
import mlflow
import tensorflow as tf
import subprocess

from mlflow.tracking import client
from mlflow.models.signature import infer_signature
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, LSTM, Dropout, InputLayer
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

sys.path.insert(1, os.path.abspath(os.getcwd()) + '/har')
sys.path.insert(1, os.path.abspath(os.getcwd()) + '/har_test')

from har import partition_data
from har_test import *

params = yaml.safe_load(open("params.yaml"))["train_test_evaluate"] 

if len(sys.argv) != 2:
    sys.stderr.write("Arguments error. Usage:\n")
    sys.stderr.write("\tpython prepare.py data-file\n")
    sys.exit(1)

# Test data set split ratio
hyperparameters = {}
hyperparameters["window"] = params["window"]
hyperparameters["overlap"] = params["overlap"]
hyperparameters["epochs"] = params["epochs"]
hyperparameters["dropout_rate"] = params["dropout_rate"]
hyperparameters["learning_rate"] = params["learning_rate"]
hyperparameters["num_cell_dense1"] = params["num_cell_dense1"]
hyperparameters["num_cell_lstm1"] = params["num_cell_lstm1"]
hyperparameters["num_cell_lstm2"] = params["num_cell_lstm2"]
hyperparameters["num_cell_lstm3"] = params["num_cell_lstm3"]
hyperparameters["train_subjects"] = tuple(map(int, params["train_subjects"].split("-")))
hyperparameters["validation_subjects"] = tuple(map(int, params["validation_subjects"].split("-")))
hyperparameters["test_subjects"] = tuple(map(int, params["test_subjects"].split("-")))
hyperparameters["accuracy_thresold"] = params["accuracy_thresold"]
hyperparameters["environment"] = params["environment"]
hyperparameters["classes"] = params["classes"].split("-")


input = sys.argv[1]
x_data_input = os.path.join(sys.argv[1], "x_data.npy")
y_data_input = os.path.join(sys.argv[1], "y_data.npy")
subj_inputs_input = os.path.join(sys.argv[1], "subj_inputs.npy")

def create_model(hyperparameters, num_classes, num_features):
    """Create LSTM model.
    Args:
        args (argparse): Argparse arguments objects.
        num_classes (int): Number of classes.
        num_features (int): Number of input features
    Returns:
        tuple: Partionned input data, Partionned output data
    """
        
    model = Sequential()
    
    model.add(InputLayer(input_shape=(hyperparameters["window"], num_features)))
    model.add(Dense(hyperparameters["num_cell_dense1"], name='dense1'))
    model.add(BatchNormalization(name='norm'))
    
    model.add(LSTM(hyperparameters["num_cell_lstm1"], return_sequences=True, name='lstm1'))
    model.add(Dropout(hyperparameters["dropout_rate"], name='drop2'))
    
    model.add(LSTM(hyperparameters["num_cell_lstm2"], return_sequences=True, name='lstm2'))
    model.add(Dropout(hyperparameters["dropout_rate"], name='drop3'))
    
    model.add(LSTM(hyperparameters["num_cell_lstm3"], name='lstm3'))
    model.add(Dropout(0.5, name='drop4'))
    
    model.add(Dense(num_classes, name='dense2')) 
    
    optimizer = Adam(hyperparameters["learning_rate"])
    
    model.compile(loss=SparseCategoricalCrossentropy(from_logits=True), optimizer=optimizer, metrics=['accuracy'])
    return model 

def promotes_new_model(stage, model_name):
    """Archive all model wih the given stage and promotes the last one.
    Args:
        stage (string): Model stage
        model_name (string): Model name
    """
    mlflowclient = client.MlflowClient()
    max_version = 0

    for mv in mlflowclient.search_model_versions(f"name='{model_name}'"):
        current_version = int(dict(mv)['version'])
        if current_version > max_version:
            max_version = current_version
        if dict(mv)['current_stage'] == stage:
            version = dict(mv)['version']                                   
            mlflowclient.transition_model_version_stage(model_name, version, stage="Archived")

    mlflowclient.transition_model_version_stage(model_name, max_version, stage=stage)

def get_git_revision_hash():
  return subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()

if __name__ == "__main__":
    mlflow.set_tracking_uri('http://deplo-mlflo-5bgwmw63yikr-a96d79bfc9da58f5.elb.us-east-2.amazonaws.com/')

    x_data = np.load(x_data_input)
    y_data = np.load(y_data_input)
    subj_inputs = np.load(subj_inputs_input)

    x_data_train, y_data_train = partition_data(hyperparameters["train_subjects"], subj_inputs, x_data, y_data)
    x_data_validation, y_data_validation = partition_data(hyperparameters["validation_subjects"], subj_inputs, x_data, y_data)
    x_data_test, y_data_test = partition_data(hyperparameters["test_subjects"], subj_inputs, x_data, y_data)

    model_name = "rnn-model"

    mlflow.set_experiment("rnn-model-experiment")

    with mlflow.start_run(run_name='lstm_har') as run:
        mlflow.log_param("epochs", hyperparameters["epochs"])
        mlflow.log_param("learning_rate", hyperparameters["learning_rate"])
        mlflow.log_param("train_subjects", hyperparameters["train_subjects"])
        mlflow.log_param("validation_subjects", hyperparameters["validation_subjects"])
        mlflow.log_param("test_subjects", hyperparameters["test_subjects"])
        mlflow.log_param("num_cell_dense1", hyperparameters["num_cell_dense1"])
        mlflow.log_param("num_cell_lstm1", hyperparameters["num_cell_lstm1"])
        mlflow.log_param("num_cell_lstm2", hyperparameters["num_cell_lstm2"])
        mlflow.log_param("num_cell_lstm3", hyperparameters["num_cell_lstm3"])
        mlflow.log_param("dropout_rate", hyperparameters["dropout_rate"])
        mlflow.log_param("window", hyperparameters["window"])
        mlflow.log_param("overlap", hyperparameters["overlap"])
        mlflow.log_param("accuracy_thresold", hyperparameters["accuracy_thresold"])
        mlflow.log_param("environment", hyperparameters["environment"])
        mlflow.log_param("classes", hyperparameters["classes"])
        mlflow.log_param("data_version", get_git_revision_hash())

        num_classes = len(set(y_data.flatten()))
        num_features = x_data.shape[2]

        model = create_model(hyperparameters, num_classes, num_features)
        
        history = model.fit(x_data_train, y_data_train, epochs=hyperparameters["epochs"], validation_data=(x_data_validation, y_data_validation), steps_per_epoch=5, batch_size=16)
        
        y_pred_proba = model.predict(x_data_test)
        y_pred = np.argmax(y_pred_proba,axis=1)

        accuracy = accuracy_score(y_data_test, y_pred)
        precision = precision_score(y_data_test, y_pred, average='weighted')
        recall = recall_score(y_data_test, y_pred, average='weighted')
        f1 = f1_score(y_data_test, y_pred, average='weighted')
        
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1", f1)

        extra_params ={
            "x_data":x_data,
            "y_data":y_data,
            "model":model,
            "window":hyperparameters["window"],
            "num_features":num_features,
            "num_classes":num_classes
        }

        test_names = ['test_shape_input', 'test_shape_output', 'test_model_input', 'test_model_output', 'test_prediction_shape']
        test_result = execute_test(test_names, mlflow, extra_params)

        signature = infer_signature(x_data_test, model.predict(x_data_test))

        input_example = np.expand_dims(x_data_train[0], axis=0)

        if not False in test_result and accuracy >= hyperparameters["accuracy_thresold"]: 
          mlflow.keras.log_model(model, model_name,signature=signature, input_example=input_example,registered_model_name=model_name)

          run = mlflow.active_run()

          artifact_file = "artifact_location.txt"
          if os.path.exists(artifact_file):
            os.remove(artifact_file)

          with open(artifact_file,"w+") as f:
            f.write(run.info.artifact_uri + "/" + model_name)

          if hyperparameters["environment"] == "development":
            pass 
          if hyperparameters["environment"] == "staging":
            promotes_new_model("Staging", model_name)
          if hyperparameters["environment"] == "main":
            promotes_new_model("Production", model_name)
          if hyperparameters["environment"] == "environement-replace":
            pass

        else:
         mlflow.keras.log_model(model, model_name,signature=signature, input_example=input_example)
         raise ValueError("The tests on the models did not pass or the required accuracy was not achieved.")