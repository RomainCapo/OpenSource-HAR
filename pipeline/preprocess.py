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

from sklearn.preprocessing import StandardScaler

sys.path.insert(1, os.path.abspath(os.getcwd()) + '/har')

from har import create_dataset, clean_data, normalize, oversampling

params = yaml.safe_load(open("params.yaml"))["preprocess"]

if len(sys.argv) != 2:
    sys.stderr.write("Arguments error. Usage:\n")
    sys.stderr.write("\tpython prepare.py data-file\n")
    sys.exit(1)

# Test data set split ratio
window = params["window"]
overlap = params["overlap"] 
classes = params["classes"].split("-")


input = sys.argv[1]
output_x_data = os.path.join("data", "prepared", "x_data.npy")
output_y_data = os.path.join("data", "prepared", "y_data.npy")
output_subj_inputs = os.path.join("data", "prepared", "subj_inputs.npy")

os.makedirs(os.path.join("data", "prepared"), exist_ok=True)

if __name__ == "__main__":

    subjects_index = list(range(len(os.listdir(input))))  

    x_data, y_data, subj_inputs = create_dataset(input, subjects_index, classes, window, overlap)

    clean_data(x_data)
    x_data = normalize(x_data)
    x_data, y_data, subj_inputs = oversampling(x_data, y_data, subj_inputs, 5)

    np.save(output_x_data, x_data)
    np.save(output_y_data, y_data)
    np.save(output_subj_inputs, subj_inputs)
