from sklearn.preprocessing import StandardScaler

import numpy as np
import scipy.io
import scipy.signal
import pandas as pd

from sklearn.preprocessing import StandardScaler

import numpy as np
import scipy.io
import scipy.signal
import pandas as pd
import os

def create_dataset(dataset_dir, subject_index, classes, window, overlap):
    """Create dataset as numpy array format from .mat file

    Args:
        dataset_dir (string): Directory where subjects folder are contained
        subject_index (list): List of subjects index start from 0 for subject 1
        classes (list): List of classes in string
        window (int): Sample length
        overlap (int): Window overlap

    Returns:
        tuple: Input data as numpy array format, Output data as numpy array format, Demarcation index of each subject in the numpy table 
    """
  
    x_data = np.empty((0, window, 4))
    y_data = np.empty((0, 1))  # labels
    subj_inputs = []  # number of inputs for every subject
    tot_rows = 0

    for subject in subject_index:
        subj_inputs.append(0)
    
        for category, name in enumerate(classes):
            matrix_files = os.listdir(f"{dataset_dir}/S{str(subject + 1)}")      
            num_class_files = len([file for file in matrix_files if name in file and file[-4:] == '.mat'])//2

            for record in range(num_class_files):
                acc = scipy.io.loadmat(f'{dataset_dir}/S{subject + 1}/{name}{record + 1}_acc.mat')['ACC']
                ppg = scipy.io.loadmat(f'{dataset_dir}/S{subject + 1}/{name}{record + 1}_ppg.mat')['PPG'][:, 0:2]  # some PPG files have 3 columns instead of 2
                fusion = np.hstack((acc[:, 1:], ppg[:, 1:]))  # remove x axis (time)
                tot_rows += len(fusion)

                # windowing
                # compute number of windows (lazy way)
                i = 0
                num_w = 0
                while i + window  <= len(fusion):
                    i += (window - overlap)
                    num_w += 1
                # compute actual windows
                x_data_part = np.empty((num_w, window, 4))  # preallocate
                i = 0
                for w in range(0, num_w):
                    x_data_part[w] = fusion[i:i + window]
                    i += (window - overlap)
                x_data = np.vstack((x_data, x_data_part))
                y_data = np.vstack((y_data, np.full((num_w, 1), category)))
                subj_inputs[-1] += num_w

    return x_data, y_data, subj_inputs

def clean_data(x_data):
  """Clean input data. Replacement of the Nan values by an interpolated value of the two adjacent points.
    Replacement of zeros values by an interpolated value of the two adjacent points for the PPG.
    Replacement of some missing values by an interpolated value of the two adjacent points for the accelerometer 

    Args:
        x_data (np.array): Cleaned input data
  """

  for i in range(x_data.shape[0]):
    for col in range(0,4):
      ids = np.where(np.isnan(x_data[i,:, col]))[0]
      for row in ids:
        x_data[i, row, col] = 0.5 * (x_data[i, row - 1, col] + x_data[i, row + 1, col])

    for col in range(3, 4):
      ids = np.where(x_data[i,:, col] == 0)[0]
      for row in ids:
        x_data[i,row, col] = 0.5 * (x_data[i,row - 1, col] + x_data[i,row + 1, col])

    for col in range(0, 3):
      for row in range(1, x_data.shape[1] - 1):
        if abs(x_data[i,row, col] - x_data[i,row - 1, col]) > 5000 and abs(x_data[i,row, col] - x_data[i,row + 1, col]) > 5000:
          x_data[i,row, col] = 0.5 * (x_data[i,row - 1, col] + x_data[i,row + 1, col])

def normalize(x_data):
  """Normalize input data. Subtraction of the mean for the accelerometer components, z-norm for the PPG.  

    Args:
        x_data (np.array): Input data.

    Returns:
        np.array: Normalized data.
  """

  for w in x_data:
    # remove mean value from ACC
    w[:, 0] -= np.mean(w[:, 0])  # acc 1
    w[:, 1] -= np.mean(w[:, 1])  # acc 2
    w[:, 2] -= np.mean(w[:, 2])  # acc 3
    # standardize PPG
    w[:, 3] = StandardScaler().fit_transform(w[:, 3].reshape(-1, 1)).reshape((w.shape[0],))  # PPG

  return x_data

def oversampling(x_data, y_data, subj_inputs, num_subjects):
  """Duplicate inputs with classes occurring less, so to have a more balanced distribution.
    We want to do that on a per-subject basis, so to keep subjects separate.
    Moreover, we do that only to the first num_subjects subjects, so to leave test subjects unaltered.

    Args:
        x_data (np.array): Input data.
        y_data (np.array): Output data.
        subj_inputs (List): List of subjects index separation.
        num_subjects (List): List of subjects index separation.

    Returns:
        Tuple: Input data oversampled, Output data oversampled, Corrected subjects index
  """

  x_data_over = None
  y_data_over = None
  subj_inputs_over = []
  skip = 0
  for subj_num in subj_inputs[:num_subjects]:
    x_part = x_data[skip : skip + subj_num]
    y_part = y_data[skip : skip + subj_num]
    occurr = (np.sum(y_part == 0), np.sum(y_part == 1), np.sum(y_part == 2))
    assert(occurr[0] == max(occurr))
    mul = (1, occurr[0] // occurr[1], occurr[0] // occurr[2])
    for cl in (1, 2):
      mask = y_part[:, 0] == cl
      x_dup = x_part[mask].copy()
      y_dup = y_part[mask].copy()
      for n in range(0, mul[cl] - 1):
        x_part = np.vstack((x_part, x_dup))
        y_part = np.vstack((y_part, y_dup))
    if x_data_over is None:
      x_data_over = x_part
      y_data_over = y_part
    else:
      x_data_over = np.vstack((x_data_over, x_part))
      y_data_over = np.vstack((y_data_over, y_part))
    subj_inputs_over.append(len(x_part))
    skip += subj_num
  x_data_over = np.vstack((x_data_over, x_data[skip:]))  # subjects not oversampled
  y_data_over = np.vstack((y_data_over, y_data[skip:]))
  subj_inputs_over.extend(subj_inputs[num_subjects:])

  return x_data_over, y_data_over, subj_inputs_over

def partition_data(subjects, subj_inputs, x_data, y_data):
  """Retrieval of subject data based on subject indices passed in parameters.
    Args:
        subjects (list): List of subjects index.
        subj_inputs (List): List of index subject separation in input data.
        x_data (np.array): Input data
        y_data (np.array): Output data
    Returns:
        tuple: Partionned input data, Partionned output data
  """

  # subjects = tuple (0-based)
  x_part = None
  y_part = None
  for subj in subjects:
    skip = sum(subj_inputs[:subj])
    num = subj_inputs[subj]
    xx = x_data[skip : skip + num]
    yy = y_data[skip : skip + num]
    if x_part is None:
      x_part = xx.copy()
      y_part = yy.copy()
    else:
      x_part = np.vstack((x_part, xx))  # vstack creates a copy of the data
      y_part = np.vstack((y_part, yy))
  return x_part, y_part