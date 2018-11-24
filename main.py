import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import os
import matplotlib.pyplot as plt
from models import *
from configs import cfg
import pandas as pd
from nltk.translate import bleu_score

# Functions we needed to implement to one-hot encode the vectors
def char2oh(str):
    alphabet  = """abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789&-\",:$%!();.[]?+/' """
    vector = [[0 if char != letter else 1 for char in alphabet]
              for letter in str]
    return vector

def load_data(fname):
    # TODO: From the csv file given by filename and return a pandas DataFrame of the read csv.

    file = pd.read_csv(fname)
    return file


def process_train_data(data):
    # TODO: Input is a pandas DataFrame and return a numpy array (or a torch Tensor/ Variable)
    # that has all features (including characters in one hot encoded form).

    b = data['review/text'].values.tolist()[0:100]


    #b = pd.unique(data['review/text'].values.ravel())
    print(type(b))

def train_valid_split(data, labels):
    # TODO: Takes in train data and labels as numpy array (or a torch Tensor/ Variable) and
    # splits it into training and validation data.
    raise NotImplementedError
    
    
def process_test_data(data):
    # TODO: Takes in pandas DataFrame and returns a numpy array (or a torch Tensor/ Variable)
    # that has all input features. Note that test data does not contain any review so you don't
    # have to worry about one hot encoding the data.
    raise NotImplementedError

    
def pad_data(orig_data):
    # TODO: Since you will be training in batches and training sample of each batch may have reviews
    # of varying lengths, you will need to pad your data so that all samples have reviews of length
    # equal to the longest review in a batch. You will pad all the sequences with <EOS> character 
    # representation in one hot encoding.
    raise NotImplementedError
    

def train(model, X_train, y_train, X_valid, y_valid, cfg):
    # TODO: Train the model!
    raise NotImplementedError
    
    
def generate(model, X_test, cfg):
    # TODO: Given n rows in test data, generate a list of n strings, where each string is the review
    # corresponding to each input row in test data.
    raise NotImplementedError
    
    
def save_to_file(outputs, fname):
    # TODO: Given the list of generated review outputs and output file name, save all these reviews to
    # the file in .txt format.
    raise NotImplementedError
    


if __name__ == "__main__":
    pd.set_option('display.expand_frame_repr', False)

    train_data_fname = "Beeradvocate_Train.csv"
    test_data_fname = "Beeradvocate_Test.csv"
    out_fname = "output.txt"

    
    train_data = load_data(train_data_fname) # Generating the pandas DataFrame
    test_data = load_data(test_data_fname) # Generating the pandas DataFrame

    train_data, train_labels = process_train_data(train_data) # Converting DataFrame to numpy array
    X_train, y_train, X_valid, y_valid = train_valid_split(train_data, train_labels) # Splitting the train data into train-valid data
    X_test = process_test_data(test_data) # Converting DataFrame to numpy array
    
    model = baselineLSTM(cfg) # Replace this with model = <your model name>(cfg)
    if cfg['cuda']:
        computing_device = torch.device("cuda")
    else:
        computing_device = torch.device("cpu")
    model.to(computing_device)
    
    train(model, X_train, y_train, X_valid, y_valid, cfg) # Train the model
    outputs = generate(model, X_test, cfg) # Generate the outputs for test data
    save_to_file(outputs, out_fname) # Save the generated outputs to a file

