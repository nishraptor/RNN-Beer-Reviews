import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as utils
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
    alphabet  = """abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789&-\",:$%!();.[]?+/'{}@ """
    for i in range(len(str)):
        if str[i] not in alphabet:
            s = list(str)
            s[i] = '@'
            str = "".join(s)

    vector = [[0 if char != letter else 1 for char in alphabet]
              for letter in str]
    return np.array(vector)

def oh2char(vector):
    alphabet  = """abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789&-\",:$%!();.[]?+/'{}@ """
    str = ''
    for i in vector:
        for j in range(len(i)):
            if i[j] != 0:
                str += alphabet[j]
    return str

def load_data(fname):
    # TODO: From the csv file given by filename and return a pandas DataFrame of the read csv.
    file = pd.read_csv(fname)
    return file


def process_train_data(data, beer_styles):
    # TODO: Input is a pandas DataFrame and return a numpy array (or a torch Tensor/ Variable)
    # that has all features (including characters in one hot encoded form).


    # one hot encoded vector
    style_vector = [[0 if char != letter else 1 for char in beer_styles]
                    for letter in data['beer/style']]

    style_vector = np.array(style_vector)

    #Numeric Values for the overall review score
    score_vector = data['review/overall'].values


    #Get review texts and pad them with <EOS> character '}'
    text_list = data['review/text'].values
    text_list = ['{' + str(text) + '}' for text in text_list]
    padded_text_list = pad_data(text_list)

    #One-hot encoding the review text
    text_arrays = [char2oh(review) for review in padded_text_list]
    text_arrays = np.array(text_arrays)

    #Convert the style vector to a 3D-tensor
    style_arrays = np.repeat(style_vector[:, np.newaxis, :], text_arrays.shape[1], axis=1)

    #Convert the score vector to a 3D-tensor
    score_vector = score_vector.reshape(-1,1)
    score_arrays = np.repeat(score_vector[:, np.newaxis, :], text_arrays.shape[1], axis=1)

    #Append the style arrays and the score arrays to text arrays
    review_arrays = np.append(text_arrays, style_arrays, axis=2)
    review_arrays = np.append(review_arrays, score_arrays, axis=2)

    #Remove the last character to get the train array
    train_array = review_arrays[:,:-1,:]

    #Remove the first character to get the label array
    label_array = review_arrays[:,1:,:84]

    #Swap the axes
    train_array = np.swapaxes(train_array, 0, 1)
    label_array = np.swapaxes(label_array, 0, 1)

    target = np.argmax(label_array, 2)

    target = torch.from_numpy(target).long()

    return torch.from_numpy(train_array).float(), target.permute(1,0)


def train_valid_split(data):

    #Shuffle the dataset
    data = data.sample(frac=1).reset_index(drop=True)

    #Get the index to the end of the validation set
    val_index = int(len(data.index)* 0.1)

    return data, val_index

    

def process_test_data(data, beer_styles):
    # TODO: Takes in pandas DataFrame and returns a numpy array (or a torch Tensor/ Variable)
    # that has all input features. Note that test data does not contain any review so you don't
    # have to worry about one hot encoding the data.

    # one hot encoded vector
    style_vector = [[0 if char != letter else 1 for char in beer_styles]
                    for letter in data['beer/style']]

    style_vector = np.array(style_vector)

    # Numeric Values for the overall review score
    score_vector = data['review/overall'].values

    # Generate a text review array of size batch_size x 1 (only one char '{' to one-hot encode)
    sos_string = '{' * len(data.index)
    sos_list = list(sos_string)
    text_array = [char2oh(char) for char in sos_list]
    text_array = np.array(text_array)

    # Convert the style vector to a 3D-tensor
    style_arrays = np.repeat(style_vector[:, np.newaxis, :], text_array.shape[1], axis=1)

    # Convert the score vector to a 3D-tensor
    score_vector = score_vector.reshape(-1, 1)
    score_arrays = np.repeat(score_vector[:, np.newaxis, :], text_array.shape[1], axis=1)

    # Append the style arrays and the score arrays to text arrays
    review_arrays = np.append(text_array, style_arrays, axis=2)
    review_arrays = np.append(review_arrays, score_arrays, axis=2)

    train_array = np.swapaxes(review_arrays, 0, 1)

    return torch.from_numpy(train_array).float()


def get_beer_style(data):

    style = data['beer/style'].unique()

    return style



def pad_data(orig_data):
    # TODO: Since you will be training in batches and training sample of each batch may have reviews
    # of varying lengths, you will need to pad your data so that all samples have reviews of length
    # equal to the longest review in a batch. You will pad all the sequences with <EOS> character 
    # representation in one hot encoding.

    padded_data = []

    #Get length of largest string
    max_len = len(max(orig_data, key=len))

    for text in orig_data:

        #Length difference between current string and longest string
        diff = max_len - len(text)

        #Generate padding of <EOS> chars of length diff
        padding = '}' * diff
        padded_data.append(text + padding)

    return padded_data

def train(model, data, val_index, cfg,computing_device):
    # TODO: Train the model!

    #Define the loss function
    criterion = nn.CrossEntropyLoss()

    #Define the optimizer
    optimizer = torch.optim.Adam(model.parameters(), cfg['learning_rate'])

    #Get all beer style
    beer_styles = get_beer_style(data)

    #Create the seperate dataloaders for Train and Validation

    train_df, val_df = data[0:val_index], data[val_index:]

    #Training loss per epoch, in a list.
    epoch_train_loss = []
    epoch_val_loss = []

    #Iterate over cfg[epochs]
    for epoch in range(cfg['epochs']):
        
        avg_train_loss = []
    
        #Iterate through minbatch (cfg[batch})
        minibatch_size = cfg['batch_size']
        num_batch = int(len(train_df.index) / minibatch_size)

        model.init_hidden(computing_device)

        for minibatch_num in range(num_batch):


            optimizer.zero_grad()

            start_index = minibatch_num * minibatch_size
            end_index = (minibatch_num + 1) * minibatch_size

            minibatch_df = train_df[start_index:end_index]

            # Create the one-hot encoding of training data and labels

            input, target = process_train_data(minibatch_df, beer_styles)

            input, target = input.to(computing_device), target.to(computing_device)

            #Forward pass on minibatch
            output = model(input)

            #Swap axes
            output = output.permute(1,2,0)

            #Compute Loss
            loss = criterion(output, target)

            avg_train_loss.append(loss)

            #Backpropogate
            loss.backward()

            #Update weights
            optimizer.step()

            #Reinitialize the hidden states
            model.init_hidden(computing_device)

            del input
            del output
            print('Loss is %s for minibatch num %s out of total: %s'% (str(loss), str(minibatch_num),str(num_batch)))


        epoch_train_loss.append(sum(avg_train_loss) / len(avg_train_loss))



        avg_val_loss = 0
        #Measure the loss on validation data

    print(epoch_train_loss)


    
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
    np.set_printoptions(threshold=np.nan)

    train_data_fname = "/datasets/cs190f-public/BeerAdvocateDataset/BeerAdvocate_Train.csv"
    test_data_fname = "/datasets/cs190f-public/BeerAdvocateDataset/BeerAdvocate_Test.csv"
    out_fname = "output.txt"

    
    train_data = load_data(train_data_fname) # Generating the pandas DataFrame
    test_data = load_data(test_data_fname) # Generating the pandas DataFrame

    shuffled_data, val_index = train_valid_split(train_data) # Splitting the train data into train-valid data
    X_test = process_test_data(test_data, get_beer_style(shuffled_data)) # Converting DataFrame to numpy array
    
    model = goodLSTM(cfg) # Replace this with model = <your model name>(cfg)
    if cfg['cuda']:
        computing_device = torch.device("cuda")
    else:
        computing_device = torch.device("cpu")
    model.to(computing_device)
    
    train(model, shuffled_data, val_index, cfg, computing_device) # Train the model
    outputs = generate(model, X_test, cfg) # Generate the outputs for test data
    save_to_file(outputs, out_fname) # Save the generated outputs to a file

