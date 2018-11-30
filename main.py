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


def process_train_data(data, beer_styles, computing_device):
    # TODO: Input is a pandas DataFrame and return a numpy array (or a torch Tensor/ Variable)
    # that has all features (including characters in one hot encoded form).

    # One-hot encoding the beer style
    style_vector = [[0 if char != letter else 1 for char in beer_styles]
                    for letter in data['beer/style']]

    style_tensor = torch.from_numpy(np.array(style_vector)).to(computing_device)

    # Numeric Values for the overall review score (Not one-hot encoded)
    score_tensor = torch.from_numpy(data['review/overall'].values).to(computing_device)

    # Get review texts
    text_list = data['review/text'].values

    # One-hot encoding the review text
    text_arrays = [char2oh('{' + str(text) + '}') for text in text_list]

    text_tensor = pad_data(text_arrays, computing_device)

    # Convert the style vector to a 3D-tensor
    style_tensor = style_tensor.unsqueeze(0)
    style_tensor = style_tensor.expand(text_tensor.size()[1], style_tensor.size()[1], style_tensor.size()[2])

    # Convert the score vector to a 3D-tensor
    score_tensor = score_tensor.unsqueeze(0).permute(1, 0).unsqueeze(0).to(computing_device)
    score_tensor = score_tensor.expand(text_tensor.size()[1], score_tensor.size()[1], score_tensor.size()[2])

    # Append the style arrays and the score arrays to text arrays
    text_tensor = text_tensor.permute(1, 0, 2)
    review_tensor = torch.cat((text_tensor, style_tensor, score_tensor.long()), dim=2)

    #Swap axes of review tensor
    review_tensor = review_tensor.permute(1, 0, 2)

    #Remove last character in the train tensor
    train_tensor = review_tensor[:, :-1, :]

    # Swap the axes
    train_tensor = train_tensor.permute(1, 0, 2).float()

    # Remove the first character to get the label array
    label_tensor = review_tensor[:, 1:, :84]

    # Get the max index of the one-hot encoded vectors in labels
    # for use with CrossEntropyLoss, also typing requirement
    target = label_tensor.argmax(dim=2).long()

    return train_tensor, target


def train_valid_split(data):

    #Shuffle the dataset
    data = data.sample(frac=1).reset_index(drop=True)

    #Get the index of the end of the validation set
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

    test_array = np.swapaxes(review_arrays, 0, 1)

    return torch.from_numpy(test_array).float()


def get_beer_style(data):

    style = data['beer/style'].unique()

    return style



def pad_data(orig_data, computing_device):
    # TODO: Since you will be training in batches and training sample of each batch may have reviews
    # of varying lengths, you will need to pad your data so that all samples have reviews of length
    # equal to the longest review in a batch. You will pad all the sequences with <EOS> character 
    # representation in one hot encoding.

    max_len = len(max(orig_data, key=len))
    eos_array = char2oh('}')

    #Get list of padded tensors of one-hot encodings
    tensor_list = [torch.cat((torch.from_numpy(array).to(computing_device),
                              torch.from_numpy(eos_array).to(computing_device).expand(max_len - array.shape[0], eos_array.shape[1])), dim=0)
                   for array in orig_data]

    #Return 3D tensor of one-hot encodings
    return torch.stack(tensor_list)


def train(model, data, val_index, cfg,computing_device):
    # TODO: Train the model!

    print("Training Model!")

    #Define the loss function
    criterion = nn.CrossEntropyLoss()

    #Define the optimizer
    optimizer = torch.optim.Adam(model.parameters(), cfg['learning_rate'], weight_decay=cfg['L2_penalty'])

    #Get all beer style
    beer_styles = get_beer_style(data)

    #Create the seperate dataloaders for Train and Validation

    val_df, train_df = data[0:val_index], data[val_index:]

    minibatch_size = cfg['batch_size']
    num_batch = int(len(train_df.index) / minibatch_size)

    #Training loss per epoch, in a list.
    epoch_minibatch_train_loss = []
    epoch_minibatch_val_loss = []
    epoch_avg_mb_train_loss = []

    #Iterate over cfg[epochs]
    for epoch in range(cfg['epochs']):

        minibatch_train_loss = []
        minibatch_val_loss = []
        avg_mb_train_loss = []

        #Iterate through minbatch (cfg[batch})
        minibatch_size = cfg['batch_size']
        num_batch = int(len(train_df.index) / minibatch_size)

        #Reset hidden states
        model.init_hidden(computing_device)

        #Avg loss
        avg_loss = 0

        #Train over every minibatch
        for minibatch_num in range(num_batch):


            optimizer.zero_grad()

            start_index = minibatch_num * minibatch_size
            end_index = (minibatch_num + 1) * minibatch_size

            minibatch_df = train_df[start_index:end_index]

            # Create the one-hot encoding of training data and labels

            input, target = process_train_data(minibatch_df, beer_styles, computing_device)

            input, target = input.to(computing_device), target.to(computing_device)

            #Forward pass on minibatch
            output = model(input)

            #Swap axes
            output = output.permute(1,2,0)

            #Compute Loss
            loss = criterion(output, target)

            #Save loss value
            minibatch_train_loss.append(loss)

            #Add loss to average calculation
            avg_loss += loss

            #Get the average mb loss every 1000 mb
            if minibatch_num % 1000 == 0:

                #Save avg mb loss
                avg_mb_train_loss.append(avg_loss / minibatch_size)
                avg_loss = 0

            #Backpropogate
            loss.backward()

            #Update weights
            optimizer.step()

            #Reinitialize the hidden states
            model.init_hidden(computing_device)

            #Reduce memory consumption
            del input
            del output

            #Print Loss
            print('Loss is %s for minibatch num %s out of total: %s'% (str(loss), str(minibatch_num),str(num_batch)))


        #Save loss
        epoch_avg_mb_train_loss.append(avg_mb_train_loss)
        epoch_minibatch_train_loss.append(minibatch_train_loss)

        #Run model on validation set

        num_val_batch = int(len(val_df.index) / minibatch_size)

        for val_minibatch_num in range(num_val_batch):

            #Get indices of minibatch
            start_index = val_minibatch_num * minibatch_size
            end_index = (val_minibatch_num + 1) * minibatch_size

            #Get the minibatch df
            minibatch_df = val_df[start_index:end_index]

            #Get the input + target
            val_input, val_target = process_train_data(minibatch_df, beer_styles, computing_device)
            val_input, val_target = val_input.to(computing_device), val_target.to(computing_device)

            #Don't save gradients to lower memory usage
            with torch.no_grad():
                val_output = model(val_input)

            #Preprocess to get the loss
            val_output = val_output.permute(1,2,0)
            loss = criterion(val_output, val_target)

            #Save loss
            minibatch_val_loss.append(loss)

        #Save loss for this epoch
        epoch_minibatch_val_loss.append(minibatch_val_loss)

        #Reinit hidden states
        model.init_hidden(computing_device)


    #Print total losses
    print('avg mb train loss and minibatch val loss:')
    print(epoch_avg_mb_train_loss)
    print(epoch_minibatch_val_loss)

    #Save model to model file
    torch.save(model.state_dict(), cfg['model_name'] +'.pth')

    #Return loss values
    return (epoch_minibatch_train_loss, epoch_avg_mb_train_loss, epoch_minibatch_val_loss)
    
def generate(model, X_test, cfg, computing_device):
    # TODO: Given n rows in test data, generate a list of n strings, where each string is the review
    # corresponding to each input row in test data.

    alphabet = """abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789&-\",:$%!();.[]?+/'{}@ """

    X_test = X_test.to(computing_device)
    print("X_test:", X_test.shape)

    #Iterate through the testing array in batches of the batch size.
    for batch_num in range(len(X_test), cfg['batch_size']):

        #Compute the output of the model over the batch
        with torch.no_grad():
            start = batch_num * cfg['batch_size']
            end = (batch_num + 1) * cfg['batch_size']
            output = model(X_test[:,start:end,:])
            print("Model size:", output.size())


        softmax = softmax_with_temperature(output.cpu().numpy())
        print("Softmax type:",type(softmax))
        print("Softmax shape:", softmax.shape)


        gen_chars = [np.random.choice(list(alphabet), 1, p=softmax[:,dist,:].flatten()) for dist in range(softmax.shape[1])]
        print("Gen chars: ", gen_chars)

        for char in range(cfg['max_len']):

            #Get meta data
            meta_data = X_test[:,start:end,84:]
            char_tensor_list = [torch.from_numpy(char2oh(char)) for char in gen_chars]
            input = torch.cat((torch.stack(char_tensor_list), meta_data), dim=2)
            print(input)

            break


        break

        #Go through each review in the batch
        for review in range(cfg['batch_size']):

            #Softmax the output of the review
            softmax = softmax_with_temperature(output[:,review,:].cpu().numpy())
            [softmax] = softmax.tolist()

            #Prevent floating point errors:
            softmax = [x + (1 - sum(softmax))/len(softmax) for x in softmax]

            print("Max softmax:", max(softmax))
            print("Sum softmax:", sum(softmax))

            #Generate character distribution
            [gen_char] = np.random.choice(list(alphabet), 1, p=softmax)
            print("Character choice:", gen_char)

            for char in range(cfg['max_len']):
                print(X_test[:,start+review-1:start+review,84:].size())
                #Get the metadata information from this review
                meta_data = X_test[:,start+review-1:start+review,84:]
                char_tensor = torch.from_numpy(char2oh(str(gen_char)))
                next_char = torch.cat((char_tensor, meta_data), dim=2)



                #Append it to the character sampled
                # go again until max length or escape char is hitkj.

                break





def loss_to_file(outputs, fname):

    f = open(fname, 'w')

    for i in range(len(outputs)):

        if (i == 0):

            f.write("Train loss per minibatch\n")
        elif (i == 1):

            f.write("Avg train loss per minibatch\n")
        elif (i == 2):

            f.write("Val loss per minibatch\n")

        for j in range(len(outputs[i])):

            f.write("Epoch num: "+ str(j))
            f.write("\n")

            for k in range(len(outputs[i][j])):

                f.write(str(outputs[i][j][k].item()))
                f.write(', ')

            f.write('\n')

def get_model(cfg):

    if (cfg['model_name'] == 'baselineLSTM'):
        return baselineLSTM(cfg)
    elif (cfg['model_name'] == 'biLSTM'):
        return biLSTM(cfg)
    elif (cfg['model_name'] == 'GRU'):
        return GRU(cfg)

def old_softmax(output):
    temperature = cfg['gen_temp']

    return np.exp(output/temperature)/np.sum(np.exp(output/temperature))

def softmax_with_temperature(output):
    temperature = cfg['gen_temp']
    return np.exp(output/temperature)/np.sum(np.exp(output/temperature), axis=2)[:,:,np.newaxis]

def save_to_file(outputs, fname):
    # TODO: Given the list of generated review outputs and output file name, save all these reviews to
    # the file in .txt format.
    raise NotImplementedError


if __name__ == "__main__":
    pd.set_option('display.expand_frame_repr', False)
    np.set_printoptions(threshold=np.nan)

    train_data_fname = "/datasets/cs190f-public/BeerAdvocateDataset/BeerAdvocate_Train.csv"
    test_data_fname = "/datasets/cs190f-public/BeerAdvocateDataset/BeerAdvocate_Test.csv"
    out_fname = cfg['model_name'] + "output.txt"
    loss_out_fname = cfg['model_name'] + "loss_output.txt"

    
    train_data = load_data(train_data_fname) # Generating the pandas DataFrame
    test_data = load_data(test_data_fname) # Generating the pandas DataFrame

    shuffled_data, val_index = train_valid_split(train_data) # Splitting the train data into train-valid data
    X_test = process_test_data(test_data, get_beer_style(shuffled_data)) # Converting DataFrame to numpy array
    
    model = get_model(cfg) # Replace this with model = <your model name>(cfg)
    if cfg['cuda']:
        computing_device = torch.device("cuda")
    else:
        computing_device = torch.device("cpu")
    model.to(computing_device)

    if cfg['train']:
        loss = train(model, shuffled_data, val_index, cfg, computing_device) # Train the model
        loss_to_file(loss, loss_out_fname)
    else:
        model.load_state_dict(torch.load(cfg['model_name'] + '.pth'))
        model.eval()
        outputs = generate(model, X_test, cfg, computing_device) # Generate the outputs for test data
        save_to_file(outputs, out_fname) # Save the generated outputs to a file

