# Beer Review Text Generator

We built a beer review text generator in pytorch based on the dataset from https://arxiv.org/abs/1303.4402 using several different
recurrent neural network topologies:

    - LSTM
    - RNN
    - GRU
    - Bi-directional LSTM

The network was trained using teacher forcing

Once trained, the network produces reviews based on a beer style, rating, and a given temperature for the softmax alphabet output. 
The networks were trained and using 1-2 1080-TI GPUs and tested using various temperatures and model sizes.



# Files

    - main.py
        Trains or generates some output from the network
    - configs.py
        Controls all the parameters of the training/generation
    - models.py
        implementations of different network types
    - results
        Example results at different temperatures
    - weights
        Weights for the trained networks
        
# Paper

The original paper: https://arxiv.org/pdf/1511.03683.pdf which has a demo available at 
http://deepx.ucsd.edu/#/home/beermind


