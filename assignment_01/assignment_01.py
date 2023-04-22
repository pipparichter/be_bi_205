import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from tqdm import tqdm
from time import perf_counter
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

data_dir = '../keren/keren/'
files = os.listdir(data_dir)

# It seems to me that each X array contains a separate image of the cells. 

# I am a little confused as to how to extract each individual cell from the
# image. Ah OK nevermind, this is what the y array is for (I think)

def load_metadata():
    path = data_dir + 'meta.yaml'

    cell_type_to_label = {}
    channels = []

    with open(path, 'r') as metadata:
        lines = metadata.read().splitlines()

        for l in lines[1:19]:
            cell_type, label = l.split(': ')
            cell_type_to_label[int(cell_type[2:])] = label

        for l in lines [20:]:
            channels.append(l[2:])

    return cell_type_to_label, channels


def get_average_data():
    '''
    Average the expression data for each cell for each marker. Produces a file
    where each row contains data for each individual cell (across all files),
    and each column indicates the average expression for a particular marker (so
    there are ncells rows and 51 columns)
    '''
    # Need to make a training dataset with expression information for each cell. 
    training_data = {'cell_number':[], 'marker':[], 'expression':[], 'cell_type':[]}

    for file in tqdm([f for f in files if f != 'meta.yaml'], desc='Loading files...'):
        npz = np.load(data_dir + file, allow_pickle=True)
        # Because npz['cell_types'] has shape (), need to use item() to extract the
        # dictionary. 
        X, y, cell_types = npz['X'], npz['y'], npz['cell_types'].item()
        # Get rid of the weird first dimension.
        X, y = X[0], y[0]
        # X array has shape (1, 2048, 2048, 51). Last index is channels (i.e. markers)
        # y array has shape (1, 2048, 2048, 2) 
        # cell_types array is a dictionary mapping the mask indices to the cell types.
        
        # First, normalize the data... Rohit said to normalize across each
        # marker in each file.
        # Some of the channels have no expression... maybe just handle by
        # subbing out 1, and keeping things the same?
        normalizer = np.amax(X, axis=(0, 1), keepdims=True)
        # Some of the channels have no expression, so ensure no NaNs. 
        normalizer[normalizer == 0] = 1
        X = X/normalizer

        # Probably should use the whole-cell segments, not nuclear...
        y = y[:, :, 0]
       
        cell_num_min, cell_num_max = np.min(y), np.max(y)
        for i in range(cell_num_min, cell_num_max + 1):
            # Take the mean expression across each channel.
            cell_data = np.mean(X[y == i, :], axis=0)
            
            training_data['expression'] += list(cell_data)
            training_data['marker'] += [n for n in range(51)]
            training_data['cell_type'] += [cell_types[i]] * 51

    # Write all the cleaned-up data to a CSV file to avoid re-generating. 
    training_data = pd.DataFrame(training_data)
    training_data.to_csv('data.csv')


def filter_channels(nchannels):
    '''
    Load the data, and select the channels to keep according to which ones show the highest
    variance across cell types. Also generates the data for the marker
    expression plot. 
    '''
    # cell_type_to_label, channels = load_metadata()
    
    data = pd.read_csv('./data.csv')
    data = data.groupby(['cell_type', 'marker'], as_index=False)['expression'].mean()
    data = data.pivot(index='cell_type', columns='marker', values='expression')
    # I think we throw away the first channel? It seems like it might be some

    fig, ax = plt.subplots(figsize=(20, 10))
    
    # For the purposes of selecting which markers should be kept, find the
    # columns with the highest variance. 
    sort_idxs = np.argsort(np.var(data.to_numpy(), axis=0))[51 - nchannels:]
    channels_to_keep = np.arange(51)[sort_idxs]
    # print('Channels to keep:', ', '.join(list(np.array(channels)[sort_idxs])))
    
    return channels_to_keep, data


def create_confusion_matrix(y_pred, y_test):
    '''
    data should be a tuple containing y_test and y_pred. This function creates
    and saves a confusion matrix based on this data. Also need to pass in
    cell_types for labeling purposes. 
    '''

    cell_type_to_label, _ = load_metadata()
    fig, ax = plt.subplots(1, figsize=(15, 10))
    ax.set_title('CelltypeClassifier confusion matrix')
   
    y_pred = torch.argmax(y_pred, 1) # I think we need to actually make predictions. 
    matrix = confusion_matrix(y_test.detach().numpy(), y_pred.detach().numpy())
    
    # It seems as though some cell types might not be included?
    cell_types = np.sort(np.unique(y_test.detach().numpy()))

    # Not totally sure if these are properly labeled. 
    labels = [cell_type_to_label[t] for t in cell_types]
    sns.heatmap(matrix, ax=ax, annot=True, yticklabels=labels, xticklabels=labels)

    fig.savefig('confusion_matrix.png', format='png')


def create_model_trajectory_plots(hists):
    '''
    Create plots showing how model performance (along metrics of accuracy and
    cross-entropy loss) changes over epochs. 
    '''
    train_loss_hist, test_loss_hist, train_acc_hist, test_acc_hist = hists
    
    # Plot loss and accuracy over 200 epochs. 
    fig, ax = plt.subplots(2, figsize=(20, 20))
    ax[0].plot(test_loss_hist, label='test')
    ax[0].plot(train_loss_hist, label='train')
    ax[0].set_xlabel('epoch')
    ax[0].set_ylabel('cross entropy loss')
    ax[0].set_title('CelltypeClassifier loss')
    ax[0].legend()
    
    ax[1].plot(test_acc_hist, label='test')
    ax[1].plot(train_acc_hist, label='train')
    ax[1].set_xlabel('epoch')
    ax[1].set_ylabel('accuracy')
    ax[1].set_title('CelltypeClassifier accuracy')
    ax[1].legend()
 
    fig.savefig('history.png')


def create_marker_expression_panel(nchannels=25):
    '''
    Create and save the marker expression panel for the first deliverable. 
    '''
    _, data = filter_channels(nchannels)
    cell_type_to_label, channels = load_metadata()

    ax.set_title('Keren data marker expression panel')
    
    labels = [cell_type_to_label[i] for i in range(len(cell_type_to_label))]
    sns.heatmap(data, ax=ax, yticklabels=labels, xticklabels=channels, cmap=sns.color_palette("viridis", as_cmap=True))

    fig.savefig('expression_panel.png', format='png')


class CelltypeClassifier(nn.Module):
    '''
    Defining the classifier. 
    '''
    def __init__(self):
        
        super().__init__()
        
        # Based on the expression heat-map, about 16 channels are worth

        # and include 25 channels. 
        input_dim = 25
        hidden_dim = 20
        output_dim = 18
        # Because input dimensions are so low, probably just need one layer. 
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, output_dim)

        self.act = nn.ReLU()
    
    def forward(self, x):
        
        x = self.layer1(x)
        x = self.act(x)
        x = self.layer2(x)
        
        return F.softmax(x)


def accuracy(y, y_target):
    '''
    Calculate the accuracy of a model prediction. 
    '''
    return (torch.argmax(y, 1) == y_target).float().mean().item()


def train(model, data, batch_size=64, epochs=200):
    '''
    Model seems to stop improving after 200 epochs, so I'll set that as the
    default. 
    '''
    X_train, X_test, y_train, y_test = data
    
    # train_loader = torch.utils.data.DataLoader(data, batch_size=batch_size)
    loss_function = nn.CrossEntropyLoss()
    # What is LR? And honestly, what does an optimizer do?
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    train_loss_hist, test_loss_hist, train_acc_hist, test_acc_hist = [], [], [], []
    
    # Keep track of the best accuracy. 
    best_acc = -np.inf

    batches_per_epoch = len(X_train) // batch_size
    for epoch in range(epochs):
        
        # Put the model in train mode. 
        model.train()

        for i in range(batches_per_epoch):
            X_batch = X_train[i * batch_size: i * batch_size + batch_size]
            y_batch = y_train[i * batch_size: i * batch_size + batch_size]
            
            y_pred = model(X_batch)
            loss = loss_function(y_pred, y_batch)
            # What does this do?
            loss.backward()
            
            # What does this do?
            optimizer.step()     
            optimizer.zero_grad()

            train_loss, train_acc = float(loss), accuracy(y_pred, y_batch)
        
        # Put the model in "testing mode" whatever that means. 
        model.eval()
        # Check the test accuracy once every epoch. 
        y_pred = model(X_test)
        test_loss, test_acc = float(loss_function(y_pred, y_test)), accuracy(y_pred, y_test)

        train_loss_hist.append(train_loss)
        test_loss_hist.append(test_loss)
        train_acc_hist.append(train_acc)
        test_acc_hist.append(test_acc)

        print(f'LOSS: {train_loss}')
        
        # Save the model if it beats the previous best test accuracy. 
        if test_acc > best_acc:
            torch.save(model, 'model.pickle')

    return train_loss_hist, test_loss_hist, train_acc_hist, test_acc_hist


def load_data(path='./data.csv', nchannels=25):
    '''
    Load in the average data from a CSV file. 
    '''
    data = pd.read_csv(path, index_col=0)
    # Filter for the channels we care about
    channels_to_keep, _ = filter_channels(nchannels)
    data = data.loc[data['marker'].isin(channels_to_keep)]
    
    y = data[['cell_number', 'cell_type']].groupby('cell_number').first()['cell_type']
    y = y.to_numpy()
    # Remove cell_type column
    data = data[['marker', 'expression', 'cell_number']]
    X = data.pivot_table(columns='marker', values='expression', index='cell_number', fill_value=0, aggfunc='sum')
        
    # X and y need to be tensors. 
    X = torch.tensor(X.to_numpy(), dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.long)

    return X, y


if __name__ == '__main__':
    
    X, y = load_data()
    data = train_test_split(X, y, train_size=0.7, shuffle=True)

#     model = CelltypeClassifier()
#     hists = train(model, data)
#     create_model_trajectory_plots(hists)
#     # get_average_data()

    # Load the best model. 
    model = torch.load('model.pickle')
    y_pred = model(data[1])
    y_test = data[3].to(torch.long)
    print('Final model test accuracy:', accuracy(y_pred, y_test))
    
    create_confusion_matrix(y_pred, y_test)

    pass

# class LinearClassifier(object):
#     '''
#     '''
# 
#     # I think the output_dim should be the number of cell types. 
#     def __init__(self, input_shape, output_dim):
#         '''
#         Shape of the input X is ncells by 51. Output dimension should be the
#         number of potential cell categories. 
#         '''
#         self.input_shape = input_shape
#         self.output_dim = output_dim
#         # NOTE: The example does not seem to include a bias term?
#         # Initialize a bias term. 
#         
#         # Initialize the weights matrix accordint to Glorot uniform distribution.
#         # This ensures variance of each layer (?) is the same. NOTE: Confused
#         # because I didn't think linear classifiers really had layers? 
#         input_nrows, input_ncols = input_shape
#         low, high = -1/np.sqrt(input_ncols), high=1/np.sqrt(input_ncols)
#         self.W = np.random.uniform(low=low, high=high, size=(input_ncols, output_dim))
# 
#     def predict(self, X, epsilon=1e-5):
#         '''
#         '''
#         # In the example code, take the first slice of X... I don't think I will
#         # need to do this.
#         y_pred = np.matmul(X, self.W) + self.b
#         # Apply softmax... NOTE: What does keepdims do?
#         # It seems as though axis = -1 means it is kept as a column array. 
#         y_pred = np.exp(y_pred)/np.sum(np.exp(y_pred) + epsilon, axis=-1, keepdims=True)
#         return y_pred
# 
#     def grad(self, X, y_true):
#         '''
#         Weight vector is like a function of output_dim variables, so the
#         gradient will have dimensions output_dim.
#         '''
#         # Get the predicted probabilities of each class.
#         y_pred = self.predict(X) # Dimensions of predicted y should be input_nrows, output_dim
#         
#         g = []
#         
#         for i in range(self.output_dim):
#             # Function which is 1 if the channel matches.
#             indicator_func = np.zeros(self.input_shape[1])
#             indicator_func[np.where(y_true == i)] = 1
# 
#             g.append(np.dot(y_pred[:, i] - indicator_func, X[:, i]))
#         
#         g = np.stack(g)
# 
#         return grad
#         
# 
#     def loss():
#         # Rohit mentioned something called focal loss, which might be useful to
#         # implement, as this particular dataset is somewhat imbalanced.
#         pass
# 
#     def fit():
#         pass


