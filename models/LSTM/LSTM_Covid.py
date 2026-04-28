import torch
import numpy as np
import pandas as pd
from collections import Counter
from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler, Dataset, random_split
import torch.nn as nn
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

#Determining correct backend
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Training on Apple GPU")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("Training on CUDA")
else:
    print ("MPS device not found.")

df = pd.read_csv("clean_COVIDSenti.csv")

reviews = df.iloc[:, 0].tolist()
encoded_labels = np.array([(label + 1) for label in df.iloc[:, 1].tolist()])

## Build a dictionary that maps words to integers
words = []
for review in reviews:
  new_words = review.split()
  words.extend(new_words)

counts = Counter(words)
vocab = sorted(counts, key=counts.get, reverse=True)
vocab_to_int = {word: ii for ii, word in enumerate(vocab,1)}


## use the dict to tokenize each review in reviews_split
## store the tokenized reviews in reviews_ints
reviews_ints = []
for review in reviews:
  reviews_ints.append([vocab_to_int[word] for word in review.split()])


# stats about vocabulary
print('Unique words: ', len((vocab_to_int)))  # should ~ 74000+
print()

# print tokens in first review
print('Tokenized review: \n', reviews_ints[:1])

# outlier review stats
review_lens = Counter([len(x) for x in reviews_ints])
print("Zero-length reviews: {}".format(review_lens[0]))
print("Maximum review length: {}".format(max(review_lens)))

def pad_features(reviews_ints, seq_length):
    ''' Return features of review_ints, where each review is padded with 0's 
        or truncated to the input seq_length.
    '''
    ## getting the correct rows x cols shape
    features = np.zeros((len(reviews_ints), seq_length), dtype=int)
    
    ## for each review, I grab that review
    for i, row in enumerate(reviews_ints):
      features[i, -len(row):] = np.array(row)[:seq_length]
    
    return features

seq_length = 28
features = pad_features(reviews_ints, seq_length=seq_length)

assert len(features)==len(reviews_ints), "Your features should have as many rows as reviews."
assert len(features[0])==seq_length, "Each feature row should contain seq_length values."

# print first 10 values of the first 30 batches 
print(features[:30,:28])

class TweetDataset(Dataset):
    def __init__(self, features, labels):
        self.x = features
        self.y = labels
        
    def __getitem__(self, index):
        x = self.x[index]
        y = self.y[index]
        return (x, y)
    
    def __len__(self):
        return len(self.x)


# First checking if GPU is available
train_on_gpu=torch.cuda.is_available()

if(train_on_gpu):
    print('Training on GPU.')
else:
    print('No GPU available, training on CPU.')

class SentimentRNN(nn.Module):
    """
    The RNN model that will be used to perform Sentiment analysis.
    """

    def __init__(self, vocab_size, output_size, embedding_dim, hidden_dim, n_layers, drop_prob=0.5):
        """
        Initialize the model by setting up the layers.
        """
        super(SentimentRNN, self).__init__()

        self.output_size = output_size
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        
        # embedding and bidirectional LSTM layers
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers,
                            dropout=drop_prob, batch_first=True, bidirectional=True)
        
        # dropout layer
        self.dropout = nn.Dropout(0.3)
        
        # linear and sigmoid layer (hidden_dim * 2 for bidirectional)
        self.fc = nn.Linear(hidden_dim * 2, output_size)
        self.softmax = nn.Softmax()

    def forward(self, x, hidden):
        """
        Perform a forward pass of our model on some input and hidden state.
        """
        batch_size = x.size(0)
        
        # embeddings and lstm_out
        embeds = self.embedding(x)
        lstm_out, hidden = self.lstm(embeds, hidden)
        
        # stack up lstm outputs
        lstm_out = lstm_out[:, -1, :]
        
        # dropout and fully connected layer
        out = self.dropout(lstm_out)
        out = self.fc(out)
        return out, hidden
    
    
    def init_hidden(self, batch_size):
        ''' Initializes hidden state '''
        # Create two new tensors with sizes n_layers x batch_size x hidden_dim,
        # initialized to zero, for hidden state and cell state of LSTM
        weight = next(self.parameters()).data
        
        if(train_on_gpu):
          hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().cuda(),
                   weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().cuda())
        else:
          hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_(),
                   weight.new(self.n_layers, batch_size, self.hidden_dim).zero_())
        
        return hidden


# dataloaders
batch_size = 50
folds = 5
train_frac = 0.8
test_frac = 0.1
val_frac = 0.1
test_accuracies = []
early_stopping = 5
data = TweetDataset(features, encoded_labels)

for fold in range(folds):
    print(f"FOLD {fold}")
    gen = torch.Generator().manual_seed(fold)
    train, val, test = random_split(data, lengths=[train_frac, val_frac, test_frac], generator = gen)

    #Dealing with imbalanced class weights for train dataset
    labels_for_counts = list(map(lambda x: x[-1], train))
    frequency = 1 / np.bincount(labels_for_counts)
    class_weights = torch.tensor(frequency, dtype=torch.float32)
    obs_weights = list(map(lambda x: class_weights[x[-1]], train))
        
    #train_sampler = WeightedRandomSampler(weights = obs_weights, num_samples = len(obs_weights))
    train_loader = DataLoader(train, batch_size=batch_size, shuffle = True) #Test with shuffle instead of sampler, maybe?
    val_loader = DataLoader(val, shuffle=False, batch_size=batch_size)
    test_loader = DataLoader(test, shuffle=False, batch_size=batch_size)

    # Instantiate the model w/ hyperparams
    vocab_size = len(vocab_to_int) + 1 # +1 for zero padding + our word tokens
    output_size = 3
    embedding_dim = 400 
    hidden_dim = 256
    n_layers = 2
    net = SentimentRNN(vocab_size, output_size, embedding_dim, hidden_dim, n_layers)
    print(net)

    # loss and optimization functions
    lr=0.001

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    # training params

    epochs = 4 # 3-4 is approx where I noticed the validation loss stop decreasing
    epoch = 0
    counter = 0
    print_every = 100
    clip=5 # gradient clipping
    no_improvement = 0
    curr_acc = 0


    # move model to GPU, if available
    if(train_on_gpu):
        net.cuda()

    net.train()

    while no_improvement < early_stopping:
        epoch += 1
        print(f"Epoch {epoch}")

        # Training the model
        # initialize hidden state
        h = net.init_hidden(batch_size)

        # batch loop
        for inputs, labels in train_loader:
            counter += 1

            if(train_on_gpu):
                inputs, labels = inputs.cuda(), labels.cuda()

            # Creating new variables for the hidden state, otherwise
            # we'd backprop through the entire training history
            h = tuple([each.data for each in h])

            # zero accumulated gradients
            net.zero_grad()

            # get the output from the model
            output, h = net(inputs, h)

            # calculate the loss and perform backprop
            loss = criterion(output, labels)
            loss.backward()
            # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
            nn.utils.clip_grad_norm_(net.parameters(), clip)
            optimizer.step()

            #Early stopping
            net.eval()
            correct = torch.tensor(0)
            incorrect = torch.tensor(0)

            # loss stats
            if counter % print_every == 0:
                # Get validation loss
                val_h = net.init_hidden(batch_size)
                val_losses = []
                net.eval()
                all_preds = []
                all_labels = []
                for inputs, labels in val_loader:

                    # Creating new variables for the hidden state, otherwise
                    # we'd backprop through the entire training history
                    val_h = tuple([each.data for each in val_h])

                    if(train_on_gpu):
                        inputs, labels = inputs.cuda(), labels.cuda()

                    output, val_h = net(inputs, val_h)
                    val_loss = criterion(output, labels)

                    val_losses.append(val_loss.item())
                    preds = torch.argmax(output, dim=1)
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
                    correct += (preds == labels).sum()
                    incorrect += (preds != labels).sum()

                accuracy = correct / (correct + incorrect)
                if accuracy > curr_acc:
                    print(f"New accuracy has been reached: {accuracy}")
                    curr_acc = accuracy
                    no_improvement = 0
                else:
                    no_improvement += 1

                    

                net.train()
                print("Epoch: {}".format(epoch),
                    "Step: {}...".format(counter),
                    "Loss: {:.6f}...".format(loss.item()),
                    "Val Loss: {:.6f}".format(np.mean(val_losses)))
    
    net.eval()
    correct = torch.tensor(0)
    incorrect = torch.tensor(0)
    test_h = net.init_hidden(batch_size)
    test_losses = []
    all_preds = []
    all_labels = []
    
    #Getting test accuracy for CV purposes
    for inputs, labels in test_loader:
        test_h = tuple([each.data for each in test_h])
        if(train_on_gpu):
            inputs, labels = inputs.cuda(), labels.cuda()

        output, test_h = net(inputs, test_h)
        test_loss = criterion(output, labels)
        preds = torch.argmax(output, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        correct += (preds == labels).sum()
        incorrect += (preds != labels).sum()

    test_accuracy = correct / (correct + incorrect)
    test_accuracies.append(test_accuracy)
    print(f"FOR FOLD {fold}, THE TEST ACCURACY WAS {test_accuracy}")
    print("---------------------------------------")

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    class_names = ["Negative", "Neutral", "Positive"]

    # Plotting
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix (Test Set)')
    plt.tight_layout()
    plt.show()