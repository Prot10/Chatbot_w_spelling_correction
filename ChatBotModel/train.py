import numpy as np
import json
import torch
import torch.nn as nn
import os
from torch.utils.data import Dataset, DataLoader
from nltk_utils import bag_of_words, tokenize1, stem
from model import NeuralNet


# get the current path
current_dir = os.getcwd()
# get the path of the father folder
parent_dir = os.path.dirname(current_dir)
# set the wd to the father's folder
os.chdir(parent_dir)


##########################################
######## Create the train dataset ########
##########################################

path_train = f'{os.getcwd()}/Chatbot_w_spelling_correction/Files/df_train.json'
with open(path_train, 'r') as f:
    df_train = json.load(f)


all_words = []
tags = []
xy = []
# loop through each sentence in our df_train patterns
for intent in df_train['intents']:
    tag = intent['tag']
    # add to tag list
    tags.append(tag)
    for pattern in intent['patterns']:
        # tokenize each word in the sentence
        w = tokenize1(pattern)
        # add to our words list
        all_words.extend(w)
        # add to xy pair
        xy.append((w, tag))


# stem and lower each word
ignore_words = ['?', '.', '!', ',', ';', ':', "'", '"', 'é', 'è']
all_words = [stem(w) for w in all_words if w not in ignore_words]
# remove duplicates and sort
all_words = sorted(set(all_words))
tags = sorted(set(tags))


print(len(xy), "patterns")
print(len(tags), "tags:", tags)
print(len(all_words), "unique stemmed words:", all_words)


# create training data
X_train = []
y_train = []
for (pattern_sentence, tag) in xy:
    # X: bag of words for each pattern_sentence
    bag = bag_of_words(pattern_sentence, all_words)
    X_train.append(bag)
    # y: PyTorch CrossEntropyLoss needs only class labels, not one-hot
    label = tags.index(tag)
    y_train.append(label)


X_train = np.array(X_train)
y_train = np.array(y_train)


##########################################
##### Create the validation dataset ######
##########################################
path_validation = f'{os.getcwd()}/Chatbot_w_spelling_correction/Files/df_validation.json'
with open(path_validation, 'r') as file:
    df_validation = json.load(file)


all_words_validation = []
tags_validation = []
xy_validation = []
# loop through each sentence in our df_train patterns
for intent in df_validation['intents']:
    tag = intent['tag']
    # add to tag list
    tags_validation.append(tag)
    for pattern in intent['patterns']:
        # tokenize each word in the sentence
        w = tokenize1(pattern)
        # add to our words list
        all_words_validation.extend(w)
        # add to xy pair
        xy_validation.append((w, tag))


# stem and lower each word
all_words_validation = [stem(w) for w in all_words if w not in ignore_words]
# remove duplicates and sort
all_words_validation = sorted(set(all_words_validation))
tags_validation = sorted(set(tags_validation))


print(len(xy_validation), "patterns validation")
print(len(tags_validation), "tags validation:", tags_validation)
print(len(all_words_validation), "unique stemmed words:", all_words_validation)


# create training data
X_validation = []
y_validation = []
for (pattern_sentence, tag) in xy_validation:
    # X: bag of words for each pattern_sentence
    bag = bag_of_words(pattern_sentence, all_words_validation)
    X_validation.append(bag)
    # y: PyTorch CrossEntropyLoss needs only class labels, not one-hot
    label = tags_validation.index(tag)
    y_validation.append(label)


X_validation = np.array(X_validation)
y_validation = np.array(y_validation)


############################
##### Hyper-parameters #####
############################

num_epochs = 1000
batch_size = 32
# learning_rate = 0.001
input_size = len(X_train[0])
# hidden_size = 8
output_size = len(tags)
print(input_size, output_size)


class ChatDataset(Dataset):

    def __init__(self):
        self.n_samples = len(X_train)
        self.x_data = X_train
        self.y_data = y_train

    # support indexing such that dataset[i] can be used to get i-th sample
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    # we can call len(dataset) to return the size
    def __len__(self):
        return self.n_samples


# train dataset
dataset = ChatDataset()
train_loader = DataLoader(dataset=dataset,
                          batch_size=batch_size,
                          shuffle=True,
                          num_workers=0)


# validation dataset
validation_dataset = ChatDataset()
validation_loader = DataLoader(dataset=validation_dataset,
                          batch_size=batch_size,
                          shuffle=True,
                          num_workers=0)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# model = NeuralNet(input_size, hidden_size, output_size).to(device)


# Loss and optimizer
# criterion = nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

'''
# Train the model
for epoch in range(num_epochs):
    for (words, labels) in train_loader:
        words = words.to(device)
        labels = labels.to(dtype=torch.long).to(device)
        # Forward pass
        outputs = model(words)
        loss = criterion(outputs, labels)
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # switch to evaluation mode
    model.eval()
    # deactivate the gradient calculation
    with torch.no_grad():
         validation_loss = 0
         for inputs, labels in validation_data:
             outputs = model(inputs)
             loss = criterion(outputs, labels)
             validation_loss += loss.item()
    # switch back to training mode
     model.train()

    if (epoch + 1) % 100 == 0:
         print(f'Epoch [{epoch + 1}/{num_epochs}]\n- Loss: {loss.item():.6f}\n- Validation Los: {loss.item():.6f}')
'''


###################################################################################
###################################################################################
###################################################################################

best_model = None
best_validation_loss = float('inf')

# Test different hyperparameters
for learning_rate in [0.001, 0.0001]:
    for hidden_size in [32, 48, 64, 128]:
        for weight_decay in [0, 0.01]:
            # Define the model, criterion, and optimizer
            model = NeuralNet(input_size, hidden_size, output_size)
            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

            # Train the model and evaluate it on the validation set
            model.train()
            for epoch in range(num_epochs):
                for (words, labels) in train_loader:
                    words = words.to(device)
                    labels = labels.to(dtype=torch.long).to(device)
                    # Forward pass
                    outputs = model(words)
                    loss = criterion(outputs, labels)
                    # Backward and optimize
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                # Evaluate the model on the validation set
                model.eval() # switch to evaluation mode
                with torch.no_grad(): # deactivate the gradient calculation
                    validation_loss = 0
                    for words, labels in validation_loader:
                        words = words.to(device)
                        labels = labels.to(dtype=torch.long).to(device)
                        outputs = model(words)
                        val_loss = criterion(outputs, labels)
                        #validation_loss += val_loss.item()
                model.train() # switch back to training mode

                # Print the training and validation loss
                if (epoch + 1) % 100 == 0:
                    print(f'Epoch [{epoch + 1}/{num_epochs}]\n- Loss: {loss.item():.8f}\n- Validation loss: {val_loss.item():.8f}\n- Parameters: learning rate={learning_rate} |  hidden size={hidden_size} | weight decay={weight_decay}')

            # Compare the validation loss with the best validation loss so far
            if val_loss.item() < best_validation_loss:
                best_model = model
                best_validation_loss = val_loss.item()

# Use the best hyperparameters
model = best_model


###################################################################################
###################################################################################
###################################################################################


print(f'Final loss: {loss.item():.8f}')


model_data = {
    "model_state": model.state_dict(),
    "input_size": input_size,
    "hidden_size": hidden_size,
    "output_size": output_size,
    "all_words": all_words,
    "tags": tags
}


FILE = f"{os.getcwd()}/Chatbot_w_spelling_correction/Files/model_data.pth"
torch.save(model_data, FILE)


print(f'Training complete.\nFile saved to: {FILE}')
