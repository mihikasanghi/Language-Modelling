import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.vocab import GloVe
from torchtext.data.utils import get_tokenizer
from collections import Counter
import re
import random
import math
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import nltk
from gensim.models import Word2Vec
import numpy as np
import pandas as pd

# Set random seed for reproducibility
torch.manual_seed(42)
random.seed(42)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Data cleaning and preprocessing
def clean_text(text):
    sentences = nltk.sent_tokenize(text)
    cleaned_sentences = [re.sub(r'([^A-Za-z0-9\s])', lambda m: ' ' if m.group(1) == '-' else '', sentence) for sentence in sentences]
    final_sentences = []
    for line in sentences:
        line = line.lower()
        if len(line.split()) >= 6:
            final_sentences.append(line)
    return final_sentences

def load_and_preprocess_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = file.read()
    data = data.lower()
    data = data.replace('\n', ' ')
    
    cleaned_data = clean_text(data)
    # tokenizer = get_tokenizer("basic_english")
    # tokens = tokenizer(cleaned_data)
    
    return cleaned_data

# Vocabulary creation
def create_vocab(train_dataset, min_freq=1):
    # counter = Counter(tokens)
    vocab_temp = set([word for line in train_dataset for word in line.split()])
    vocab = {'<PAD>': 0, '<UNK>': 1}
    for word in vocab_temp:
        # if count >= min_freq and word in glove.stoi:
        vocab[word] = len(vocab)
    return vocab

# Split data
def split_data(data, train_ratio=0.7, val_ratio=0.1):
    random.shuffle(data)
    train_size = int(len(data) * train_ratio)
    val_size = int(len(data) * val_ratio)
    train_data = data[:train_size]
    val_data = data[train_size:train_size+val_size]
    test_data = data[train_size+val_size:]
    return train_data, val_data, test_data

# Val, test dataset
def val_test_dataset(dataset, vocab):
    data = []
    for line in dataset:
        words = line.split()
        # print(words)
        new_words = ['<UNK>' if word not in vocab else word for word in words]
        # print(new_words)
        new_line = ' '.join(new_words)
        data.append(new_line)
    return data

def Embedding_Matrix(train_dataset, vocab, epochs=100):
    tokenized_sentences = [sentence.split() for sentence in train_dataset]
    vector_size = 300
    window = 5
    min_count = 1
    word2vec_model = Word2Vec(vector_size=vector_size, window=window, min_count=min_count, workers=4)
    word2vec_model.build_vocab(tokenized_sentences)
    word2vec_model.train(tokenized_sentences, total_examples=len(tokenized_sentences), epochs=100)
    mean_embedding = np.mean(word2vec_model.wv.vectors, axis=0)
    word2vec_model.wv.add_vector('<UNK>', mean_embedding)
    embedding_matrix = torch.zeros((len(vocab), vector_size))
    for word in vocab.keys():
        if word in word2vec_model.wv:
            embedding_matrix[vocab[word]] = torch.tensor(word2vec_model.wv[word])
        else:
            embedding_matrix[vocab[word]] = torch.randn(vector_size)
    return embedding_matrix

def create_dataframe(dataset, vocab, context_size=5):
    contexts = []
    targets = []
    
    for sentence in dataset:
        tokens = sentence.split()
        token_ids = [vocab.get(token, vocab['<UNK>']) for token in tokens]
        
        for i in range(len(token_ids) - context_size):
            context = token_ids[i:i+context_size]
            target = token_ids[i+context_size]
            
            if target != vocab['<UNK>']:
                contexts.append(context)
                targets.append(target)
    
    df = pd.DataFrame({
        'context': contexts,
        'target': targets
    })
    
    return df

# train_df = create_dataframe(train_data, vocab)
# val_df = create_dataframe(val_data, vocab)
# test_df = create_dataframe(test_data, vocab)

# print(f"Train dataset shape: {train_df.shape}")
# print(f"Validation dataset shape: {val_df.shape}")
# print(f"Test dataset shape: {test_df.shape}")

class NeuralLM(nn.Module):
    def __init__(self, vocab_size, embedding_matrix, embedding_dim=1500, hidden_dim=300, dropout_rate=0.5):
        super(NeuralLM, self).__init__()
        self.embeddings = torch.nn.Embedding.from_pretrained(embedding_matrix, freeze=False)
        # if pretrained_embeddings is not None:
        #     self.embeddings.weight.data.copy_(pretrained_embeddings)
        
        # First hidden layer
        self.hidden1 = nn.Linear(embedding_dim, hidden_dim)
        
        # Second hidden layer
        self.hidden2 = nn.Linear(hidden_dim, vocab_size)
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout_rate)
        
        # Softmax layer
        self.softmax = nn.LogSoftmax(dim=1)
    
    def forward(self, inputs):
        # inputs shape: (batch_size, 5)
        embeds = self.embeddings(inputs)  # Shape: (batch_size, 5, embedding_dim)
        embeds = embeds.view(embeds.size(0), -1)  # Shape: (batch_size, 5 * embedding_dim)
        
        hidden1_out = torch.relu(self.dropout(self.hidden1(embeds)))
        hidden2_out = self.hidden2(hidden1_out)
        # output = self.softmax(hidden2_out)
        
        return hidden2_out
    
class LanguageModelDataset(Dataset):
    def __init__(self, dataframe):
        self.dataframe = dataframe
    
    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, idx):
        context = torch.tensor(self.dataframe.iloc[idx]['context'])
        target = torch.tensor(self.dataframe.iloc[idx]['target'])
        
        return context, target


try:        
    # Load and preprocess data
    processed_data = load_and_preprocess_data('./Auguste_Maquet.txt')
    # split data
    train_data, val_data, test_data = split_data(processed_data)
    # Create vocabulary
    vocab = create_vocab(train_data)
    # Get embeddings
    embedding_matrix = Embedding_Matrix(train_data, vocab)
    # val and test dataset
    val_data = val_test_dataset(val_data, vocab)
    test_data = val_test_dataset(test_data, vocab)
    
    # Create DataFrames
    train_df = create_dataframe(train_data, vocab)
    val_df = create_dataframe(val_data, vocab)
    test_df = create_dataframe(test_data, vocab)

    # Create DataClasses
    train_dataclass = LanguageModelDataset(train_df)
    val_dataclass = LanguageModelDataset(val_df)
    test_dataclass = LanguageModelDataset(test_df)
    
    # Create model
    NeuralLM_model = NeuralLM(len(vocab), embedding_matrix, embedding_dim=1500, hidden_dim=300, dropout_rate=0.5)
    NeuralLM_model.to(device)

    
    # Create DataLoaders
    train_loader = DataLoader(train_dataclass, batch_size=8192, shuffle=True, num_workers=8)
    val_loader = DataLoader(val_dataclass, batch_size=8192, shuffle=False, num_workers=8)
    test_loader = DataLoader(test_dataclass, batch_size=8192, shuffle=False, num_workers=8)
    
    datapoint = next(iter(train_loader))
    context, target = datapoint
    context, target = context.to(device), target.to(device)
    output = NeuralLM_model(context)
    context.shape, target.shape, output.shape
    
except Exception as e:
    print(f"An error occurred: {str(e)}")
    raise  # This will print the full traceback for debugging

num_epochs = 10
learning_rate = 0.001
optimizer = torch.optim.Adam(NeuralLM_model.parameters(), lr=learning_rate)
criterion = torch.nn.CrossEntropyLoss(ignore_index=vocab['<UNK>'])

print("Id of <UNK>: ", vocab['<UNK>'])

for epoch in range(num_epochs):
    NeuralLM_model.train()
    train_loss = 0.0
    for context, target in train_loader:
        context = context.to(device)
        target = target.to(device)
        optimizer.zero_grad()
        
        output = NeuralLM_model(context)
        
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    NeuralLM_model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for context, target in val_loader:
            context = context.to(device)
            target = target.to(device)
            
            output = NeuralLM_model(context)

            loss = criterion(output, target)
            val_loss += loss.item()

    train_perplexity = math.exp(train_loss / len(train_loader))
    val_perplexity = math.exp(val_loss / len(val_loader))

    print(f"Epoch {epoch+1}/{num_epochs}:")
    print(f"  Train Loss: {train_loss/len(train_loader):.4f} | Train Perplexity: {train_perplexity:.4f}")
    print(f"  Val Loss: {val_loss/len(val_loader):.4f} | Val Perplexity: {val_perplexity:.4f}")
    
optimizers = ["Adam", "SGD"]
learning_rates = [0.001, 0.01]
hidden_dims = [100, 300]
dropouts = [0.3, 0.5]

def train_model(optimizer, learning_rate, hidden_dim, dropout):
    NeuralLM_model = NeuralLM(len(vocab), embedding_matrix, embedding_dim=1500, hidden_dim=300, dropout_rate=0.5)
    NeuralLM_model.to(device)
    
    if optimizer == "Adam":
        optimizer = torch.optim.Adam(NeuralLM_model.parameters(), lr=learning_rate)
    elif optimizer == "SGD":
        optimizer = torch.optim.SGD(NeuralLM_model.parameters(), lr=learning_rate)

    criterion = torch.nn.CrossEntropyLoss(ignore_index=vocab['<UNK>'])

    for epoch in range(10):
        NeuralLM_model.train()
        train_loss = 0.0
        for context, target in train_loader:
            context = context.to(device)
            target = target.to(device)
            optimizer.zero_grad()

            output = NeuralLM_model(context)

            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_perplexity = math.exp(train_loss / len(train_loader))

        print(f"Epoch {epoch+1}/{num_epochs}:")
        print(f"  Train Loss: {train_loss/len(train_loader):.4f} | Train Perplexity: {train_perplexity:.4f}")

    NeuralLM_model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for context, target in test_loader:
            context = context.to(device)
            target = target.to(device)

            output = NeuralLM_model(context)

            loss = criterion(output, target)
            test_loss += loss.item()

    test_perplexity = math.exp(test_loss / len(test_loader))

    return test_perplexity

results = []

for optimizer in optimizers:
    for learning_rate in learning_rates:
        for hidden_dim in hidden_dims:
            for dropout in dropouts:
                test_perplexity = train_model(optimizer, learning_rate, hidden_dim, dropout)
                results.append({
                    "optimizer": optimizer,
                    "learning_rate": learning_rate,
                    "hidden_dim": hidden_dim,
                    "dropout": dropout,
                    "test_perplexity": test_perplexity
                })
                
best_result = min(results, key=lambda x: x['test_perplexity'])
print(best_result)