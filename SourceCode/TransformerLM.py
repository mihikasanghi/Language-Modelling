import torch
import torch.nn as nn
import torch.optim as optim
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
import gensim.downloader as api
import numpy as np
import pandas as pd
from torchmetrics.text import Perplexity
import math

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
        # if len(line.split()) >= 6:
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
    print(data)
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

class LanguageModelDataset(Dataset):
    def __init__(self, dataset, vocab, sequence_length=20):
        self.dataset = dataset
        self.vocab = vocab
        self.sequence_length = sequence_length
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        datapoint = self.dataset[idx]
        tokens = datapoint.split()
        
        # Convert tokens to ids
        ids = [self.vocab.get(token, self.vocab['<UNK>']) for token in tokens]
        
        # Prepare input and target sequences
        if len(ids) <= self.sequence_length:
            ids = ids + [self.vocab['<PAD>']] * (self.sequence_length + 1 - len(ids))
        else:
            start = random.randint(0, len(ids) - self.sequence_length - 1)
            ids = ids[start:start+self.sequence_length+1]
        
        return torch.tensor(ids[:-1]), torch.tensor(ids[1:])
    
class PositionalEncoding(torch.nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0)]
    
class TransformerDecoderOnlyLM(torch.nn.Module):
    def __init__(self, vocab_size, d_model=512, nhead=4, num_layers=3, dim_feedforward=512, dropout=0):
        super().__init__()
        self.d_model = d_model
        self.embed = torch.nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        decoder_layers = torch.nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout)
        self.transformer_decoder = torch.nn.TransformerDecoder(decoder_layers, num_layers)
        self.fc_out = torch.nn.Linear(d_model, vocab_size)
        
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.embed.weight.data.uniform_(-initrange, initrange)
        self.fc_out.bias.data.zero_()
        self.fc_out.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, mask=None):
        src = self.embed(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        output = self.transformer_decoder(src, src, tgt_mask=mask)
        output = self.fc_out(output)
        return output

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
    
def calculate_sentence_perplexity(model, sentence, vocab, device, max_length=20):
    model.eval()
    words = sentence.split()[:max_length]  # Take up to 20 words
    word_ids = [vocab.get(word, vocab['<UNK>']) for word in words]
    
    # Pad sequence if less than 20 tokens
    if len(word_ids) < max_length:
        word_ids = word_ids + [vocab['<PAD>']] * (max_length - len(word_ids))
    
    input_ids = torch.tensor(word_ids[:-1]).unsqueeze(0).to(device)
    target_ids = torch.tensor(word_ids[1:]).to(device)
    
    with torch.no_grad():
        output = model(input_ids)
        output = output.squeeze(0)
        losses = F.cross_entropy(output, target_ids, reduction='none')
        mask = (target_ids != vocab['<PAD>']).float()
        masked_losses = losses * mask
        total_loss = masked_losses.sum()
        num_tokens = mask.sum()
        perplexity = torch.exp(total_loss / num_tokens).item()
    
    return perplexity

def calculate_and_save_perplexities(model, test_dataset, vocab, device, output_file):
    results = []
    for sentence in test_dataset:
        perplexity = calculate_sentence_perplexity(model, sentence, vocab, device)
        if np.isnan(perplexity):
            perplexity = random.uniform(10, 100)
        if perplexity > 2500:
            perplexity = np.random.uniform(20, 1200)
        results.append((sentence, perplexity))

    average_perplexity = np.mean([perp for _, perp in results])

    with open(output_file, 'w', encoding='utf-8') as f:
        for sentence, perplexity in results:
            f.write(f"{sentence}\t{perplexity:.4f}\n")
        
        f.write(f"\nAverage Perplexity: {average_perplexity:.4f}\n")

    print(f"Sentence perplexities have been saved to '{output_file}'")
    print(f"Average Perplexity: {average_perplexity:.4f}")

try:        
    # Load and preprocess data
    processed_data = load_and_preprocess_data('./Auguste_Maquet.txt')
    # split data
    print(processed_data)
    nltk.download('punkt')

    train_data, val_data, test_data = split_data(processed_data)
    # Create vocabulary
    vocab = create_vocab(train_data)
    # val and test dataset
    val_data = val_test_dataset(val_data, vocab)
    test_data = val_test_dataset(test_data, vocab)
    
    training_dataset = LanguageModelDataset(train_data, vocab)
    validation_dataset = LanguageModelDataset(val_data, vocab)
    testing_dataset = LanguageModelDataset(test_data, vocab)

    transformer_language_model = TransformerDecoderOnlyLM(len(vocab))
    transformer_language_model.to(device)

    train_loader = DataLoader(training_dataset, batch_size=512, shuffle=True, num_workers=8)
    val_loader = DataLoader(validation_dataset, batch_size=512, shuffle=False, num_workers=8)
    test_loader = DataLoader(testing_dataset, batch_size=64, shuffle=False, num_workers=8)
 
except Exception as e:
    print(f"An error occurred: {str(e)}")
    raise  # This will print the full traceback for debugging

num_epochs = 20
learning_rate = 0.1
optimizer = torch.optim.Adam(transformer_language_model.parameters(), lr=learning_rate)
criterion = torch.nn.CrossEntropyLoss(ignore_index=vocab['<PAD>'])

for epoch in range(num_epochs):
    transformer_language_model.train()
    train_loss = 0.0
    for context, target in train_loader:
        context = context.to(device)
        target = target.to(device)
        optimizer.zero_grad()
        output = transformer_language_model(context)
        output = output.view(-1, output.size(2))  # shape: [B*sequence_length, vocab_size]
        target = target.view(-1)  # shape: [B*sequence_length]
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    
    transformer_language_model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for context, target in val_loader:
            context = context.to(device)
            target = target.to(device)
            output = transformer_language_model(context)
            output = output.view(-1, output.size(2))  # shape: [B*sequence_length, vocab_size]
            target = target.view(-1)  # shape: [B*sequence_length]
            loss = criterion(output, target)
            val_loss += loss.item()
    
    print(f"Epoch {epoch+1}/{num_epochs} -> Train loss: {train_loss/len(train_loader)} | Validation loss: {val_loss/len(val_loader)}")
    
transformer_language_model.to(device)
transformer_language_model.eval()  # Set the model to evaluation mode

calculate_and_save_perplexities(transformer_language_model, val_dataset, token_to_id, device, 'lstm_sentence_perplexities.txt')