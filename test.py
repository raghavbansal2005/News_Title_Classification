from calendar import EPOCH
import json
import os
from turtle import backward
import numpy as np
import torch
from tokenizers import BertWordPieceTokenizer
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
import nltk
from nltk.corpus import stopwords

class Classifier(nn.Module):
    def __init__(self, num_words, emb_size, outputs, dropout):
        super().__init__()
        self.word_embeddings = nn.Embedding(num_words, emb_size)
        self.lstm = nn.LSTM(emb_size, emb_size, batch_first = True)
        self.linear = nn.Linear(emb_size, outputs)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, lengths):

        x = self.word_embeddings(x)
        x_pack = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)

        outputs, (hn, cn) = self.lstm(x_pack)
        dense_outputs=self.linear(hn[-1])

        #outputs = nn.Softmax(dense_outputs)
        return dense_outputs



BATCH_SIZE = 16
EVAL_BATCH_SIZE = 64
LEARNING_RATE = 0.65
EPOCHS = 13

class Articles(torch.utils.data.Dataset):
    def __init__(self, json_file):
        super().__init__()
        with open(json_file, "r") as data_file:
            self.examples = json.loads(data_file.read())

    def __getitem__(self,idx):
        return self.examples[idx]
    
    def __len__(self):
        return len(self.examples)


    def map_items(
        self,
        tokenizer,
        category_to_id
    ):
        for idx, example in enumerate(self.examples):
            encoded = tokenizer.encode(example["headline"], add_special_tokens=False).ids
            self.examples[idx]["mapped_headlines"] = encoded
            self.examples[idx]["label"] = category_to_id.get(example["category"])

def collate_fn(examples):
    words = []
    labels = []
    for example in examples:
        words.append(example['headline'])
        labels.append(example['label'])

    batch_words = [tokenizer.encode(word) for word in words]
    num_words = [len(x) for x in batch_words]
    [entry.pad(max(num_words)) for entry in batch_words]
    batch_words = [entry.ids for entry in batch_words]
    batch_words = torch.tensor(batch_words, dtype=torch.long)
    lengths = torch.tensor(num_words, dtype=torch.long)

    labels_arr = torch.zeros((len(examples), len(category_to_id)), dtype=torch.float32)
    for i in range(len(labels)):
        labels_arr[i][labels[i]] = 1
    return batch_words, lengths, labels_arr


if __name__ == "__main__":

    train_data = Articles("splits/train.json")
    test_data = Articles("splits/test.json")
    val_data = Articles("splits/val.json")

    with open("category_to_id.json", 'r') as f:
        category_to_id = json.load(f)

    tokenizer = BertWordPieceTokenizer("bert-base-uncased-vocab.txt", lowercase=True)
    num_words = tokenizer.get_vocab_size()

    train_data.map_items(tokenizer, category_to_id)
    test_data.map_items(tokenizer, category_to_id)
    val_data.map_items(tokenizer, category_to_id)

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE, collate_fn=collate_fn)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=EVAL_BATCH_SIZE, collate_fn=collate_fn)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=EVAL_BATCH_SIZE, collate_fn=collate_fn)

    model = Classifier(num_words, 64, len(category_to_id), .1)

    loss = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.SGD(model.parameters(),lr=LEARNING_RATE)
    running_loss = 0

    for epoch in range(EPOCHS):
        print(f"Starting epoch {epoch}")
        model.train()
        torch.enable_grad()
        for idx, batch in enumerate(train_loader):
            optimizer.zero_grad()
            batch_words, lengths, labels = batch
            logits = model(batch_words, lengths)
            L = loss(logits, labels)
            L.backward()
            optimizer.step()
            running_loss += L.item()
            
        print("Total Loss: ", running_loss)
        running_loss = 0

        total_correct = 0
        torch.no_grad()
        model.eval()

        for idx, batch in enumerate(val_loader):
            batch_words, lengths, labels  = batch
            logits = model(batch_words, lengths)
            n_logits = logits.detach().numpy()
            n_labels = labels.numpy()
            max_indexes = np.argmax(n_logits, axis=1)
            actual_labels = np.argmax(n_labels, axis=1)
            total_correct += sum(np.equal(max_indexes, actual_labels) * 1)

        print("Accuracy: ", (total_correct/len(val_data)))   


    
    total_correct = 0
    for idx, batch in enumerate(test_loader):
        batch_words, lengths, labels = batch
        logits = model(batch_words, lengths)
        n_logits = logits.detach().numpy()
        n_labels = labels.numpy()
        max_indexes = np.argmax(n_logits, axis=1)
        actual_labels = np.argmax(n_labels, axis=1)
        total_correct += sum(np.equal(max_indexes, actual_labels) * 1)

    print("Final Accuracy: ", (total_correct/len(test_data)))
            


        


