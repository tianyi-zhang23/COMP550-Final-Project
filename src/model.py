# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F


# %% Text CNN model
class textCNN(nn.Module):

    def __init__(self, vocab_built, emb_dim, dim_channel, kernel_wins, dropout_rate, num_class):
        super(textCNN, self).__init__()
        # load pretrained embedding in embedding layer.
        emb_dim = vocab_built.vectors.size()[1]
        pretrained_embeddings = vocab_built.vectors
        pretrained_embeddings = torch.cat((torch.zeros(1,pretrained_embeddings.shape[1]),pretrained_embeddings))
        self.embed = nn.Embedding(len(pretrained_embeddings), emb_dim, padding_idx=0)
        self.embed.weight.data.copy_(pretrained_embeddings)

        # Convolutional Layers with different window size kernels
        self.convs = nn.ModuleList([nn.Conv2d(1, dim_channel, (w, emb_dim)) for w in kernel_wins])
        # Dropout layer
        self.dropout = nn.Dropout(dropout_rate)

        # FC layer
        self.fc = nn.Linear(len(kernel_wins) * dim_channel, num_class)

    def forward(self, x):
        emb_x = self.embed(x)
        emb_x = emb_x.unsqueeze(1)

        con_x = [conv(emb_x) for conv in self.convs]

        pool_x = [F.max_pool1d(x.squeeze(-1), x.size()[2]) for x in con_x]

        fc_x = torch.cat(pool_x, dim=1)

        fc_x = fc_x.squeeze(-1)

        fc_x = self.dropout(fc_x)
        logit = self.fc(fc_x)
        return logit


def train(model, device, train_loader, optimizer, epoch, max_epoch):
    model.train()
    corrects, train_loss = 0.0, 0

    for idx, (target,text) in enumerate(train_loader):
        optimizer.zero_grad()
        logit = model(text)
        loss = F.cross_entropy(logit, target)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        result = torch.max(logit, 1)[1]
        corrects += (result.view(target.size()).data == target.data).sum()

    size = len(train_loader.dataset)
    train_loss /= size
    accuracy = 100.0 * corrects / size

    return train_loss, accuracy


def valid(model, device, test_loader):
    model.eval()
    corrects, test_loss = 0.0, 0
    for idx, (target, text) in enumerate(test_loader):
        logit = model(text)
        loss = F.cross_entropy(logit, target)

        test_loss += loss.item()
        result = torch.max(logit, 1)[1]
        corrects += (result.view(target.size()).data == target.data).sum()

    size = len(test_loader.dataset)
    test_loss /= size
    accuracy = 100.0 * corrects / size

    return test_loss, accuracy