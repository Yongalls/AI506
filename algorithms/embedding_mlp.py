import torch
import torch.nn as nn
import numpy as np
import random
import neptune.new as neptune
from utils import AverageMeter

NUM_ING = 6714
torch.manual_seed(1)
np.random.seed(1)
random.seed(1)

class Completion(nn.Module):
    def __init__(self, input_size, n_hidden, n_class):
        super(Completion, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_size, n_hidden),
            torch.nn.BatchNorm1d(n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_class),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = x.type(torch.float32)
        x = self.network(x)
        return x
    

class TrainDataset(torch.utils.data.Dataset):
    def __init__(self, train_list, embedding):
        self.input = []
        self.target = [] # missing ingredient
        self.embedding = embedding
        
        for data in train_list:
            nodes = [int(node) for node in data[0:-1]]
            
            for node in nodes:
                other_nodes = [nd for nd in nodes if nd != node]
                if len(other_nodes) == 0:
                    continue
                self.input.append(other_nodes)
                self.target.append(node)                

    def __len__(self):
        return len(self.input)

    def __getitem__(self, idx):
        nodes = self.input[idx]
        if len(nodes) > 1 and random.random() < 0.2:
            nodes = random.sample(nodes, k = len(nodes) - 1)
        if len(self.embedding[0]) == NUM_ING:
            inp = np.sum(self.embedding[nodes][:], axis=0)
        else:
            inp = np.average(self.embedding[nodes][:], axis=0)
        target = self.target[idx]
        return inp, target
    
class ValDataset(torch.utils.data.Dataset):
    def __init__(self, val_question, val_answer, embedding):
        self.input = []
        self.target = [] # missing ingredient
        
        for i in range(len(val_question)):
            nodes = [int(node) for node in val_question[i]]
            if len(embedding[0]) == NUM_ING:
                example = np.sum(embedding[nodes][:], axis=0)
            else:
                example = np.average(embedding[nodes][:], axis=0)
            self.input.append(example)
            self.target.append(int(val_answer[i][0]))

    def __len__(self):
        return len(self.input)

    def __getitem__(self, idx):
        inp = self.input[idx]
        target = self.target[idx]
        return inp, target
    
class TestDataset(torch.utils.data.Dataset):
    def __init__(self, test_question, embedding):
        self.input = []
        
        for data in test_question:
            nodes = [int(node) for node in data]
            if len(embedding[0]) == NUM_ING:
                example = np.sum(embedding[nodes][:], axis=0)
            else:
                example = np.average(embedding[nodes][:], axis=0)
            self.input.append(example)

    def __len__(self):
        return len(self.input)

    def __getitem__(self, idx):
        inp = self.input[idx]
        return inp

def run(trains, val_cpt_q, val_cpt_a, test_cpt_q, embedding, batch_size=256, hidden_size=512, epochs=150, lr=0.001, use_neptune=False):
    if use_neptune:
        nep = neptune.init(
            project="lym7505/AI506", api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJkNzVlY2RiNi0xNGI4LTRlOWEtOWY0ZC05MDMyNzZiN2U5YWUifQ==",
        )
    
    train_loader = torch.utils.data.DataLoader(
        TrainDataset(trains, embedding), 
        batch_size=batch_size, 
        shuffle=True
    )
    
    val_loader = torch.utils.data.DataLoader(
        ValDataset(val_cpt_q, val_cpt_a, embedding), 
        batch_size=batch_size, 
        shuffle=False
    )
    
    test_loader = torch.utils.data.DataLoader(
        TestDataset(test_cpt_q, embedding), 
        batch_size=batch_size, 
        shuffle=False
    )
    
    input_size = embedding.shape[1]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = Completion(input_size=input_size, n_hidden=hidden_size, n_class=NUM_ING)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best = 0
    for epoch in range(epochs):
        losses = AverageMeter()
        train_acc = 0
        num_datas = 0
        for i, (input_data, label) in enumerate(train_loader):
            num_datas += input_data.size(0)
            input_data = input_data.to(device)
            label = label.to(device)
            output = model(input_data)
            loss = criterion(output, label)
            pred = torch.argmax(output, dim=1)

            losses.update(loss.item(), input_data.size(0))
            pred = pred.detach().cpu().numpy()
            label = label.detach().cpu().numpy()
            train_acc += (pred == label).sum()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        val_acc = 0
        predictions = []
        with torch.no_grad():
            for i, (input_data, label) in enumerate(val_loader):
                input_data = input_data.to(device)
                label = label.to(device)
                output = model(input_data)
                pred = torch.argmax(output, dim=1)
                pred = pred.detach().cpu().numpy()
                predictions.append(pred)
                label = label.detach().cpu().numpy()
                val_acc += (pred == label).sum()

            if val_acc > best:
                best = val_acc
                val_pred = np.concatenate(predictions)

        if use_neptune:
            nep["train/loss"].log(losses.avg)
            nep["train/acc"].log(train_acc / num_datas * 100)
            nep["val/acc"].log(val_acc / len(val_cpt_q) * 100)
        print("Epoch %d // train loss: %.3f, train acc: %.3f, val acc: %.3f" % 
              (epoch, losses.avg, train_acc / num_datas * 100, val_acc / len(val_cpt_q) * 100))

    print("best accuracy: ", best / len(val_cpt_q) * 100)
    
    predictions = []
    with torch.no_grad():
        for i, input_data in enumerate(test_loader):
            input_data = input_data.to(device)
            output = model(input_data)
            pred = torch.argmax(output, dim=1)
            pred = pred.detach().cpu().numpy()
            predictions.append(pred)
        test_pred = np.concatenate(predictions)
    return train_acc / num_datas * 100, val_acc / len(val_cpt_q) * 100, val_pred, test_pred