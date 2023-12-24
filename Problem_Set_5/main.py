import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.nn.functional as F
import numpy as np

import argparse


class LinearClassifier(nn.Module):
    # define a linear classifier
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # inchannels: dimenshion of input data. For example, a RGB image [3x32x32] is converted to vector [3 * 32 * 32], so dimenshion=3072
        # out_channels: number of categories. For CIFAR-10, it's 10
        self.layer = nn.Linear(in_channels, out_channels)
    def forward(self, x: torch.Tensor):
        return self.layer(x)


class FCNN(nn.Module):
    # def a full-connected neural network classifier
    def __init__(self, in_channels, hidden_channels, out_channels) -> None:
        super().__init__()
        # inchannels: dimenshion of input data. For example, a RGB image [3x32x32] is converted to vector [3 * 32 * 32], so dimenshion=3072
        # hidden_channels
        # out_channels: number of categories. For CIFAR-10, it's 10

        # full connected layer
        # activation function
        # full connected layer
        # ......
        self.layer = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, out_channels)
        )

    def forward(self, x: torch.Tensor): 
        return self.layer(x)
    

def svmloss(scores: torch.Tensor, label: torch.Tensor):
    '''
    compute SVM loss
    input:
        scores: output of model 
        label: true label of data
    return:
        svm loss
    '''
    # implement SVM loss function from sractch
    loss = 0
    for i in range(scores.shape[0]):
        for j in range(scores.shape[1]):
            if j != label[i]:
                loss += max(0, scores[i][j] - scores[i][label[i]] + 1)
    return loss

def crossentropyloss(logits: torch.Tensor, label: torch.Tensor):
    '''
    Object: implement Cross Entropy loss function
    input:
        logits: output of model, (unnormalized log-probabilities). shape: [batch_size, c]
        label: true label of data. shape: [batch_size]
    return: 
        cross entropy loss
    '''
    # implement cross entropy loss function from sractch
    loss = 0 
    # softmax function
    logits = torch.exp(logits) / torch.sum(torch.exp(logits), dim=1).view(-1, 1)
    # cross entropy loss
    for i in range(logits.shape[0]):
        loss += -torch.log(logits[i][label[i]])

    return loss


def train(model, loss_function, optimizer, scheduler, args):
    '''
    Model training function
    input: 
        model: linear classifier or full-connected neural network classifier
        loss_function: SVM loss of Cross-entropy loss
        optimizer: Adamw or SGD
        scheduler: step or cosine
        args: configuration
    '''
    # create dataset CIFAR-10
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transforms.ToTensor())
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transforms.ToTensor())

    # create dataloader
    trainloader = torch.utils.data.DataLoader(trainset, batch_size = 64, shuffle =True, num_workers = 2)
    testloader = torch.utils.data.DataLoader(testset, batch_size = 64, shuffle =False, num_workers = 2)
    # for-loop 
    for epoch in range(args.epoch):
        # train
        model.train()
        for i, data in 
            # get the inputs; data is a list of [inputs, labels]

            # zero the parameter gradients

            # forward

            # loss backward

            # optimize

        # adjust learning rate

        # test
            # forward

    # save checkpoint (Tutorial: https://pytorch.org/tutorials/recipes/recipes/saving_and_loading_a_general_checkpoint.html)

def test(model, loss_function, args):
    '''
    input: 
        model: linear classifier or full-connected neural network classifier
        loss_function: SVM loss of Cross-entropy loss
    '''
    # load checkpoint (Tutorial: https://pytorch.org/tutorials/recipes/recipes/saving_and_loading_a_general_checkpoint.html)
    # create testing dataset
    # create dataloader
    # test
        # forward

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='The configs')

    parser.add_argument('--run', type=str, default='train')
    parser.add_argument('--model', type=str, default='linear')
    parser.add_argument('--loss', type=str, default='svmloss')
    parser.add_argument('--optimizer', type=str, default='adamw')
    parser.add_argument('--scheduler', type=str, default='step')
    args = parser.parse_args()

    # create model
    if args.model == 'linear':
        model = LinearClassifier(3072, 10)
    elif args.model == 'fcnn':
        model = FCNN(3072, 100, 10)
    else: 
        raise AssertionError

    # create optimizer
    if args.optimizer == 'adamw':
        # create Adamw optimizer
        optimizer = optim.AdamW(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08)
    elif args.optimizer == 'sgd':
        # create SGD optimizer
        optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    else:
        raise AssertionError
    
    # create scheduler
    if args.scheduler == 'step':
        # create torch.optim.lr_scheduler.StepLR scheduler
        scheduler = lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    elif args.scheduler == 'cosine':
        # create torch.optim.lr_scheduler.CosineAnnealingLR scheduler
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=5, eta_min=0)
    else:
        raise AssertionError

    if args.run == 'train':
        train(model, eval(args.loss), optimizer, scheduler, args)
    elif args.run == 'test':
        test(model, eval(args.loss), args)
    else: 
        raise AssertionError
    
# You need to implement training and testing function that can choose model, optimizer, scheduler and so on by command, such as:
# python main.py --run=train --model=fcnn --loss=crossentropyloss --optimizer=adamw --scheduler=step


