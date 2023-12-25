import torch
import torch.nn.functional as F
from models import *
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import argparse
import wandb
import random, os
from tqdm import tqdm

def seed_all(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True

DIR = {'CIFAR10': './data'}

def GetCifar10(batchsize, attack=False):
    trans_t = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                  transforms.RandomHorizontalFlip(),
                                  transforms.ToTensor(),
                                  transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                                  ])
    if attack:
        trans = transforms.Compose([transforms.ToTensor()])
    else:
        trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    train_data = datasets.CIFAR10(DIR['CIFAR10'], train=True, transform=trans_t, download=True)
    test_data = datasets.CIFAR10(DIR['CIFAR10'], train=False, transform=trans, download=True) 
    train_dataloader = DataLoader(train_data, batch_size=batchsize, shuffle=True, num_workers=4)
    test_dataloader = DataLoader(test_data, batch_size=batchsize, shuffle=False, num_workers=4)
    return train_dataloader, test_dataloader

def evaluate(model, test_loader, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = F.cross_entropy(outputs, labels)
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    return total_loss / len(test_loader), 100.0 * correct / total


def train(model, args):
    '''
    Model training function
    input: 
        model: linear classifier or full-connected neural network classifier
        args: configuration
    '''
    # create wandb project
    wandb.init(project="HW6",config=args)
    wandb.runname = f'{args.model}_{args.lr}_{args.epochs}'
    # create dataset, data augmentation for cifar10
    train_loader, test_loader = GetCifar10(args.batch_size)
    # create optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    # create scheduler 
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.5)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    # train
    best_acc = 0.0
    best_model_state = None
    for epoch in tqdm(range(args.epochs)):
        model.train()
        for i, data in enumerate(train_loader):
            images, labels = data
            images = images.to(args.device)
            labels = labels.to(args.device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = F.cross_entropy(outputs, labels)
            loss.backward()
            optimizer.step()
            # wandb.log({"loss": loss.item()})
        scheduler.step()
        train_loss, train_acc = evaluate(model, train_loader, args.device)
        test_loss, test_acc = evaluate(model, test_loader, args.device)
        # log the wandb
        wandb.log({"epoch": args.epochs,"train_loss": train_loss, "train_acc": train_acc, "test_loss": test_loss, "test_acc": test_acc})
        print(f'Epoch {epoch+1}/{args.epochs}, train loss: {train_loss:.4f}, train acc: {train_acc:.2f}, test loss: {test_loss:.4f}, test acc: {test_acc:.2f}')

        if test_acc > best_acc:
            best_acc = test_acc
            best_model_state = model.state_dict()
    # save checkpoint (Tutorial: https://pytorch.org/tutorials/recipes/recipes/saving_and_loading_a_general_checkpoint.html)
    if best_model_state is not None:
        if not os.path.exists('./model_best'):
            os.makedirs('./model_best')
        torch.save(best_model_state, f'./model_best/{args.model}_best.pth')

    print(f'Finish training for {args.model}')
    print(f'Best test acc: {best_acc}')
    wandb.finish()

def test(model, args):
    '''
    input: 
        model: linear classifier or full-connected neural network classifier
        args: configuration
    '''
    _, test_loader = GetCifar10(args.batch_size)
    # load model state dict
    model.load_state_dict(torch.load(f'./model_best/{args.model}_best.pth'))
    # evaluate
    test_loss, test_acc = evaluate(model, test_loader, args.device)
    print(f'Finish testing for {args.model}')
    print(f'Test acc: {test_acc}')

def main():
    parser = argparse.ArgumentParser(description='The configs')

    parser.add_argument('--model', type=str, default='VGG', help='VGG, ResNet, ResNext')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--lr', type=float, default=4e-4, help='learning rate')
    parser.add_argument('--epochs', type=int, default=50, help='epochs')
    parser.add_argument('--device', type=str, default='cuda', help='cpu, cuda')
    args = parser.parse_args()
    # args.device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    seed_all()
    print("-"*10)
    print(f'{args.model} with lr={args.lr}, epochs={args.epochs} begin training!')
    print("-"*10)
    # train / test
    # load model
    if args.model == 'VGG':
        model = VGG16()
    elif args.model == 'ResNet':
        model = ResNet50()
        # model = ResNet34()
    elif args.model == 'ResNext':
        model = ResNeXt50_32x4d()
    else:
        raise NotImplementedError(f'{args.model} is not implemented!')
    model.to(args.device)
    # train
    train(model, args)
    # test
    # test(model, args)
    print("-"*10)
    print(f'{args.model} with lr={args.lr}, epochs={args.epochs} finish training!')
    print("-"*10)

if __name__ == '__main__':
    main()