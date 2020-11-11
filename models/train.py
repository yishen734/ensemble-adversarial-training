import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import time
from torch.utils import data
from torch.optim.lr_scheduler import ReduceLROnPlateau
from VGG import *

# Define the device and set the seed
cuda = torch.cuda.is_available()
device = torch.device("cuda" if cuda else "cpu")
np.random.seed(11785)
torch.manual_seed(11785)

# Hyper-parameters
num_classes = 10


def train(model, train_loader, optimizer, criterion):
    model.train()
    running_loss = 0.0
    total_correct = 0

    for x, y in train_loader:
        optimizer.zero_grad()
        x, y = x.to(device), y.to(device)

        # predict
        output = model(x)
        y_hat = output.argmax(dim=1)

        # loss
        loss = criterion(output, y)

        running_loss += loss.item()
        correct = (y_hat == y).float().sum()
        total_correct += correct.cpu().detach().numpy()

        # backward, update weights
        loss.backward()
        optimizer.step()
    avg_loss = running_loss / len(train_loader)
    return avg_loss, total_correct

def test(model, test_loader, criterion):
    model.eval()
    running_loss = 0.0
    total_correct = 0

    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)

            # predict
            output = model(x)
            y_hat = output.argmax(dim=1)

            # calculate loss and the number of correct predictions
            loss = criterion(output, y)
            running_loss += loss.item()
            correct = (y_hat == y).float().sum()
            total_correct += correct.cpu().detach().numpy()
    avg_loss = running_loss / len(test_loader)
    return avg_loss, total_correct

def main():
    # Define transforms
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=6),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomVerticalFlip(0.5),
        transforms.RandomAffine(degrees=20, scale=(.9, 1.1), shear=0),
        transforms.ToTensor()
    ])

    test_transform = transforms.Compose([transforms.ToTensor()])

    # Create dataset
    train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
    test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)

    # Create loaders
    train_loader = data.DataLoader(dataset=train_dataset,
                                   shuffle=True,
                                   batch_size=200,
                                   drop_last=True,
                                   num_workers=1,
                                   pin_memory=True)

    test_loader = data.DataLoader(dataset=test_dataset,
                                  shuffle=True,
                                  batch_size=200,
                                  drop_last=True,
                                  num_workers=1,
                                  pin_memory=True)

    # Define the model
    model = create_VGG('VGG19', num_classes)
    model = model.to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.005, weight_decay=0.0006, momentum=0.9)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.8, patience=3, verbose=True, min_lr=1.0e-04)
    criterion = nn.CrossEntropyLoss()

    # Start Training
    n_epochs = 300
    ls_train_loss = []
    ls_test_loss = []
    ls_train_acc = []
    ls_test_acc = []
    best_accuracy = -1

    for epoch in range(n_epochs):
        """--------- Training ---------"""
        start_time = time.time()
        avg_loss, total_correct = train(model, train_loader, optimizer, criterion)
        end_time = time.time()

        ls_train_loss.append(avg_loss)
        accuracy = total_correct / len(train_dataset)
        ls_train_acc.append(accuracy)
        print('=' * 60)
        print("Epoch:", epoch + 1)
        print('Train Loss: ', round(avg_loss, 5), 'Train Accuracy: ', round(accuracy, 5), 'Time: ',
              int(end_time - start_time), 's')

        """--------- Testing ---------"""
        start_time = time.time()
        avg_loss, total_correct = test(model, test_loader, criterion)
        end_time = time.time()

        ls_test_loss.append(avg_loss)
        accuracy = total_correct / len(test_dataset)
        ls_test_acc.append(accuracy)

        print('Test Loss: ', round(avg_loss, 5), 'Test Accuracy: ', round(accuracy, 5), 'Time: ',
              int(end_time - start_time), 's')
        print()

        # print the current learning rate
        for param_group in optimizer.param_groups:
            print("Current Learning rate:", param_group['lr'])

        # update and save the best model
        if accuracy > best_accuracy:
            print("Improve the best, update the best and save it")
            best_accuracy = accuracy
            torch.save(model, "../model_files/best_model.pth")
        scheduler.step(accuracy)

        vgg = vgg()



if __name__ == '__main__':
    main()



