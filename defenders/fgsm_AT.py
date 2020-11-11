from defenders.defender_base import DefenderBase

import sys
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils
import torch.optim as optim
import matplotlib.pyplot as plt
import time
from torch.utils import data
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from VGG import *

# Define the device and set the seed
cuda = torch.cuda.is_available()
device = torch.device("cuda" if cuda else "cpu")
np.random.seed(11785)
torch.manual_seed(11785)
num_classes = 10


class Mydataset(torch.utils.data.Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, index):
        X = self.X[index].float()
        Y = self.Y[index]
        return X, Y


def fgsm_attack(image, epsilon, data_grad):
    sign_data_grad = data_grad.sign()
    perturbed_image = image + epsilon * sign_data_grad
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    return perturbed_image


def train(model, train_loader, optimizer, criterion):
    model.train()
    running_loss = 0.0
    total_correct = 0

    for x, y in tqdm(train_loader):
        optimizer.zero_grad()
        x, y = x.to(device), y.to(device)

        # Predict
        output = model(x)
        y_hat = output.argmax(dim=1)

        # Loss
        loss = criterion(output, y)

        running_loss += loss.item()
        correct = (y_hat == y).float().sum()
        total_correct += correct.cpu().detach().numpy()

        # Backward, update weights
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

            # Predict
            output = model(x)
            y_hat = output.argmax(dim=1)

            # Calculate loss and the number of correct predictions
            loss = criterion(output, y)
            running_loss += loss.item()
            correct = (y_hat == y).float().sum()
            total_correct += correct.cpu().detach().numpy()
    avg_loss = running_loss / len(test_loader)
    return avg_loss, total_correct


def test_attack(model, attack_loader, epsilon):
    model.eval()
    correct = 0
    adv_examples = []

    start_time = time.time()
    cnt = 0
    for x, y in attack_loader:
        x, y = x.to(device), y.to(device)

        x.requires_grad = True

        output = model(x)
        output = F.log_softmax(output, dim=1)

        init_pred = output.max(1, keepdim=True)[1]
        if init_pred.item() != y.item():
            continue

        cnt += 1
        loss = F.nll_loss(output, y)
        model.zero_grad()
        loss.backward()

        data_grad = x.grad.data
        perturbed_data = fgsm_attack(x, epsilon, data_grad)

        output = model(perturbed_data)
        output = F.log_softmax(output, dim=1)

        final_pred = output.max(1, keepdim=True)[1]
        if final_pred.item() == y.item():
            correct += 1
            if (epsilon == 0) and (len(adv_examples) < 5):
                adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                adv_examples.append((init_pred.item(), final_pred.item(), adv_ex))
        else:
            if len(adv_examples) < 5:
                adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                adv_examples.append((init_pred.item(), final_pred.item(), adv_ex))
    end_time = time.time()

    final_acc = correct / cnt
    print("Epsilon: {}\tTest Accuracy = {} / {} = {}, time:{} s".format(epsilon, correct, cnt, final_acc,
                                                                        int(end_time - start_time)))

    return final_acc, adv_examples


def train_model(defense_dataset, defense_loader):
    # Define transforms
    test_transform = transforms.Compose([transforms.ToTensor()])
    test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)
    test_loader = data.DataLoader(dataset=test_dataset, shuffle=True, batch_size=200, drop_last=True, num_workers=1,
                                  pin_memory=True)

    # Define the model
    model = create_VGG('VGG19', num_classes)
    model = model.to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.005, weight_decay=0.0006, momentum=0.9)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.8, patience=3, verbose=True, min_lr=1.0e-04)
    criterion = nn.CrossEntropyLoss()

    # Start Training
    n_epochs = 20
    ls_train_loss = []
    ls_test_loss = []
    ls_train_acc = []
    ls_test_acc = []
    best_accuracy = -1

    for epoch in range(n_epochs):
        """--------- Training ---------"""
        start_time = time.time()
        avg_loss, total_correct = train(model, defense_loader, optimizer, criterion)
        end_time = time.time()

        ls_train_loss.append(avg_loss)
        accuracy = total_correct / len(defense_dataset)
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

        # Print the current learning rate
        for param_group in optimizer.param_groups:
            print("Current Learning rate:", param_group['lr'])

        # Update and save the best model
        if accuracy > best_accuracy:
            print("Improve the best, update the best and save it")
            best_accuracy = accuracy
            # torch.save(model, "/content/best_model_defense.pth")
        scheduler.step(accuracy)

    return model


# Generate defense model
def generate_adversarial_sample(model, attack_loader, epsilon):
    model.eval()
    correct = 0
    adv_image = []
    adv_label = []

    start_time = time.time()
    i = 0
    for x, y in tqdm(attack_loader):

        if i % 10000 == 0:
            print(i)
        i += 1

        x, y = x.to(device), y.to(device)

        x.requires_grad = True

        output = model(x)
        output = F.log_softmax(output, dim=1)

        init_pred = output.max(1, keepdim=True)[1]
        if init_pred.item() != y.item():
            continue

        loss = F.nll_loss(output, y)
        model.zero_grad()
        loss.backward()

        data_grad = x.grad.data

        perturbed_data = fgsm_attack(x, epsilon, data_grad)

        output = model(perturbed_data)
        output = F.log_softmax(output, dim=1)

        final_pred = output.max(1, keepdim=True)[1]

        adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
        adv_image.append(adv_ex)
        adv_label.append(init_pred.item())

    return adv_image, adv_label


class FGSM_AT(DefenderBase):

    def __init__(self):
        super(FGSM_AT, self).__init__()

    def train_adversarial_samples(self, model):
        train_transform = transforms.Compose([transforms.ToTensor()])
        train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True,
                                                     transform=train_transform)
        train_attack_loader = data.DataLoader(dataset=train_dataset,
                                              shuffle=True,
                                              batch_size=1,
                                              drop_last=True,
                                              num_workers=1,
                                              pin_memory=True)

        adv_images, adv_labels = generate_adversarial_sample(model, train_attack_loader, 0.01)

        for adv_data in train_dataset.data:
            adv_images.append(adv_data.transpose(2, 0, 1) / 255)
            adv_labels.extend(train_dataset.targets)

        defense_dataset = Mydataset(torch.tensor(adv_images), torch.tensor(adv_labels))
        defense_loader = torch.utils.data.DataLoader(dataset=defense_dataset,
                                                     shuffle=True,
                                                     num_workers=4,
                                                     batch_size=200,
                                                     pin_memory=True,
                                                     drop_last=True)

        defense_model = train_model(defense_dataset, defense_loader)

        return defense_model

    def eval_adversarial_samples(self, model):
        test_transform = transforms.Compose([transforms.ToTensor()])
        test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)
        test_loader = data.DataLoader(dataset=test_dataset,
                                      shuffle=True,
                                      batch_size=200,
                                      drop_last=True,
                                      num_workers=1,
                                      pin_memory=True)

        attack_loader = data.DataLoader(dataset=test_dataset,
                                        shuffle=True,
                                        batch_size=1,
                                        drop_last=True,
                                        num_workers=1,
                                        pin_memory=True)

        # Load the model to be attacked
        criterion = nn.CrossEntropyLoss()
        epsilon = 0.01
        avg_loss, correct = test(model, test_loader, criterion)
        original_acc = correct / len(test_dataset)
        attacked_acc, adv_examples = test_attack(model, attack_loader, epsilon)
        return original_acc, attacked_acc
