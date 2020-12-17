from VGG import create_VGG
import torch
import torchvision
from torch.utils import data
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
from torchvision import transforms
import random
from torch import nn
import torch.nn.functional as F
import math
import sys
import os

device = torch.device("cuda")

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

def sample_train_data(sample_rate, train_dataset):
    n_train = len(train_dataset)
    split = int(n_train * sample_rate)
    indices = list(range(n_train))
    random.shuffle(indices)
    randomsampler = torch.utils.data.sampler.SubsetRandomSampler(indices[:split])
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size= 1, pin_memory=True, sampler = randomsampler)

    return train_loader

def fgsm_attack(image, epsilon, data_grad):
    sign_data_grad = data_grad.sign()
    perturbed_image = image + epsilon * sign_data_grad
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    return perturbed_image
    
def generate_adversarial_sample(model, attack_loader, epsilon):
    model.eval()
    correct = 0
    adv_image = []
    adv_label = []
    
    num = 0
    i = 0
    for x, y in attack_loader:
        if num == 7500:
          break
        if i % 2000 == 0:
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
        num += 1
        adv_image.append(adv_ex)
        adv_label.append(init_pred.item())
    
    print(num)
    
    return adv_image, adv_label

def train_model_defense(dataloader):
    model = create_VGG('VGG19', 10).to(device)

    print("train model {}".format(i+1))
    model = train(model, dataloader)
    
    return model

# TODO: 用sampled data训练单个模型
def train(model, train_loader):
    n_epoches = 50
    optimizer = optim.SGD(model.parameters(), lr=0.005, weight_decay=0.0006, momentum=0.9)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.8, patience=3, min_lr=1.0e-04)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(n_epoches):
    
        avg_loss = train_epoch(model, train_loader, optimizer, criterion)
        #acc = test(model, test_loader)
        if epoch % 10 == 0:
          print("Epoch:", epoch)
          print('Train Loss: ', round(avg_loss, 5))
        #print('acc: {}'.format(acc))
        scheduler.step(avg_loss)
    
    return model
        

def train_epoch(model, train_loader, optimizer, criterion):
    model.train()
    running_loss = 0.0
    total_correct = 0

    scaler = torch.cuda.amp.GradScaler()
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            output = model(x)
            loss = criterion(output, y)
            running_loss += loss.item()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    
    avg_loss = running_loss / len(train_loader)
    return avg_loss

def resume_model(num_model):
    model_pool = []
    for i in range(num_model):
        model = create_VGG('VGG19', 10).to(device)
        model.load_state_dict(torch.load('/content/drive/MyDrive/Project/models/30_sample_rate/model{}_parameter.pkl'.format(i + 1)))
        model_pool.append(model)
  
    return model_pool

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


def main(sample_rate, input_dir, output_dir):

    Transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=Transform)


    # get 20 AT-Ensemble models
    for i in range(20):
        print("model_{}".format(i+1))
        model = create_VGG('VGG19', 10).to(device)
        model.load_state_dict(torch.load(os.path.join(input_dir , 'model{}_parameter.pkl'.format(i+1))))

        # train ensemble model
        train_loader = sample_train_data(sample_rate, train_dataset)

        # generate adversarial samples
        adv_images, adv_labels = generate_adversarial_sample(model, train_loader, 0.01)

        # mix adversarial samples and benign samples
        for data, labels in train_loader:
            adv_images.extend(data.numpy())
            adv_labels.extend(labels.numpy())
            print(len(adv_images))

        # mixed dataset and dataloader
        defense_dataset = Mydataset(torch.tensor(adv_images), torch.tensor(adv_labels))
        defense_loader = torch.utils.data.DataLoader(dataset = defense_dataset, shuffle= True, num_workers=4, batch_size = 200, pin_memory = True)

        # train AT-Ensemble models
        model = train_model_defense(defense_loader)
        torch.save(model.state_dict(), os.path.join(output_dir, 'model{}_parameter.pkl'.format(i+1)))
        print("save")

if __name__ == '__main__':
    SAMPLE_RATE = float(sys.argv[1])
    INPUT_DIR = sys.argv[2]
    OUTPUT_DIR = sys.argv[3]
    main(SAMPLE_RATE, INPUT_DIR, OUTPUT_DIR)