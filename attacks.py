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

# Define the device and set the seed
cuda = torch.cuda.is_available()
device = torch.device("cuda" if cuda else "cpu")
np.random.seed(11785)
torch.manual_seed(11785)

# FGSM attack code
def fgsm_attack(image, epsilon, data_grad):
    # Collect the element-wise sign of the data gradient
    sign_data_grad = data_grad.sign()
    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_image = image + epsilon * sign_data_grad
    # Adding clipping to maintain [0,1] range
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    # Return the perturbed image
    return perturbed_image

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


def test_attack(model, attack_loader, epsilon):
    model.eval()
    correct = 0
    adv_examples = []

    start_time = time.time()
    for x, y in attack_loader:
        x, y = x.to(device), y.to(device)

        # We need the gradient of x, so set requires_grad to True
        x.requires_grad = True

        # Forward pass the data, log softmax the output to get softmax probability
        output = model(x)
        output = F.log_softmax(output, dim=1)

        init_pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability

        # If the initial prediction is wrong, no need to attack, just move on
        if init_pred.item() != y.item():
            continue

        # Calculate the loss
        loss = F.nll_loss(output, y)

        # Zero all existing gradients
        model.zero_grad()

        # Calculate gradients of model in backward pass
        loss.backward()

        # Collect the gradient of x
        data_grad = x.grad.data

        # Call FGSM Attack
        perturbed_data = fgsm_attack(x, epsilon, data_grad)

        # Re-classify the perturbed image
        output = model(perturbed_data)
        output = F.log_softmax(output, dim=1)

        # Check for success
        final_pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
        if final_pred.item() == y.item():
            correct += 1
            # Special case for saving 0 epsilon examples
            if (epsilon == 0) and (len(adv_examples) < 5):
                adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                adv_examples.append((init_pred.item(), final_pred.item(), adv_ex))
        else:
            # Save some adv examples for visualization later
            if len(adv_examples) < 5:
                adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                adv_examples.append((init_pred.item(), final_pred.item(), adv_ex))
    end_time = time.time()

    # Calculate final accuracy for this epsilon
    final_acc = correct/float(len(attack_loader))
    print("Epsilon: {}\tTest Accuracy = {} / {} = {}, time:{} s".format(epsilon, correct, len(attack_loader),
                                                                        final_acc, int(end_time - start_time)))

    # Return the accuracy and an adversarial example
    return final_acc, adv_examples

def main():
    # create test loader
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

    # load model which will be attacked
    model = torch.load("./model_files/model_files.pth")
    criterion = nn.CrossEntropyLoss()

    avg_loss, correct = test(model, test_loader, criterion)
    original_acc = correct / len(test_dataset)
    print("Original Acc:", original_acc)
    attacked_acc, adv_examples = test_attack(model, attack_loader, 0.2)
    print("Attacked Acc:", attacked_acc)

    plt.figure(figsize=(8, 10))
    for j in range(len(adv_examples)):
        plt.subplot(1, 5)
        plt.xticks([], [])
        plt.yticks([], [])
        if j == 0:
            plt.ylabel("Eps: {}".format(0.2), fontsize=14)
        orig, adv, ex = adv_examples[j]
        plt.title("{} -> {}".format(orig, adv))
        plt.imshow(ex, cmap="gray")
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()

