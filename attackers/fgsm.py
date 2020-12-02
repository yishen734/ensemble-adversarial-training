"""
@author: Yi Shen
@contact: yishen734@gmail.com
@version: 0.1
@file: fgsm.py
@time: 27/10/2020
"""

from attackers.attacker_base import AttackerBase
import numpy as np
import torch
import torch.nn.functional as F
import time
from torch.utils import data
from tqdm import tqdm
import torchvision.transforms as transforms

# Define the device and set the seed
cuda = torch.cuda.is_available()
device = torch.device("cuda" if cuda else "cpu")
np.random.seed(11785)
torch.manual_seed(11785)


class FGSM(AttackerBase):
    def __init__(self, epsilon):
        super(FGSM, self).__init__()
        self.epsilon = epsilon  # control how strong the perturbation is
        self.adv_samples = []

    # FGSM attack code
    def fgsm_attack(self, image, epsilon, data_grad):
        # Collect the element-wise sign of the data gradient
        sign_data_grad = data_grad.sign()

        # Create the perturbed image by adjusting each pixel of the input image
        perturbed_image = image + epsilon * sign_data_grad

        # Adding clipping to maintain [0,1] range
        perturbed_image = torch.clamp(perturbed_image, 0, 1)
        return perturbed_image

    def train_adversarial_samples(self, model, dataset, **kwargs):
        """
        Train adversarial examples
        Args:
            model: the pre-trained model which will be attacked
            dataset: the CIFAR10 dataset
        """
        test_transform = transforms.Compose([transforms.ToTensor()])
        dataset.transform = test_transform

        # Generate the attack data loader
        attack_loader = data.DataLoader(dataset=dataset,
                                        shuffle=True,
                                        batch_size=1,
                                        drop_last=True,
                                        num_workers=1,
                                        pin_memory=True)

        # Start to generate adversarial samples
        model.eval()
        adv_samples = []
        start_time = time.time()
        for x, y in tqdm(attack_loader):
            x, y = x.to(device), y.to(device)

            # We need the gradient of x, so set requires_grad to True
            x.requires_grad = True

            # Forward pass the data, log softmax the output to get softmax probability
            output = model(x)
            output = F.log_softmax(output, dim=1)
            # init_pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability

            # If the initial prediction is wrong, no need to attack, just move on
            # if init_pred.item() != y.item():
            #     continue

            # Calculate the loss
            loss = F.nll_loss(output, y)

            # Zero all existing gradients
            model.zero_grad()

            # Calculate gradients of model in backward pass
            loss.backward()

            # Collect the gradient of x
            data_grad = x.grad.data

            # Call FGSM Attack
            adv_sample = self.fgsm_attack(x, self.epsilon, data_grad)
            adv_samples.append((adv_sample, y))
        self.adv_samples = adv_samples
        end_time = time.time()
        print("Generate adversarial samples, epsilon: {}, time:{} s".format(self.epsilon, int(end_time - start_time)))

    def eval_adversarial_samples(self, model, dataset):
        """
        Evaluate the effects of adversarial examples
        Args:
            model: the pre-trained model which will be evaluated on
            dataset: None - For FGSM, the evaluation stage doesn't need dataset.
                     Necessary data has been saved in self.adv_samples
        """

        model.eval()
        wrong = 0
        start_time = time.time()
        for adv_sample, y in self.adv_samples:
            adv_sample, y = adv_sample.to(device), y.to(device)

            # Re-classify the perturbed image
            output = model(adv_sample)
            output = F.log_softmax(output, dim=1)

            # Check for success
            adv_pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
            if adv_pred.item() != y.item():
                wrong += 1
        end_time = time.time()

        # Calculate attack success rate for this epsilon
        if len(self.adv_samples) != 0:
            attack_rate = wrong / float(len(self.adv_samples))
        else:
            attack_rate = 0
        print("Epsilon: {}\tAttack Success Rate = {} / {} = {}, time:{} s".format(self.epsilon, wrong,
                                                                                  len(self.adv_samples),
                                                                                  attack_rate,
                                                                                  int(end_time - start_time)))
        return attack_rate
