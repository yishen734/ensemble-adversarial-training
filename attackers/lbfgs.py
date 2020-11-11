# encoding: utf-8
"""
@author: Jingrong Feng
@contact: flo.fjr@gmail.com
@version: 0.1
@file: lbfgs_attacker.py
@time: 23/10/2020
"""
import os
from datetime import date
from typing import List

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

from attackers.attacker_base import AttackerBase

TODAY = date.today()
INDEX_TO_LABEL = {0: 'plane', 1: 'car', 2: 'bird', 3: 'cat', 4: 'deer',
                  5: 'dog', 6: 'frog', 7: 'horse', 8: 'ship', 9: 'truck'}
ATTACKER_CHECKPOINTS_DIR = "/gdrive/11-785/proj/checkpoints/attacker"
VIS_DIR = "/gdrive/11-785/proj/visualization"


class MyDataSet(Dataset):
    def __init__(self, dataset: Dataset):
        super(MyDataSet, self).__init__()
        dataset.transform = transforms.Compose([transforms.ToTensor()])
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img, label = self.dataset[index]
        return torch.tensor(index).long(), img.float(), torch.tensor(label).long()


class LBFGS(AttackerBase):
    def __init__(self, num_instances: int, input_size: List[int], c: float):
        """

        Args:
            num_instances: number of attackers instances (instances should be forwarded in the same order)
            input_size: [in_channels, width, height]
            c: a hyper parameter
        """
        super(LBFGS, self).__init__()
        self.perturbations = nn.Parameter(torch.Tensor(num_instances, *input_size), requires_grad=True)
        nn.init.uniform_(self.perturbations)
        self.c = c

        self.num_epoch = 40
        self.batch_size = 32
        self.lr = 1e-3

    def forward(self, index: torch.LongTensor, input: torch.FloatTensor):
        perturbation = self.perturbations[index]
        assert input.shape == perturbation.shape
        l2_norm = torch.norm(perturbation.reshape(input.shape[0], -1), dim=-1)
        reg = self.c * torch.mean(l2_norm)
        clamped_input = torch.clamp(input + perturbation.to(input.device), min=0., max=1.)
        return clamped_input, reg

    def train_adversarial_samples(self, model, dataset, **kwargs):
        test_set = MyDataSet(dataset)
        loader_args = dict(shuffle=True, batch_size=self.batch_size, num_workers=8, pin_memory=True)
        test_loader = DataLoader(dataset=test_set, **loader_args)

        # loss
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.parameters(), lr=self.lr)

        device = torch.device(f"cuda:0")
        self.to(device)
        model.to(device)

        self.train()
        model.eval()

        max_success_rate = 0.
        for epoch in range(self.num_epoch):
            training_loss = 0.
            run_step = 0
            batch = tqdm(test_loader, desc=f"Train epoch {epoch}", ncols=80, total=len(test_loader))
            for index, input, label in batch:
                index = index.to(device)
                input = input.to(device)
                # TODO: try other attack label
                attack_label = torch.ones_like(label).long().to(device)

                perturbed_input, reg = self.forward(index, input)
                output = model(perturbed_input)
                loss = criterion(output, attack_label) + reg
                optimizer.zero_grad()
                training_loss += loss.item()
                run_step += 1
                loss.backward()
                optimizer.step()

            success_rate = self.eval_adversarial_samples(model, dataset)
            print(f"epoch = {epoch}: loss = {training_loss / run_step}, success_rate = {success_rate}")
            # if success_rate > max_success_rate and success_rate > 0.999:
            #     checkpoint_path = os.path.join(
            #         ATTACKER_CHECKPOINTS_DIR,
            #         f"{TODAY}_{self.attacker_name}_{self.c}_{self.lr}_{success_rate * 100:.3f}.pt"
            #     )
            #     torch.save(self, checkpoint_path)
            #     print("Save model to checkpoint_path")
            #     max_success_rate = success_rate
            self.train()

    def eval_adversarial_samples(self, model, dataset):
        test_set = MyDataSet(dataset)
        loader_args = dict(shuffle=True, batch_size=self.batch_size, num_workers=8, pin_memory=True)
        test_loader = DataLoader(dataset=test_set, **loader_args)

        device = torch.device(f"cuda:0")
        self.to(device)
        model.to(device)

        self.eval()
        model.eval()

        success_cnt = 0.
        total_cnt = 0.
        with torch.no_grad():
            batch = tqdm(test_loader, desc=f"Evaluate", ncols=80, total=len(test_loader))
            for index, input, label in batch:
                index = index.to(device)
                input = input.to(device)
                attack_label = torch.ones_like(label).long().to(device)
                perturbed_input, _ = self.forward(index, input)
                output = model(perturbed_input)
                predictions = torch.argmax(output, dim=-1)
                success_cnt += (predictions == attack_label).sum().item()
                total_cnt += label.shape[0]

        success_rate = success_cnt / total_cnt
        return success_rate

    def plot_adversarial_samples(self, model, dataset):
        from PIL import Image

        test_set = MyDataSet(dataset)

        device = torch.device(f"cuda:0")
        self.to(device)
        model.to(device)

        self.eval()
        model.eval()

        with torch.no_grad():
            for idx in tqdm(range(len(test_set)), desc=f"Predict", ncols=80, total=len(test_set)):
                index, input, label = test_set[idx]
                ext_original_input = (input.permute(1, 2, 0).data.numpy() * 255).astype(np.uint8)
                img_original = Image.fromarray(ext_original_input)
                img_original.save(os.path.join(
                    VIS_DIR,
                    f"attacker-lbfgs-c{self.c}-{index}-{INDEX_TO_LABEL[label.item()]}.png"
                ))

                index = index.to(device)
                input = input.to(device)
                perturbed_input, _ = self.forward(index, input)
                output = model(perturbed_input.reshape(1, *perturbed_input.shape))
                prediction = torch.argmax(output)

                ext_perturbed_input = (perturbed_input.permute(1, 2, 0).cpu().data.numpy() * 255).astype(np.uint8)
                img_perturbed = Image.fromarray(ext_perturbed_input)
                img_perturbed.save(os.path.join(
                    VIS_DIR,
                    f"attacker-lbfgs-c{self.c}-{index}-"
                    f"{INDEX_TO_LABEL[label.item()]}-{INDEX_TO_LABEL[prediction.item()]}.png"
                ))
                if idx >= 10:
                    break

    @property
    def attacker_name(self):
        return "L-BFGS"
