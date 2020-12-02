# Author : Zhihao Wang
# Date : 2/12/2020

import torch
from VGG import create_VGG
import torch.nn.functional as F
import torchvision
from torch.utils import data
from torchvision import transforms
from attackers.fgsm import FGSM
from random_ensemble import generate_test_samples,cal_correlation,resume_model
from tqdm import tqdm
import numpy as np



transform = transforms.Compose([transforms.ToTensor()])
device = torch.device("cuda")

class Evaluator():
    def __init__(self,fgsm_eps=0.2,base_classifier_dir=None):
        self.attacker = FGSM(fgsm_eps)
        self.fgsm_eps = fgsm_eps
        self.base_classifier =  torch.load(base_classifier_dir)
        self.test_raw = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=None)
        self.test_loader = data.DataLoader(dataset=self.test_raw,
                                        shuffle=False,
                                        batch_size=1,
                                        drop_last=False,
                                        num_workers=1,
                                        pin_memory=True)
        print("generating adv samples")
        self.attacker.train_adversarial_samples(self.base_classifier,self.test_raw)
        self.attacker.eval_adversarial_samples(self.base_classifier,None)
        print("generate adv samples done")
        self.model_pool = None
        self.correlation = None

    def reset_model_pool(self,num_models=20,pool_dir="."):
        torch.cuda.empty_cache()
        try:
            self.model_pool = resume_model(num_models,pool_dir)
        except:
            raise ValueError("model pool reset failed, maybe wrong directory?")

        test_samples = generate_test_samples(self.test_raw,num_sample=50)
        self.correlation = cal_correlation(self.model_pool,test_samples)
        print(f"reset model pool done, correlation is {self.correlation}")

    def model_pool_eval(self):
        total = 0
        fail_attack = 0
        sample_inedx = 0
        for i,model in enumerate(self.model_pool):
            self.model_pool[i].to(device)
            self.model_pool[i].eval()

        for (x,y) in tqdm(self.test_loader):
            total += x.size(0)
            model_index = np.random.randint(0,len(self.model_pool))
            x, y = x.to(device), y.to(device)

            origin_y =  self.model_pool[model_index](x)
            origin_y = F.log_softmax(origin_y, dim=1)
            origin_y = origin_y.max(1, keepdim=True)[1]

            adv_x,_ = self.attacker.adv_samples[sample_inedx]
            adv_x = adv_x.to(device)
            adv_y = self.model_pool[model_index](adv_x)
            adv_y = F.log_softmax(adv_y, dim=1)
            adv_y = adv_y.max(1, keepdim=True)[1]

            attack_fail_num = (origin_y == adv_y).sum()
            fail_attack += attack_fail_num

            sample_inedx += 1

        defend_rate = torch.true_divide(fail_attack ,total)
        print(f"correlation as {self.correlation}, defense rate is {defend_rate}")

if __name__ == "__main__":
    base_classifier_dir = "./model_files/best_model.pth"
    num_models = 20
    pool_dir = "./"
    evaluator = Evaluator(fgsm_eps=0.2,base_classifier_dir=base_classifier_dir)
    evaluator.reset_model_pool(num_models=num_models,pool_dir=pool_dir)
    evaluator.model_pool_eval()
