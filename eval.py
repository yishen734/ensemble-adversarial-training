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
import copy
device = torch.device("cuda")

class Evaluator():
    def __init__(self,fgsm_eps=0.2,base_classifier_dir=None,left_aside_model_dir=None):
        self.attacker = FGSM(fgsm_eps)
        self.fgsm_eps = fgsm_eps
        self.batch_size = 16
        self.base_classifier = torch.load(base_classifier_dir)
        self.left_aside_model = create_VGG('VGG19', 10)
        self.left_aside_model.load_state_dict(torch.load(left_aside_model_dir))
        self.test_raw = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=None)
        test_transform = transforms.Compose([transforms.ToTensor()])
        self.test_raw.transform = test_transform
        self.test_loader = data.DataLoader(dataset=self.test_raw,
                                        shuffle=False,
                                        batch_size=self.batch_size,
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
        print(f"reset model pool done,pearson correlation: {self.correlation[0]:.4f}, "
              f"cosine correlation: {self.correlation[1]:.4f}, "
              f"average pearson: {self.correlation[2]:.6f},"
              f"average cosine: {self.correlation[3]:.6f}")

    def model_pool_eval(self):
        for i,model in enumerate(self.model_pool):
            self.model_pool[i].to(device)
            self.model_pool[i].eval()
        self.left_aside_model.to(device)
        self.left_aside_model.eval()
        true_y = []
        left_aside_origin_y = []
        left_aside_adv_y = []
        single_origin_y = []
        single_adv_y = []
        vote_origin_y = []
        vote_adv_y = []
        indexes = np.arange(0,len(self.test_loader))
        adv_samples = np.array(self.attacker.adv_samples)
        adv_samples = adv_samples[:,0]
        num_samples = len(adv_samples)
        batches = num_samples // self.batch_size
        split_indexes = [self.batch_size*(i+1) for i in range(batches)]
        split_adv_samples = np.split(adv_samples,split_indexes)
        with torch.no_grad():
            for (index,(x,y)) in tqdm(zip(indexes,self.test_loader),total=len(indexes)):
                model_index = np.random.randint(0,len(self.model_pool))
                x, y = x.to(device), y.to(device)
                adv_x = split_adv_samples[index].tolist()
                adv_x = torch.cat(adv_x,dim=0)
                adv_x = adv_x.to(device)
                true_y.append(y)

                s_origin_y =  self.model_pool[model_index](x)
                s_origin_y = F.log_softmax(s_origin_y, dim=1)
                s_origin_y = s_origin_y.max(1, keepdim=False)[1]
                single_origin_y.append(s_origin_y)

                s_adv_y = self.model_pool[model_index](adv_x)
                s_adv_y = F.log_softmax(s_adv_y, dim=1)
                s_adv_y = s_adv_y.max(1, keepdim=False)[1]
                single_adv_y.append(s_adv_y)

                las_origin_y =  self.left_aside_model(x)
                las_origin_y = F.log_softmax(las_origin_y, dim=1)
                las_origin_y = las_origin_y.max(1, keepdim=False)[1]
                left_aside_origin_y.append(las_origin_y)

                las_adv_y = self.left_aside_model(adv_x)
                las_adv_y = F.log_softmax(las_adv_y, dim=1)
                las_adv_y = las_adv_y.max(1, keepdim=False)[1]
                left_aside_adv_y.append(las_adv_y)

                pred_y_list = []
                pred_adv_y_list = []
                for mod in self.model_pool:
                    pred_ori_y = mod(x)
                    pred_ori_y = F.log_softmax(pred_ori_y, dim=1)
                    pred_ori_y = pred_ori_y.max(1, keepdim=False)[1]
                    pred_y_list.append(pred_ori_y)

                    pred_adv_y = mod(adv_x)
                    pred_adv_y = F.log_softmax(pred_adv_y, dim=1)
                    pred_adv_y = pred_adv_y.max(1, keepdim=False)[1]
                    pred_adv_y_list.append(pred_adv_y)
                pred_y_list = torch.stack(pred_y_list,dim=-1)
                pred_adv_y_list = torch.stack(pred_adv_y_list,dim=-1)
                v_origin_y = pred_y_list.mode(dim=-1).values
                v_adv_y = pred_adv_y_list.mode(dim=-1).values
                vote_origin_y.append(v_origin_y)
                vote_adv_y.append(v_adv_y)



        true_y = torch.cat(true_y,dim=-1)
        single_origin_y = torch.cat(single_origin_y,dim=-1)
        single_adv_y =torch.cat(single_adv_y,dim=-1)
        left_aside_origin_y = torch.cat(left_aside_origin_y,dim=-1)
        left_aside_adv_y =torch.cat(left_aside_adv_y,dim=-1)
        vote_origin_y = torch.cat(vote_origin_y,dim=-1)
        vote_adv_y = torch.cat(vote_adv_y,dim=-1)

        non_defense_origin_acc = self.calc_acc(true_y,left_aside_origin_y)
        non_defense_adv_acc = self.calc_acc(true_y,left_aside_adv_y)

        single_origin_acc = self.calc_acc(true_y,single_origin_y)
        single_adv_acc = self.calc_acc(true_y,single_adv_y)

        vote_origin_acc = self.calc_acc(true_y, vote_origin_y)
        vote_adv_acc = self.calc_acc(true_y, vote_adv_y)

        decrease_acc_single = single_origin_acc - single_adv_acc
        decrease_acc_vote = vote_origin_acc - vote_adv_acc
        print(f"non defense original acc is {non_defense_origin_acc:.4f},"
              f"non defense adv samples acc is {non_defense_adv_acc:.4f},"
              f"single model original acc is {single_origin_acc:.4f},"
              f"single model adv samples acc is {single_adv_acc:.4f}"
              f"decreased {decrease_acc_single:.4f} acc"
              f"\n"
              f"vote model original acc is {vote_origin_acc:.4f},"
              f"vote model adv samples acc is {vote_adv_acc:.4f}"
              f"decreased {decrease_acc_vote:.4f} acc")

        return single_origin_acc,single_adv_acc,vote_origin_acc,vote_adv_acc


    def calc_acc(self,true_y,pred_y):
        crct_num = (true_y == pred_y).sum()
        return torch.true_divide(crct_num, len(true_y))



if __name__ == "__main__":
    base_classifier_dir = "model_files/best_model.pth"
    left_aside_model_dir = "model_files/model0_parameter.pkl"
    num_models = 20
    pool_dir = ["model_files/30_sample_rate/","model_files/40_sample_rate/","model_files/50_sample_rate/",
          "model_files/60_sample_rate/","model_files/100_sample_rate/"]
    sample_rate = ["0.3","0.4","0.5","0.6","1.0"]
    evaluator = Evaluator(fgsm_eps=0.02, base_classifier_dir=base_classifier_dir,
                          left_aside_model_dir=left_aside_model_dir)
    for i,pd in enumerate(pool_dir):
        print(f"start {sample_rate[i]} sample rate pool evaluation")
        torch.cuda.empty_cache()
        evaluator.reset_model_pool(num_models=num_models, pool_dir=pd)
        evaluator.model_pool_eval()
        print(f"finish {sample_rate[i]} sample rate pool evaluation")

