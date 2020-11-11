# Author : Zhihao Wang
# Date : 10/11/2020

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from attackers.attacker_base import *
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset
from PIL import Image
from torch.utils.data import DataLoader
from torch.autograd import Variable
import copy
from torch.autograd.gradcheck import zero_gradients
import torch
import torchvision.transforms as T


class UAA(AttackerBase):
    def __init__(self):
        super(UAA, self).__init__()
        self.last_best_fooling_rate = 0.1
        self.input_size = 32
        self.delta = 0.2
        self.max_iter = 20
        self.xi = 10
        self.p = np.inf
        self.num_classes = 10
        self.overshoot = 0.2
        self.max_iter_df = 20
        self.initial_fooling_rate = 0.1
        self.batch_size = 256
        self.label_nums = 10
        self.save_path = f"perturbation/universal_pert/"
        self.load_pert = False
        self.load_pert_path = None
        self.transform = T.Compose([T.ToTensor()])
        self.universal_pert = np.zeros((self.input_size, self.input_size, 3))
        if self.load_pert:
            self.universal_pert = np.load(self.load_pert_path)

    def train_adversarial_samples(self, model, train_set, **kwargs):
        dev_set = kwargs['dev_set']
        order = np.arange(len(train_set))
        fooling_rate = 0.
        iter = 0
        while fooling_rate < 1 - self.delta and iter < self.max_iter:
            np.random.shuffle(order)
            print("Starting pass number ", iter)
            model.eval()
            for k in tqdm(order, ascii=True, total=len(order)):
                cur_img, _ = train_set[k]
                cur_img1 = self.transform(cur_img).unsqueeze(0).cuda()
                r2 = int(model(cur_img1).max(1)[1])
                torch.cuda.empty_cache()

                perte_img, _ = train_set[k]
                perte_img = np.array(perte_img)
                perte_img = Image.fromarray(perte_img + self.universal_pert.astype(np.uint8))
                perte_img1 = self.transform(perte_img).unsqueeze(0).cuda()
                r1 = int(model(perte_img1).max(1)[1])
                torch.cuda.empty_cache()

                if r1 == r2:
                    # print(">> k =", np.where(k == order)[0][0], ', pass #', iter,end='\n')
                    dr, iter_k, label, k_i, pert_image = self.deepfool(perte_img1[0], model,
                                                                       num_classes=self.num_classes,
                                                                       overshoot=self.overshoot,
                                                                       max_iter=self.max_iter_df)

                    if iter_k < self.max_iter - 1:
                        self.universal_pert[:, :, 0] += dr[0, 0, :, :]
                        self.universal_pert[:, :, 1] += dr[0, 1, :, :]
                        self.universal_pert[:, :, 2] += dr[0, 2, :, :]
                        self.universal_pert = self.project_lp(self.universal_pert, self.xi, self.p)
            iter += 1
            fooling_rate = self.eval_adversarial_samples(model=model, dev_set=dev_set)
            if fooling_rate >= self.last_best_fooling_rate:
                np.save(self.save_path + 'universal_pert' + str(iter) + '_' +
                        str(round(fooling_rate, 4)), self.universal_pert)

    def deepfool(self, image, model, num_classes, overshoot, max_iter):

        """
           :param image: Image of size HxWx3
           :param net: network (input: images, output: values of activation **BEFORE** softmax).
           :param num_classes: num_classes (limits the number of classes to test against, by default = 10)
           :param overshoot: used as a termination criterion to prevent vanishing updates (default = 0.02).
           :param max_iter: maximum number of iterations for deepfool (default = 50)
           :return: minimal perturbation that fools the classifier, number of iterations that it required,
           new estimated_label and perturbed image
        """
        is_cuda = torch.cuda.is_available()
        if is_cuda:
            image = image.cuda()
            model = model.cuda()

        f_image = model.forward(Variable(image[None, :, :, :], requires_grad=True)).data.cpu().numpy().flatten()
        I = f_image.argsort()[::-1]  # descending order of results

        I = I[0:num_classes]
        label = I[0]

        input_shape = image.cpu().numpy().shape
        pert_image = copy.deepcopy(image)
        w = np.zeros(input_shape)
        r_tot = np.zeros(input_shape)

        loop_i = 0

        x = Variable(pert_image[None, :], requires_grad=True)
        fs = model.forward(x)  # output of perturbated image
        k_i = label

        while k_i == label and loop_i < max_iter:  # loop until label != ki (get different output as before)

            pert = np.inf
            fs[0, I[0]].backward(retain_graph=True)  # get original grad
            grad_orig = x.grad.data.cpu().numpy().copy()

            for k in range(1, num_classes):  # find minimal pert_k and w_k, to send data to its decision boundry
                zero_gradients(x)

                fs[0, I[k]].backward(retain_graph=True)  # get wrong class grad
                cur_grad = x.grad.data.cpu().numpy().copy()

                # set new w_k and new f_k
                w_k = cur_grad - grad_orig  # difference of grad
                f_k = (fs[0, I[k]] - fs[0, I[0]]).data.cpu().numpy()  # difference of label score

                pert_k = abs(f_k) / np.linalg.norm(w_k.flatten())  # delta y / delta grad = pert

                # determine which w_k to use
                if pert_k < pert:
                    pert = pert_k
                    w = w_k

            # compute r_i and r_tot
            # Added 1e-4 for numerical stability
            r_i = (pert + 1e-4) * w / np.linalg.norm(w)
            r_tot = np.float32(r_tot + r_i)

            # overshoot for scale
            if is_cuda:
                pert_image = image + (1 + overshoot) * torch.from_numpy(r_tot).cuda()
            else:
                pert_image = image + (1 + overshoot) * torch.from_numpy(r_tot)

            x = Variable(pert_image, requires_grad=True)
            fs = model.forward(x)
            k_i = np.argmax(fs.data.cpu().numpy().flatten())  # test new output, if not qualified, loop again

            loop_i += 1

        return (1 + overshoot) * r_tot, loop_i, label, k_i, pert_image

    def project_lp(self, v, xi, p):
        if p == 2:
            pass
        elif p == np.inf:
            v = np.sign(v) * np.minimum(abs(v), xi)
        else:
            raise ValueError("Values of a different from 2 and Inf are currently not surpported...")

        return v

    def eval_adversarial_samples(self, model, dev_set):
        with torch.no_grad():
            # Compute fooling_rate
            dev_set_wrap = MSDataSet(self.transform, dev_set)

            est_labels_orig = torch.tensor(np.zeros(0, dtype=np.int64))
            est_labels_pert = torch.tensor(np.zeros(0, dtype=np.int64))

            dev_set_pert = copy.deepcopy(dev_set_wrap)
            dev_loader_orig = DataLoader(dataset=dev_set_wrap, batch_size=self.batch_size, pin_memory=True)

            dev_set_pert.set_perterb(self.universal_pert)
            dev_loader_perterb = DataLoader(dataset=dev_set_pert, batch_size=self.batch_size, pin_memory=True)

            for batch_idx, (_, inputs, _) in enumerate(dev_loader_orig):
                inputs = inputs.cuda()
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                est_labels_orig = torch.cat((est_labels_orig, predicted.cpu()))
            torch.cuda.empty_cache()

            for batch_idx, (_, inputs, _) in enumerate(dev_loader_perterb):
                inputs = inputs.cuda()
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                est_labels_pert = torch.cat((est_labels_pert, predicted.cpu()))
            torch.cuda.empty_cache()

            fooling_rate = float(torch.sum(est_labels_orig != est_labels_pert)) / float(len(dev_set))
            print("FOOLING RATE: ", fooling_rate)

            return fooling_rate


class MSDataSet(Dataset):
    def __init__(self, transform, dataset):
        super(MSDataSet, self).__init__()
        self.transform = transform
        self.perterb = 0
        self.data = dataset

    def __len__(self):
        return len(self.data)

    def set_perterb(self, v):
        self.perterb = v

    def __getitem__(self, index):
        img, label = self.data[index]
        img = np.array(img)
        img = Image.fromarray(np.clip(img + self.perterb, 0, 255).astype(np.uint8))
        img = self.transform(img)
        return index, img, label
