import torch
import torchvision
import torchvision.transforms as transforms
from attackers.fgsm import FGSM
from attackers.uaa import UAA

def main():
    # test_transform = transforms.Compose([transforms.ToTensor()])
    train_raw = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=None)
    test_raw = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=None)
    print(test_raw)
    print(test_raw[0])

    # # shenyi
    # model = torch.load("./model_files/best_model.pth")
    # fsgm = FGSM(0.2)
    # fsgm.train_adversarial_samples(model, test_raw)
    # attack_rate1 = fsgm.eval_adversarial_samples(model, None)
    # print("Attacked Acc:", attack_rate1)  # 72%

    # wangzhihao
    model = torch.load("./model_files/best_model.pth")
    uaa = UAA()
    uaa.train_adversarial_samples(model=model, train_set=train_set, dev_set=dev_set)
    attack_rate = uaa.eval_adversarial_samples(model=model, dev_set=dev_set)
    print("Attacked Acc:", attack_rate)  # 72%

    # # fengjingrong
    # model = torch.load("./model_files/best_model.pth")
    # lbfgs = LBFGS()
    # lbfgs.train_adversarial_samples(model, train_raw, test_raw)
    # print("Adv_samples:", adv_samples)
    # attack_rate = lbfgs.eval_adversarial_samples(model, test_raw)
    # print("Attacked Acc:", attack_rate)

    # plt.figure(figsize=(8, 10))
    # for j in range(len(adv_examples)):
    #     plt.subplot(1, 5)
    #     plt.xticks([], [])
    #     plt.yticks([], [])
    #     if j == 0:
    #         plt.ylabel("Eps: {}".format(0.2), fontsize=14)
    #     orig, adv, ex = adv_examples[j]
    #     plt.title("{} -> {}".format(orig, adv))
    #     plt.imshow(ex, cmap="gray")
    # plt.tight_layout()
    # plt.show()

    # AT_defense = AT()
    # vgg = VGG()
    # defended_model = AT.train_adversarial_samples(vgg, train_raw)  # return a pretrained defended model
    # fsgm = FGSM(0.2)
    # fsgm.train_adversarial_samples(defended_model, test_raw)
    # print("Adv_samples:", adv_samples)
    # attack_rate2 = fsgm.eval_adversarial_samples(defended_model, None)
    # print("Attacked Acc:", attack_rate2)  # 64%

if __name__ == '__main__':
    main()
