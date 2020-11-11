import sys
import torch
import torchvision
from attackers.fgsm import FGSM
from attackers.uaa import UAA
from attackers.lbfgs import LBFGS
from defenders.fgsm_AT import FGSM_AT

def conduct(name, model, train_raw, test_raw):
    if name == 'FGSM':
        fgsm = FGSM(0.2)
        fgsm.train_adversarial_samples(model, test_raw)
        attack_rate = fgsm.eval_adversarial_samples(model, None)
        print("Attacked Success Rate:", attack_rate)
    elif name == 'UAA':
        uaa = UAA()
        uaa.train_adversarial_samples(model=model, train_set=train_raw, dev_set=test_raw)
        attack_rate = uaa.eval_adversarial_samples(model=model, dev_set=test_raw)
        print("Attacked Success Rate:", attack_rate)
    elif name == 'LBFGS':
        lbfgs = LBFGS(num_instances=10000, input_size=[3, 32, 32], c=0.5)
        lbfgs.train_adversarial_samples(model, test_raw)
        attack_rate = lbfgs.eval_adversarial_samples(model, test_raw)
        print("Attacked Success Rate:", attack_rate)
    elif name == 'FGSM_AT':
        AT_defense = FGSM_AT()
        defended_model = AT_defense.train_adversarial_samples(model)
        accuracy, attack_acc = AT_defense.eval_adversarial_samples(defended_model)
        print("Original Acc:", accuracy)
        print("Attacked Acc:", attack_acc)


def main():
    # Read the user input and check its validity
    if len(sys.argv) == 1:
        raise Exception("\nPlease indicate which attack or defense to conduct.\n"
                        "Attacks: 'FGSM', 'UAA', 'LBFGS'\n"
                        "Defenses: 'FGSM-AT'")
    elif len(sys.argv) > 2:
        raise Exception("\nYou can only pass one arugument to this program.\n"
                        "Attacks: 'FGSM', 'UAA', 'LBFGS'\n"
                        "Defenses: 'FGSM-AT'")
    else:
        ls_attacks = ['FGSM', 'UAA', 'LBFGS']
        ls_defenses = ['FGSM-AT']
        tag = sys.argv[1]
        if tag not in ls_attacks and tag not in ls_defenses:
            raise Exception("\nYour tag is wrong! Please select the correct attack or defense\n"
                            "Attacks: 'FGSM', 'UAA', 'LBFGS' \n"
                            "Defenses: 'FGSM-AT'")

    # Read the raw CIFAR-10 data
    train_raw = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=None)
    test_raw = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=None)

    # Load the model
    model = torch.load("./model_files/best_model.pth")

    # Conduct attack or defense corresponding the user input
    conduct(tag, model, train_raw, test_raw)


if __name__ == '__main__':
    main()


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